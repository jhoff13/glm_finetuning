#!/usr/bin/env python
"""
LoRA MLM finetuning of gLM2-650M on a Pfam family, with live interface-precision
tracking for a target PDB pair.

Adapted from https://github.com/jhoff13/glm_finetuning (`gLM2_trainer_v0.py`), with:
  * masking restricted to amino-acid / nucleotide tokens (never `<+>`/`<->`/pad),
    BERT-style 80/10/10 corruption as in ProteinTTT (arXiv:2411.02109),
  * loss on masked positions only (labels = -100 elsewhere),
  * length-grouped dynamic padding + a real `attention_mask`,
  * per-eval logging of val MLM loss, target pseudo-perplexity and the *fast*
    categorical-Jacobian P@C of the target chain pair, so training can be run to
    saturation while watching the quantity the experiment is about.

Every checkpoint is a peft adapter dir that `pac_eval.py --lora` can score with the
slow categorical Jacobian.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

# gLM2 token ids.  Masking is restricted to these so the `<+>`/`<->` strand
# tokens and padding are never corrupted.
MODEL_NAME = os.environ.get("GLM2_MODEL", "tattabio/gLM2_650M")
NUC_TOKENS = list(range(29, 33))          # a t c g
AA_TOKENS = list(range(4, 24))            # 20 canonical AAs
ALL_TOKENS = NUC_TOKENS + AA_TOKENS
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Live interface-precision tracking (--track_pairs) needs the scoring stack from
# the research repo -- biopython, a PDB, and pac_eval.py.  Training itself does
# not, so the import is optional: without it the trainer runs exactly as normal
# and only the P@C/P@L probe is unavailable.
try:
    from pac_eval import (categorical_jacobian, chain_seqs_and_coords,  # noqa: E402
                          inter_contact_map, intra_contact_map, resolve_pdb,
                          score_intra, score_pair)
    HAVE_PAC_EVAL = True
except ImportError as _e:                                            # noqa: E402
    HAVE_PAC_EVAL = False
    _PAC_EVAL_ERR = str(_e)

import wandb  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

MASKABLE = set(AA_TOKENS) | set(NUC_TOKENS)


# ── data ──────────────────────────────────────────────────────────────────────
def read_fasta_lines(path):
    """One-line FASTA whose sequence line is a literal gLM2 token string."""
    seqs, name = [], None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                name = line[1:]
            elif line.strip():
                seqs.append(line.strip())
    return seqs


class MLMDataset(Dataset):
    def __init__(self, records, tokenizer, max_length, mask_prob=0.30, seed=0,
                 gene_mask_frac=0.0):
        self.tok = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        # probability that an example is masked as a WHOLE GENE instead of at
        # the per-residue rate.  0.0 reproduces the original behaviour exactly.
        self.gene_mask_frac = gene_mask_frac
        self.ids = [tokenizer.encode(s)[:max_length] for s in tqdm(records, desc="tokenize")]
        self.rng = random.Random(seed)

    @staticmethod
    def _gene_runs(maskable):
        """Maximal contiguous runs of maskable positions == genes, since the
        `<+>`/`<->` strand tokens that delimit them are not maskable."""
        runs, start = [], None
        for i, m in enumerate(maskable.tolist()):
            if m and start is None:
                start = i
            elif not m and start is not None:
                runs.append((start, i)); start = None
        if start is not None:
            runs.append((start, len(maskable)))
        return runs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        ids = torch.tensor(self.ids[i], dtype=torch.long)
        labels = torch.full_like(ids, -100)

        maskable = torch.tensor([int(t) in MASKABLE for t in ids])
        idx = torch.where(maskable)[0]
        n_mask = max(1, int(round(len(idx) * self.mask_prob)))
        if len(idx) == 0:
            return dict(input_ids=ids, labels=labels)

        # ── whole-gene masking branch ────────────────────────────────────────
        # Mask one entire gene (chosen uniformly at random) so the model must
        # reconstruct it from its partner alone.  Pure <mask>, no 80/10/10:
        # the point is total ablation of the gene, not local corruption.
        if self.gene_mask_frac > 0 and self.rng.random() < self.gene_mask_frac:
            runs = self._gene_runs(maskable)
            if len(runs) >= 2:                       # need a partner to infer from
                s, e = runs[self.rng.randrange(len(runs))]
                span = torch.arange(s, e)
                span = span[maskable[span]]
                ids = ids.clone()
                labels[span] = ids[span]
                ids[span] = self.tok.mask_token_id
                return dict(input_ids=ids, labels=labels)
            # single gene -> fall through to per-residue masking

        pick = idx[torch.randperm(len(idx))[:n_mask]]

        labels[pick] = ids[pick]
        # BERT / ProteinTTT corruption: 80% <mask>, 10% random aa/nuc, 10% keep
        r = torch.rand(len(pick))
        ids = ids.clone()
        ids[pick[r < 0.8]] = self.tok.mask_token_id
        rnd = pick[(r >= 0.8) & (r < 0.9)]
        if len(rnd):
            ids[rnd] = torch.tensor(ALL_TOKENS)[torch.randint(len(ALL_TOKENS), (len(rnd),))]
        return dict(input_ids=ids, labels=labels)


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        lab[i, :n] = b["labels"]
        att[i, :n] = True
    return dict(input_ids=ids, labels=lab, attention_mask=att)


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Shuffle, then sort within large chunks so batches are length-homogeneous."""

    def __init__(self, lengths, batch_size, mega=50, seed=0):
        self.lengths, self.bs, self.mega, self.seed = lengths, batch_size, mega, seed
        self.epoch = 0

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        idx = g.permutation(len(self.lengths))
        chunk = self.bs * self.mega
        out = []
        for s in range(0, len(idx), chunk):
            part = idx[s : s + chunk]
            out += list(part[np.argsort([self.lengths[i] for i in part])])
        batches = [out[s : s + self.bs] for s in range(0, len(out), self.bs)]
        g.shuffle(batches)
        self.epoch += 1
        return iter([i for b in batches for i in b])


# ── target-side metrics ───────────────────────────────────────────────────────
@torch.no_grad()
def pseudo_ppl(model, tok, sequence, stride=8):
    """Exact stride-k pseudo-perplexity: mask every k-th position, k passes."""
    ids = torch.tensor(tok.encode(sequence), dtype=torch.long)
    maskable = torch.tensor([int(t) in MASKABLE for t in ids])
    total, n = 0.0, 0
    lf = nn.CrossEntropyLoss(reduction="sum")
    for off in range(stride):
        pos = torch.where(maskable)[0][off::stride]
        if not len(pos):
            continue
        x = ids.clone()
        x[pos] = tok.mask_token_id
        logits = model(input_ids=x.unsqueeze(0).to(DEVICE)).logits[0].float().cpu()
        total += lf(logits[pos], ids[pos]).item()
        n += len(pos)
    return math.exp(total / max(n, 1))


class TargetScorer:
    """Fast-Jacobian interface P@C *and* intra-chain P@L for one chain pair.

    Both metrics come from a single jacobian pass over `<+>A<+>B`: the inter-chain
    block gives P@C, the two diagonal blocks give per-chain contact recovery, so
    adding P@L costs no extra forward passes.
    """

    def __init__(self, pdb, pair, seq_lut, tok, threshold=8.0, atom="CB", min_sep=6):
        a, b = pair.split("-")
        self.a, self.b = a, b
        path = resolve_pdb(pdb)
        self.pdb_seqs, coords = chain_seqs_and_coords(path, atom=atom)
        self.gt = inter_contact_map(coords, a, b, threshold)
        self.gt_intra = {c: intra_contact_map(coords, c, threshold) for c in (a, b)}
        self.min_sep = min_sep
        stem = Path(path).stem.upper().split(".")[0]
        raw_a = seq_lut.get(f"{stem}_{a}", "<+>" + self.pdb_seqs[a])
        raw_b = seq_lut.get(f"{stem}_{b}", "<+>" + self.pdb_seqs[b])
        self.qry_a, self.qry_b = raw_a[3:], raw_b[3:]
        self.sequence = raw_a + raw_b
        self.tok = tok
        self.K = int(self.gt.sum())

    def __call__(self, model, readout="logits", fast=True, batch_size=24):
        """Return {'pac': float, 'pl_<chain>': float, ...}."""
        _, contact, _ = categorical_jacobian(self.sequence, model, self.tok, fast=fast,
                                             batch_size=batch_size, quiet=True,
                                             readout=readout)
        score, _, _ = score_pair(contact, len(self.qry_a), len(self.qry_b), self.gt,
                                 self.pdb_seqs[self.a], self.pdb_seqs[self.b],
                                 self.qry_a, self.qry_b)
        out = {"pac": score}
        for ch, off, qs in ((self.a, 1, self.qry_a),
                            (self.b, 2 + len(self.qry_a), self.qry_b)):
            v, _, _ = score_intra(contact, off, len(qs), self.gt_intra[ch],
                                  self.pdb_seqs[ch], qs, min_sep=self.min_sep)
            out[f"pl_{ch}"] = v
        return out


# ── training ──────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--name", required=True, help="wandb run name / output subdir")
    p.add_argument("-i", "--data_dir", required=True, help="dir with train.fasta / val.fasta")
    p.add_argument("-o", "--out_dir", required=True)
    p.add_argument("--project", default="3IP4_gLM2_pfam_finetuning")
    p.add_argument("--offline", action="store_true")
    p.add_argument("-b", "--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("-r", "--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("-d", "--dropout", type=float, default=0.0,
                   help="LoRA dropout; 0 by default because overfitting is the goal")
    p.add_argument("--targets", nargs="+", default=["wqkv", "wo", "w1", "w2", "w3"])
    p.add_argument("-l", "--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--schedule", default="constant", choices=["constant", "cosine"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=3000, help="optimizer steps")
    p.add_argument("--max_length", type=int, default=640)
    p.add_argument("--mask_prob", type=float, default=0.30, help="gLM2 pretraining rate")
    p.add_argument("--gene_mask_frac", type=float, default=0.0,
                   help="fraction of TRAINING EXAMPLES masked as a whole gene "
                        "(random gene per example) instead of per-residue")
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--val_batches", type=int, default=25)
    p.add_argument("--mode", default="lora",
                   choices=["lora", "full", "last_n", "head_select"],
                   help="lora = LoRA adapters; full = all weights; "
                        "last_n = freeze all but the last N encoder layers (+ lm_head)")
    p.add_argument("--unfreeze_last", type=int, default=4,
                   help="for --mode last_n: how many top encoder layers to train")
    p.add_argument("--head_spec", default="31:6,7,4;32:0,10,8,5,4,1",
                   help="for --mode head_select: 'layer:h,h,...;layer:h,...'. "
                        "Default = the 9 heads in layers 31-32 above the P@L gap "
                        "in 5_coevo_heads/sheets/head_screen_ranked_pl.csv")
    p.add_argument("--full_ft", action="store_true",
                   help="deprecated alias for --mode full")
    p.add_argument("--min_sep", type=int, default=6,
                   help="minimum |i-j| for intra-chain P@L candidate pairs")
    p.add_argument("--seed", type=int, default=42)
    # target tracking
    p.add_argument("--pdb", default="3IP4")
    p.add_argument("--track_pairs", nargs="+", default=["A-C"])
    p.add_argument("--seqs_csv",
                   default=os.environ.get("GLM2_SEQS_CSV"),
                   help="table of query sequences for --track_pairs; only needed "
                        "when live P@C tracking is on")
    p.add_argument("--no_track_pac", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if args.mode == "head_select":
        # Per-head training needs the eager-softmax attention, not flash/SDPA.
        # The wrapper reproduces the stock parameter names exactly (verified
        # bit-identical logits), so checkpoints stay loadable by pac_eval.
        sys.path.insert(0, os.environ.get(
            "GLM2_MODELS_DIR", str(SCRIPTS / "models" / "gLM2")))
        from glm2_headselect_for_mlm import gLM2HeadSelectForMaskedLM
        model = gLM2HeadSelectForMaskedLM.from_stock(MODEL_NAME).to(DEVICE)
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32).to(DEVICE)
    model.gradient_checkpointing_disable() if hasattr(model, "gradient_checkpointing_disable") else None

    if args.full_ft:
        args.mode = "full"

    if args.track_pairs and not args.no_track_pac and not HAVE_PAC_EVAL:
        raise SystemExit(
            f">> live P@C tracking needs pac_eval.py on the path ({_PAC_EVAL_ERR}).\n"
            ">> Either drop --track_pairs / pass --no_track_pac to train without it,\n"
            ">> or copy pac_eval.py (and its `utils` dependency) alongside this file.")

    n_all = sum(p.numel() for p in model.parameters())
    if args.mode == "full":
        print(">> full finetuning: all parameters trainable")
        trainable = list(model.parameters())
    elif args.mode == "last_n":
        # gLM2 blocks live at glm2.encoder.layers[0..n-1]; train the top N of them
        # plus lm_head (norm + output projection), freeze embeddings and lower blocks.
        n_layers = len(model.glm2.encoder.layers)
        keep = set(range(n_layers - args.unfreeze_last, n_layers))
        for name, prm in model.named_parameters():
            in_top = any(f"encoder.layers.{i}." in name for i in keep)
            prm.requires_grad = bool(in_top or name.startswith("lm_head"))
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f">> last_n finetuning: layers {sorted(keep)} of {n_layers} + lm_head")
    elif args.mode == "head_select":
        heads = {}
        for part in args.head_spec.split(";"):
            if not part.strip():
                continue
            lay, hs = part.split(":")
            heads[int(lay)] = [int(h) for h in hs.split(",") if h != ""]
        # snapshot before training so the freeze can be *checked*, not assumed
        head_ref = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        masks, n_free = model.freeze_to_heads(heads)
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f">> head_select finetuning: {heads}")
        print(f">>   {n_free:,} weight entries across {len(masks)} tensors "
              f"(wqkv rows + wo columns of those heads only)")
    else:
        cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                         target_modules=args.targets, bias="none")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        trainable = [p for p in model.parameters() if p.requires_grad]
    n_train_params = sum(p.numel() for p in trainable)
    if args.mode == "head_select":
        # `trainable` holds whole tensors; only the masked entries can move, so
        # reporting tensor sizes here would overstate capacity by ~10x
        n_train_params = n_free
    print(f">> trainable {n_train_params/1e6:.1f}M / {n_all/1e6:.1f}M "
          f"({100*n_train_params/n_all:.2f}%)  mode={args.mode}  lr={args.lr:g}")

    train_recs = read_fasta_lines(Path(args.data_dir) / "train.fasta")
    val_recs = read_fasta_lines(Path(args.data_dir) / "val.fasta")
    print(f">> train {len(train_recs)} | val {len(val_recs)}")
    train_ds = MLMDataset(train_recs, tok, args.max_length, args.mask_prob, seed=args.seed,
                          gene_mask_frac=args.gene_mask_frac)
    val_ds = MLMDataset(val_recs, tok, args.max_length, args.mask_prob, seed=args.seed + 1)

    coll = lambda b: collate(b, tok.pad_token_id)
    sampler = LengthGroupedSampler([len(i) for i in train_ds.ids], args.batch_size, seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          collate_fn=coll, num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll,
                        num_workers=2)

    seq_lut = {}
    if os.path.isfile(args.seqs_csv):
        import pandas as pd
        df = pd.read_csv(args.seqs_csv)
        for _, r in df.iterrows():
            seq_lut[r.seq_a_name] = r.seq_a
            seq_lut[r.seq_b_name] = r.seq_b
    scorers = {} if args.no_track_pac else {
        p: TargetScorer(args.pdb, p, seq_lut, tok, min_sep=args.min_sep)
        for p in args.track_pairs
    }
    for p, s in scorers.items():
        print(f">> tracking P@C for {args.pdb} {p} (K={s.K}, L={len(tok.encode(s.sequence))})")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.98), eps=1e-8)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        if args.schedule == "cosine":
            t = (step - args.warmup) / max(1, args.max_steps - args.warmup)
            return args.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        return args.lr

    run = wandb.init(project=args.project, name=args.name,
                     mode="offline" if args.offline else "online",
                     dir=str(out_dir),
                     config=dict(vars(args), n_trainable=n_train_params,
                                 n_train=len(train_recs), n_val=len(val_recs),
                                 model=MODEL_NAME))
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    scaler = torch.cuda.amp.GradScaler()

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        for i, batch in enumerate(val_dl):
            if i >= args.val_batches:
                break
            with torch.cuda.amp.autocast():
                out = model(input_ids=batch["input_ids"].to(DEVICE),
                            attention_mask=batch["attention_mask"].to(DEVICE),
                            labels=batch["labels"].to(DEVICE))
            k = int((batch["labels"] != -100).sum())
            tot += out.loss.item() * k
            n += k
        metrics = {"val/mlm_loss": tot / max(n, 1), "val/ppl": math.exp(tot / max(n, 1))}
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        for pair, sc in scorers.items():
            r = sc(base, readout="logits")
            metrics[f"target/pac_fast_logits_{pair}"] = r["pac"]
            for ch in (sc.a, sc.b):
                metrics[f"target/pl_fast_logits_{ch}"] = r[f"pl_{ch}"]
            metrics[f"target/pseudo_ppl_{pair}"] = pseudo_ppl(base, tok, sc.sequence)
        model.train()
        return metrics

    print(">> step 0 eval")
    m0 = evaluate()
    print("   " + "  ".join(f"{k}={v:.4f}" for k, v in m0.items()))
    run.log({**m0, "step": 0, "lr": 0.0})

    model.train()
    step, micro, t0 = 0, 0, time.time()
    run_loss, run_n = 0.0, 0
    history = [dict(step=0, **m0)]
    pbar = tqdm(total=args.max_steps, desc="train")
    done = False
    while not done:
        for batch in train_dl:
            with torch.cuda.amp.autocast():
                out = model(input_ids=batch["input_ids"].to(DEVICE),
                            attention_mask=batch["attention_mask"].to(DEVICE),
                            labels=batch["labels"].to(DEVICE))
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()
            run_loss += out.loss.item()
            run_n += 1
            micro += 1
            if micro % args.grad_accum:
                continue

            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            pbar.update(1)

            if step % 10 == 0:
                run.log({"train/mlm_loss": run_loss / run_n,
                         "train/ppl": math.exp(min(run_loss / run_n, 20)),
                         "train/grad_norm": float(gnorm), "lr": lr_at(step),
                         "step": step, "epoch": step * args.batch_size * args.grad_accum / len(train_ds)})
                run_loss, run_n = 0.0, 0

            if step % args.eval_every == 0 or step == args.max_steps:
                m = evaluate()
                run.log({**m, "step": step})
                history.append(dict(step=step, **m))
                pbar.write(f"[{step}] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
                import pandas as pd
                pd.DataFrame(history).to_csv(out_dir / "eval_history.csv", index=False)

            if step % args.save_every == 0 or step == args.max_steps:
                ck = out_dir / f"step_{step:05d}"
                if args.mode == "lora":
                    model.save_pretrained(str(ck))
                elif args.mode in ("last_n", "head_select"):
                    # only the unfrozen tensors; pac_eval --ckpt loads with strict=False
                    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
                          if dict(model.named_parameters()).get(k) is not None
                          and dict(model.named_parameters())[k].requires_grad}
                    ck = ck.with_suffix(".pt")
                    torch.save(sd, ck)
                else:
                    ck = ck.with_suffix(".pt")
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ck)
                pbar.write(f"   saved {ck}")

            if step >= args.max_steps:
                done = True
                break
    pbar.close()
    run.summary["minutes"] = (time.time() - t0) / 60
    run.finish()
    if args.mode == "head_select":
        # Gradient masking is only as good as its proof: confirm against the
        # pre-training snapshot that nothing outside the head slices moved.
        moved_in, moved_out = model.verify_frozen(head_ref)
        print(f">> head_select check: {moved_in:,} entries moved inside the masks; "
              f"{len(moved_out)} tensors moved outside")
        assert not moved_out, f"weights leaked outside the head masks: {moved_out[:5]}"

    print(f">> done in {(time.time()-t0)/60:.1f} min; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
