#!/usr/bin/env python
"""
Interface precision (P@C) for gLM2 categorical-Jacobian contact maps.

Ground truth is the *Cbeta* (CA fallback for Gly) 8 A inter-chain contact map of a
PDB entry -- this is what reproduces the denominators in
`data/Yo_complex_alignments/sheets/interface_precision_all_methods.tsv`
(the example notebooks used CA, which gives a different K).

Model sequences are aligned to the PDB chain sequences and both contact maps are
gap-padded before scoring, so full-length UniProt sequences can be scored against
partially-observed crystal chains.

Usage
-----
  # base model, slow categorical jacobian, all three 3IP4 pairs
  python pac_eval.py --pdb 3IP4 --pairs A-C A-B B-C -o runs/base

  # a LoRA checkpoint
  python pac_eval.py --pdb 3IP4 --pairs A-C --lora runs/ft/pfam_AC/step_500 -o runs/ft_ac
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser, MMCIFParser, is_aa
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.Align import PairwiseAligner
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Repo root for data lookups.  Defaults to this file's own directory so the
# vendored `utils/` package next to it is importable standalone; point
# GLM2_REPO at the research repo to reuse its data/ tree instead.
REPO = Path(os.environ.get("GLM2_REPO", Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))
from utils import Score_PatC, get_offdiag_dict, jac_to_contact, matrix_to_df  # noqa: E402

MODEL_NAME = "tattabio/gLM2_650M"
NUC_TOKENS = list(range(29, 33))          # a t c g
AA_TOKENS = list(range(4, 24))            # 20 canonical AAs
ALL_TOKENS = NUC_TOKENS + AA_TOKENS
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"


# ── PDB ground truth ──────────────────────────────────────────────────────────
def _parse(pdb_path):
    ext = str(pdb_path).split(".")[-1].lower()
    parser = MMCIFParser(QUIET=True) if ext in ("cif", "mmcif") else PDBParser(QUIET=True)
    return parser.get_structure("s", str(pdb_path))


def resolve_pdb(pdb, out_dir=None):
    """Local path for a PDB id, downloading the biological assembly if absent.

    `out_dir` defaults to $GLM2_PDB_DIR, else ./pdb next to this file."""
    if os.path.isfile(pdb):
        return pdb
    if out_dir is None:
        out_dir = os.environ.get(
            "GLM2_PDB_DIR", str(Path(__file__).resolve().parent / "pdb"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{pdb.upper()}.pdb1")
    if not os.path.isfile(path):
        os.system(f"wget -qnc https://files.rcsb.org/download/{pdb.upper()}.pdb1.gz -O {path}.gz")
        os.system(f"gunzip -f {path}.gz")
    return path


def chain_seqs_and_coords(pdb_path, atom="CB"):
    """Per chain: 1-letter sequence and one coordinate per standard residue.

    atom='CB' uses the Cbeta and falls back to Calpha (Gly / missing CB);
    atom='CA' uses Calpha only (the convention used in the example notebooks).
    """
    model = next(_parse(pdb_path).get_models())
    seqs, coords = {}, {}
    for chain in model:
        s, xyz = [], []
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            name = "CB" if (atom == "CB" and "CB" in res) else "CA"
            if name not in res:
                continue
            s.append(protein_letters_3to1.get(res.get_resname(), "X"))
            xyz.append(res[name].get_coord())
        if s:
            seqs[chain.get_id()] = "".join(s)
            coords[chain.get_id()] = np.asarray(xyz)
    return seqs, coords


def inter_contact_map(coords, a, b, threshold=8.0):
    d = np.linalg.norm(coords[a][:, None, :] - coords[b][None, :, :], axis=-1)
    return (d <= threshold).astype(np.int8)


def intra_contact_map(coords, chain, threshold=8.0):
    """Cbeta contact map of one chain against itself."""
    xyz = coords[chain]
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    return (d <= threshold).astype(np.int8)


# ── alignment / gap padding ───────────────────────────────────────────────────
def _align(seq_a, seq_b):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score, aligner.mismatch_score = 1, -1
    aligner.open_gap_score, aligner.extend_gap_score = -2, -0.5
    aln = aligner.align(seq_a, seq_b)[0]
    return np.array(list(aln[0])), np.array(list(aln[1]))


def align_pair_maps(pdb_seq_a, query_seq_a, pdb_seq_b, query_seq_b, gt_map, query_map):
    """Insert zero rows/cols so `gt_map` and `query_map` share an alignment frame.

    Returns (gt_aligned, query_aligned, n_aligned_positions).
    """
    def pad(mat, row_gaps, col_gaps):
        out = mat.astype(float)
        for i in sorted(row_gaps):
            out = np.insert(out, min(int(i), out.shape[0]), 0.0, axis=0)
        for j in sorted(col_gaps):
            out = np.insert(out, min(int(j), out.shape[1]), 0.0, axis=1)
        return out

    ref_a, qry_a = _align(pdb_seq_a, query_seq_a)
    ref_b, qry_b = _align(pdb_seq_b, query_seq_b)
    gt = pad(gt_map, np.where(ref_a == "-")[0], np.where(ref_b == "-")[0])
    qy = pad(query_map, np.where(qry_a == "-")[0], np.where(qry_b == "-")[0])
    assert gt.shape == qy.shape, f"{gt.shape} != {qy.shape}"
    return gt, qy, gt.shape


# ── model ─────────────────────────────────────────────────────────────────────
def load_model(lora=None, dtype=torch.float32, ckpt=None):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=dtype
    )
    if ckpt:
        # full-finetune / partial-unfreeze checkpoint: a plain state_dict.
        # strict=False so a trainable-subset checkpoint layers onto pretrained weights.
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f">> loaded ckpt {ckpt}  ({len(sd)} tensors, "
              f"{len(missing)} kept pretrained, {len(unexpected)} unexpected)")
        assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora, torch_dtype=dtype)
        model = model.merge_and_unload()
        print(f">> merged LoRA adapter: {lora}")
    return model.eval().to(DEVICE), tok


@torch.no_grad()
def categorical_jacobian(sequence, model, tok, fast=False, batch_size=24, quiet=False,
                         readout="logits", amp=True):
    """Categorical Jacobian of the model output w.r.t. every input token.

    fast=False (default, "slow"): substitute all 24 aa/nuc tokens at each position.
    fast=True:                    single <mask> substitution per position.

    readout='logits' is the standard categorical Jacobian (MLM logits, 24 aa/nuc
    channels).  readout='hidden' slices the *last hidden state* at the same channel
    indices -- this is what `scripts/cat_jack_v2.py` does (its `model(x)[0]` returns
    `last_hidden_state`, not logits) and it is the convention behind the archived
    `interface_precision_all_methods.tsv` gLM2 column, so it is kept as a control.
    """
    input_ids = torch.tensor(tok.encode(sequence), dtype=torch.long)
    tokens = tok.convert_ids_to_tokens(input_ids)
    ln = input_ids.shape[0]
    n_tok = len(ALL_TOKENS)

    def f(x):
        out = model(input_ids=x, output_hidden_states=(readout == "hidden"))
        t = out.logits if readout == "logits" else out.hidden_states[-1]
        return t[..., ALL_TOKENS].float().cpu()

    autocast = torch.cuda.amp.autocast(enabled=amp and DEVICE.type == "cuda")
    with autocast:
        x0 = input_ids.unsqueeze(0).to(DEVICE)
        fx = f(x0)[0]                                    # [L, n_tok]

        rows = 1 if fast else n_tok
        sub = torch.tensor([tok.mask_token_id]) if fast else torch.tensor(ALL_TOKENS)
        fx_h = torch.zeros((ln, rows, ln, n_tok))

        with tqdm(total=ln, bar_format=BAR, disable=quiet) as pbar:
            for n in range(ln):
                x_h = input_ids.unsqueeze(0).repeat(rows, 1)
                x_h[:, n] = sub
                out = []
                for s in range(0, rows, batch_size):
                    out.append(f(x_h[s : s + batch_size].to(DEVICE)))
                fx_h[n] = torch.cat(out, 0)
                pbar.update(1)

    jac = fx_h - fx

    # Zero cross-modality entries (aa position x nucleotide token and vice versa).
    # In fast mode the [L,1,L,n] jacobian broadcasts against the [L,n,L,n] mask,
    # replicating the masked column n_tok times -- this reproduces cat_jack_v2.py.
    is_nuc_pos = torch.isin(input_ids, torch.tensor(NUC_TOKENS)).view(-1, 1, 1, 1)
    is_aa_pos = torch.isin(input_ids, torch.tensor(AA_TOKENS)).view(-1, 1, 1, 1)
    is_nuc_tok = torch.isin(torch.tensor(ALL_TOKENS), torch.tensor(NUC_TOKENS)).view(1, -1, 1, 1)
    is_aa_tok = torch.isin(torch.tensor(ALL_TOKENS), torch.tensor(AA_TOKENS)).view(1, -1, 1, 1)
    keep = (is_nuc_pos & is_nuc_tok) | (is_aa_pos & is_aa_tok)
    jac = torch.where(keep, jac, torch.zeros((), dtype=jac.dtype))

    return jac.numpy(), jac_to_contact(jac.numpy()), tokens


# ── scoring ───────────────────────────────────────────────────────────────────
def score_pair(contact_full, la, lb, gt_map, pdb_seq_a, pdb_seq_b, qry_seq_a, qry_seq_b,
               n_lead_a=1, n_lead_b=1):
    """Splice the inter-chain block out of a full contact matrix and score P@C.

    Token layout of the scored sequence is  [lead_a] seq_a [lead_b] seq_b,
    where each lead is the strand token ('<+>'/'<->'), one token each.
    """
    r0 = n_lead_a
    c0 = n_lead_a + la + n_lead_b
    block = contact_full[r0 : r0 + la, c0 : c0 + lb]
    assert block.shape == (la, lb), f"{block.shape} != {(la, lb)}"

    gt, qry, _ = align_pair_maps(pdb_seq_a, qry_seq_a, pdb_seq_b, qry_seq_b, gt_map, block)
    ref_diag = get_offdiag_dict(np.array(gt.shape))
    gt_df = matrix_to_df(np.rint(gt).astype(int), ref_diag)
    return Score_PatC(qry, gt_df, ref_diag), block, gt


def score_intra(contact_full, offset, length, gt_intra, pdb_seq, qry_seq,
                min_sep=6, topk=None):
    """P@L for one chain's own (intra-chain) contacts.

    Standard single-protein contact-prediction convention: consider residue pairs
    with |i-j| >= `min_sep`, rank them by predicted coupling, take the top L
    (L = chain length by default) and report the fraction that are true Cbeta
    contacts.  `contact_full` is already symmetric and APC-corrected by
    `jac_to_contact`, so only the upper triangle is ranked.
    """
    block = contact_full[offset : offset + length, offset : offset + length]
    gt, qry, _ = align_pair_maps(pdb_seq, qry_seq, pdb_seq, qry_seq, gt_intra, block)
    iu = np.triu_indices(gt.shape[0], k=min_sep)
    pred, truth = qry[iu], np.rint(gt[iu]).astype(int)
    L = min(topk or length, len(pred))
    top = np.argsort(pred)[-L:]
    return float(truth[top].sum()) / L, L, int(truth[top].sum())


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", default="3IP4", help="PDB id or path [3IP4]")
    p.add_argument("--pairs", nargs="+", default=["A-C"], help="chain pairs, e.g. A-C A-B")
    p.add_argument("-o", "--out_dir", required=True)
    p.add_argument("--lora", default=None, help="LoRA adapter dir to merge before scoring")
    p.add_argument("--ckpt", default=None,
                   help="full/partial-finetune checkpoint (.pt state_dict) to load")
    p.add_argument("--min_sep", type=int, default=6,
                   help="minimum |i-j| for intra-chain P@L candidate pairs")
    p.add_argument("--no_pl", action="store_true", help="skip intra-chain P@L")
    p.add_argument("--tag", default=None, help="label for the results row [dirname]")
    p.add_argument("--fast", action="store_true", help="mask-only jacobian (fast control)")
    p.add_argument("--readout", default="logits", choices=["logits", "hidden"],
                   help="'logits' = standard cat-jac; 'hidden' = cat_jack_v2.py / TSV convention")
    p.add_argument("--no_amp", action="store_true", help="disable fp16 autocast")
    p.add_argument("--atom", default="CB", choices=["CB", "CA"], help="GT contact atom [CB]")
    p.add_argument("--threshold", type=float, default=8.0)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--seqs_csv",
                   default=os.environ.get(
                       "GLM2_SEQS_CSV",
                       str(REPO / "data/Yo_complex_alignments/sheets/all_pdb_catjac.csv")),
                   help="csv with seq_a/seq_b/seq_a_name/seq_b_name; falls back to PDB seqs")
    p.add_argument("--save_npz", action="store_true", help="save full contact matrices")
    return p.parse_args()


def main():
    args = get_args()
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or out_dir.name

    pdb_path = resolve_pdb(args.pdb)
    pdb_seqs, coords = chain_seqs_and_coords(pdb_path, atom=args.atom)
    print(f">> {pdb_path}: " + ", ".join(f"{k}:{len(v)}" for k, v in pdb_seqs.items()))

    # model input sequences: prefer the curated catjac csv, else the PDB sequence
    seq_lut = {}
    if args.seqs_csv and os.path.isfile(args.seqs_csv):
        df = pd.read_csv(args.seqs_csv)
        for _, r in df.iterrows():
            seq_lut[r.seq_a_name] = r.seq_a
            seq_lut[r.seq_b_name] = r.seq_b

    model, tok = load_model(args.lora, ckpt=args.ckpt)
    stem = Path(pdb_path).stem.upper().split(".")[0]

    rows = []
    for pair in args.pairs:
        a, b = pair.split("-")
        raw_a = seq_lut.get(f"{stem}_{a}", "<+>" + pdb_seqs[a])
        raw_b = seq_lut.get(f"{stem}_{b}", "<+>" + pdb_seqs[b])
        lead_a, qry_a = raw_a[:3], raw_a[3:]
        lead_b, qry_b = raw_b[:3], raw_b[3:]
        assert lead_a in ("<+>", "<->") and lead_b in ("<+>", "<->"), (lead_a, lead_b)

        sequence = raw_a + raw_b
        print(f"-- {stem} {a}-{b}: |a|={len(qry_a)} |b|={len(qry_b)} "
              f"L={len(tok.encode(sequence))} fast={args.fast} readout={args.readout}")
        _, contact, _ = categorical_jacobian(sequence, model, tok, fast=args.fast,
                                             batch_size=args.batch_size,
                                             readout=args.readout, amp=not args.no_amp)

        gt_map = inter_contact_map(coords, a, b, args.threshold)
        score, block, gt_aln = score_pair(contact, len(qry_a), len(qry_b), gt_map,
                                          pdb_seqs[a], pdb_seqs[b], qry_a, qry_b)
        K = int(gt_map.sum())
        print(f"   P@C = {score:.6f}   (K={K}, hits={round(score * K)})")

        # intra-chain contact recovery (P@L) for each chain, from the same jacobian.
        # token layout is [<+>] seq_a [<+>] seq_b, so chain a starts at 1 and b at 2+la.
        pl = {}
        if not args.no_pl:
            for ch, off, qs in ((a, 1, qry_a), (b, 2 + len(qry_a), qry_b)):
                gt_intra = intra_contact_map(coords, ch, args.threshold)
                v, L, hits = score_intra(contact, off, len(qs), gt_intra,
                                         pdb_seqs[ch], qs, min_sep=args.min_sep)
                pl[ch] = v
                print(f"   P@L {ch} = {v:.6f}   (L={L}, hits={hits}, |i-j|>={args.min_sep})")

        rows.append(dict(tag=tag, pdb=stem, pair=f"{a}_{b}", score=score, K=K,
                         hits=int(round(score * K)), atom=args.atom,
                         threshold=args.threshold, fast=args.fast, readout=args.readout,
                         lora=args.lora or "", ckpt=args.ckpt or "",
                         pl_a=pl.get(a), pl_b=pl.get(b), min_sep=args.min_sep,
                         len_a=len(qry_a), len_b=len(qry_b)))
        if args.save_npz:
            np.savez_compressed(out_dir / f"{stem}_{a}_{b}_{args.readout}_contact.npz",
                                full=contact, block=block, gt=gt_map)

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "pac_scores.csv", index=False)
    print(res.to_string(index=False))
    print(f">> saved {out_dir/'pac_scores.csv'}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
