# glm_finetuning

Run ```python wandb_helper.py -o /home/gridsan/jhoff/models/glm_finetuning/training/wandb/latest-run``` in sbatch if running offline (slurm) to update WanB.

Update .sh for changing workflow.
```
Training Loop for gLM2 LORA finetuning.

optional arguments:
  -h, --help            show this help message and exit
  -n NAME, --name NAME  Name of training run
  -i FASTA, --fasta FASTA
                        path/to/dataset.fasta
  -w WEIGHTS_PATH, --weights_path WEIGHTS_PATH
                        LORA weights path.
  -o LOG_PATH, --log_path LOG_PATH
                        WandB log path [Default: w]
  -x OFFLINE, --offline OFFLINE
                        WandB offline run [Default: False]
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch Size [Default: 16]
  -r RANK, --rank RANK  LORA rank size [Default: 4]
  -d DROPOUT, --dropout DROPOUT
                        LORA rank size [Default: 0.15]
  -l LR, --lr LR        Learning Rate [Default: 1e-3]
  -e EPOCH, --epoch EPOCH
                        Number of Epochs [Default: 2]
  -t TRAIN, --train TRAIN
                        Training Ratio [Default 0.7]. Val/test = (1 - Train_ratio)/2
```


---

## `gLM2_trainer_v1.py`

Successor to `gLM2_trainer_v0.py`. Same MLM objective, but:

* **`--mode {lora,full,last_n,head_select}`** instead of LoRA only
* masking restricted to amino-acid / nucleotide tokens, so the `<+>`/`<->` strand
  tokens and padding are never corrupted; BERT-style 80/10/10 corruption
* loss on masked positions only (`labels = -100` elsewhere)
* length-grouped dynamic padding with a real `attention_mask`
* step-based rather than epoch-based, with periodic checkpoints and eval

### Input format

`-i` is a directory holding `train.fasta` / `val.fasta`, one record per line pair,
where the sequence line is a literal gLM2 token string — strand tokens included,
e.g. `<+>MAKQ...<+>MSTL...` for a concatenated pair.

### Interface tracking and scoring — `pac_eval.py` + `utils/`

`--track_pairs A-C --pdb 3IP4` logs the *fast* categorical-Jacobian P@C and P@L
of a target chain pair at every eval, so a run can be taken to saturation while
watching the quantity of interest. Treat the fast probe as a progress signal
only — it has disagreed in **sign** with the slow categorical Jacobian, so
rescore checkpoints properly before drawing conclusions.

`pac_eval.py` is also the standalone scorer. It computes interface precision
against the Cβ (Cα fallback for Gly) 8 Å inter-chain contact map of the
deposited biological assembly, globally aligning the model sequence to the PDB
chains and gap-padding both maps into a shared frame:

```bash
python pac_eval.py --pdb 3IP4 --pairs A-C --readout logits \
    --ckpt runs/ft/my_run/step_03000.pt -o runs/scored --save_npz
```

Two conventions that matter: **`--readout logits`** reproduces the published
numbers, `hidden` does not (they differ in absolute scale, so never mix them in
one comparison); and the Cβ ground truth is what reproduces the published
denominators — a Cα map gives a different K.

`utils/` is vendored so `pac_eval.py` runs without the research repo.
`utils.contacts` (the scoring helpers) works standalone. **`utils.model`
does not** — `load_glm2`/`vanilla_attn` need the gLM2 modelling source, which
is *not* included here; they raise a clear ImportError, and `import utils` still
succeeds (check `utils.HAVE_GLM2_SRC`). Set `GLM2_MODELS_DIR` if you have it.

Needs biopython and pandas in addition to the trainer's dependencies. PDB files
download on demand to `$GLM2_PDB_DIR` (default `./pdb`).

`--mode head_select` additionally needs the eager-attention model wrapper
(`glm2_headselect_for_mlm.py`); point `GLM2_MODELS_DIR` at the directory holding
it.

### Environment overrides

| var | default |
|---|---|
| `GLM2_MODEL` | `tattabio/gLM2_650M` |
| `GLM2_SEQS_CSV` | unset — only needed for `--track_pairs` |
| `GLM2_MODELS_DIR` | `./models/gLM2` — for `--mode head_select` and `utils.model` |
| `GLM2_REPO` | this directory — data root for `pac_eval.py` |
| `GLM2_PDB_DIR` | `./pdb` — where PDB downloads land |
