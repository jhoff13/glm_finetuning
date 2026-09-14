import os
import sys
import pickle
import math
from pathlib import Path
import torch
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import torch
import numpy as np

# local copy of the gLM2 modelling code; override with $GLM2_MODELS_DIR
GLM2_SRC = Path(os.environ.get(
    "GLM2_MODELS_DIR", Path(__file__).resolve().parent.parent / "models" / "gLM2"))
if str(GLM2_SRC) not in sys.path:
    sys.path.append(str(GLM2_SRC))

# The gLM2 modelling source is NOT vendored here (it lives in the research
# repo).  Import it lazily so the rest of this package -- in particular the
# contact/scoring helpers in utils.contacts -- works without it; load_glm2 and
# vanilla_attn then raise a clear error instead of breaking `import utils`.
try:
    from glm2_qk_select_attn_model_with_bias import gLM2Model
    HAVE_GLM2_SRC = True
except ImportError as _e:
    HAVE_GLM2_SRC = False
    _GLM2_SRC_ERR = str(_e)
    gLM2Model = None


def _require_glm2_src():
    if not HAVE_GLM2_SRC:
        raise ImportError(
            f"gLM2 modelling source not found ({_GLM2_SRC_ERR}). Point "
            f"$GLM2_MODELS_DIR at the directory holding "
            f"glm2_qk_select_attn_model_with_bias.py (currently {GLM2_SRC}).")


def load_glm2(config_path: Path, weights_path: Path, device: str | torch.device) -> "gLM2Model":
    _require_glm2_src()
    """Load a gLM2 model from a pickled config and a .pth weights file."""
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    model = gLM2Model(config)

    raw_sd   = torch.load(weights_path, map_location=device, weights_only=True)
    model_sd = model.state_dict()

    filtered = {
        k.removeprefix("glm2."): v
        for k, v in raw_sd.items()
        if k.removeprefix("glm2.") in model_sd
    }

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[warn] missing keys  ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:5]} …")

    return model.eval().to(device)


def vanilla_attn(
    q: torch.Tensor,  # [batch, heads, seq_len, dim]
    k: torch.Tensor,  # [batch, heads, seq_len, dim]
    soft = True
) -> torch.Tensor:
    """
    Adapted from Dr. Yo.
    Returns softmax attention weights [b, h, s, s].
    """
    dim_head = q.shape[-1]
    dotprod = torch.einsum("... h i k, ... h j k -> ... h i j", q, k) / math.sqrt(dim_head)
    if soft: return torch.softmax(dotprod, dim=-1)
    else: return dotprod

def parse_genbank(genbank_file):
    """Parse a GenBank file and return a DataFrame of CDS genes and intergenic regions."""
    records = SeqIO.parse(genbank_file, "genbank")
    extracted_data = []

    for record in records:
        features = record.features
        prev_end = 0
        for i, feature in enumerate(features):
            if feature.type == "CDS" and "translation" in feature.qualifiers:
                gene_seq  = feature.qualifiers["translation"][0]
                gene_name = feature.qualifiers.get("gene", ["unknown"])[0]
                start     = feature.location.start
                end       = feature.location.end
                product   = feature.qualifiers.get(
                    "product",
                    feature.qualifiers.get("note", ["unknown"]),
                )[0]
                extracted_data.append({
                    "Type": "Gene", "Name": gene_name, "Product": product,
                    "Sequence": gene_seq, "Start": start, "End": end,
                    "Strand": feature.strand,
                })
                prev_end = end

            if i > 0 and feature.type != "CDS":
                intergenic_start = prev_end
                intergenic_end   = feature.location.start
                if intergenic_end > intergenic_start:
                    intergenic_seq = record.seq[intergenic_start:intergenic_end]
                    extracted_data.append({
                        "Type": "Intergenic", "Name": f"intergenic_{i}",
                        "Sequence": str(intergenic_seq).lower(),
                        "Start": intergenic_start, "End": intergenic_end, "Strand": None,
                    })
                prev_end = feature.location.end

    return pd.DataFrame(extracted_data)


def process_embeddings(genbank_df, tokenizer):
    """
    Build a flat token sequence from a genbank DataFrame with strand tokens,
    and return both the raw string and a batched input tensor.
    """
    strand_dict = {1: "<+>", -1: "<->"}
    seq = []
    for _, row in genbank_df.iterrows():
        direction = strand_dict.get(row.Strand, "")
        seq.append(direction + row.Sequence)
    seq = "".join(seq)
    full_x = torch.tensor(tokenizer(seq)["input_ids"]).unsqueeze(0)
    return seq, full_x

def distance_decay_labeller(genbank_df, center_idx):
    """Label each row with the number of 'Gene'-type entries between it
    and the center row. Center = 0; increments by 1 each time a Gene is crossed.

    center_idx is a POSITIONAL index (0-based row number).
    """
    df = genbank_df.reset_index(drop=True).copy()
    n = len(df)
    assert 0 <= center_idx < n, 'center_idx out of range'

    dist = [0] * n
    types = df['Type'].to_numpy()

    # Walk left (toward index 0)
    offset = 0
    for i in range(center_idx - 1, -1, -1):
        # crossing the row at i+1 to reach row i
        if types[i + 1] == 'Gene':
            offset += 1
        dist[i] = offset

    # Walk right (toward last index)
    offset = 0
    for i in range(center_idx + 1, n):
        if types[i - 1] == 'Gene':
            offset += 1
        dist[i] = offset

    df['Distance_from_Center'] = dist
    return df
    
def get_perplexity(Probs):
    """Input 1D Probs, returns entropy & perplexity (base 2)"""
    entropy = -(Probs * np.log2(Probs + 1e-10)).sum(-1)
    perplexity = np.power(2, entropy)
    return entropy, perplexity
