import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser, is_aa
import torch
from tqdm import tqdm
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.Align import PairwiseAligner

# ── Core math ─────────────────────────────────────────────────────────────────
def get_categorical_jacobian(sequence, model, tokenizer, device,
                             nuc_tokens, aa_tokens, mask_token_id,
                             tqdm_bar_format=None, fast=False):
    """
    Compute a categorical Jacobian for a mixed nucleotide/amino-acid sequence.

    Parameters
    ----------
    sequence : str
    model : nn.Module
    tokenizer : tokenizer with encode / convert_ids_to_tokens
    device : str or torch.device
    nuc_tokens : list[int]
    aa_tokens : list[int]
    mask_token_id : int
    tqdm_bar_format : str or None
    fast : bool
        If True, mask each position once (faster). Otherwise substitute all tokens.

    Returns
    -------
    jac : torch.Tensor
    contact : np.ndarray
    tokens : list[str]
    """
    all_tokens = nuc_tokens + aa_tokens
    num_tokens = len(all_tokens)

    input_ids = torch.tensor(tokenizer.encode(sequence), dtype=torch.int)
    tokens    = tokenizer.convert_ids_to_tokens(input_ids)
    seqlen    = input_ids.shape[0]

    is_nuc_pos   = torch.isin(input_ids, torch.tensor(nuc_tokens)).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
    is_nuc_token = torch.isin(torch.tensor(all_tokens), torch.tensor(nuc_tokens)).view(1, -1, 1, 1).repeat(1, 1, 1, num_tokens)
    is_aa_pos    = torch.isin(input_ids, torch.tensor(aa_tokens)).view(-1, 1, 1, 1).repeat(1, 1, seqlen, 1)
    is_aa_token  = torch.isin(torch.tensor(all_tokens), torch.tensor(aa_tokens)).view(1, -1, 1, 1).repeat(1, 1, 1, num_tokens)

    input_ids = input_ids.unsqueeze(0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        f  = lambda x: model(x)[0][..., all_tokens].cpu().float()
        x  = torch.clone(input_ids).to(device)
        ln = x.shape[1]
        fx = f(x)[0]

        if fast:
            fx_h = torch.zeros((ln, 1, ln, num_tokens), dtype=torch.float32)
        else:
            fx_h = torch.zeros((ln, num_tokens, ln, num_tokens), dtype=torch.float32)
            x    = torch.tile(x, [num_tokens, 1])

        with tqdm(total=ln, bar_format=tqdm_bar_format) as pbar:
            for n in range(ln):
                x_h = torch.clone(x)
                x_h[:, n] = mask_token_id if fast else torch.tensor(all_tokens)
                fx_h[n]   = f(x_h)
                pbar.update(1)

        jac        = fx_h - fx
        valid_nuc  = is_nuc_pos & is_nuc_token
        valid_aa   = is_aa_pos  & is_aa_token
        jac        = torch.where(valid_nuc | valid_aa, jac, 0.0)
        contact    = jac_to_contact(jac.numpy())

    return jac, contact, tokens

def jac_to_contact(jac, symm=True, center=True, diag="remove", apc=True):
    """Reduce a Jacobian tensor to a 2-D contact matrix."""
    X = jac.copy()
    Lx, Ax, Ly, Ay = X.shape

    if center:
        for i in range(4):
            if X.shape[i] > 1:
                X -= X.mean(i, keepdims=True)

    contacts = np.sqrt(np.square(X).sum((1, 3)))

    if symm and (Ax != 20 or Ay != 20):
        contacts = (contacts + contacts.T) / 2

    if diag == "remove":
        np.fill_diagonal(contacts, 0)

    if diag == "normalize":
        d = np.diag(contacts)
        contacts = contacts / np.sqrt(d[:, None] * d[None, :])

    if apc:
        ap = contacts.sum(0, keepdims=True) * contacts.sum(1, keepdims=True) / contacts.sum()
        contacts = contacts - ap

    if diag == "remove":
        np.fill_diagonal(contacts, 0)

    return contacts


def contact_to_dataframe(con):
    """Flatten a square contact matrix to a long-form DataFrame [i, j, value]."""
    n   = con.shape[0]
    idx = [str(i) for i in np.arange(1, n + 1)]
    df  = pd.DataFrame(con, index=idx, columns=idx)
    df  = df.stack().reset_index()
    df.columns = ["i", "j", "value"]
    return df


# ── DataFrame helpers ──────────────────────────────────────────────────────────

def get_offdiag_dict(n: np.ndarray) -> dict:
    """
    Map every (i, j) coordinate in an (n[0] x n[1]) matrix to its diagonal index.
    0 = main diagonal, positive = superdiagonal, negative = subdiagonal.
    """
    ref_diag_dict = {}
    for d in range(n[0]):
        for sign in (1, -1):
            coords = np.vstack(np.where(np.eye(*n, k=sign * d))).T
            ref_diag_dict[sign * d] = [tuple(x) for x in coords]
    return {coord: k for k, coords in ref_diag_dict.items() for coord in coords}


def matrix_to_df(mat: np.ndarray, ref_diag_dict) -> pd.DataFrame:
    """Convert a matrix to a DataFrame with (i, j, value, ij, diag) columns."""
    i_idx, j_idx = np.indices(mat.shape)
    df = pd.DataFrame({"i": i_idx.ravel(), "j": j_idx.ravel(), "value": mat.ravel()})
    df = df.reset_index().rename({"index": "order"}, axis=1)
    df["ij"]   = list(zip(df["i"], df["j"]))
    df["diag"] = df["ij"].map(ref_diag_dict)
    return df


def Score_PatC(query_matrix, gt_df, ref_diag):
    """Precision at K against a ground-truth contact DataFrame."""
    contact_df  = matrix_to_df(query_matrix, ref_diag)
    K           = int(gt_df.value.sum())
    if K <= 0:
        return 0.0
    top_K_hits  = contact_df.sort_values("value", ascending=False).ij.values[:K]
    return gt_df[gt_df.ij.isin(top_K_hits)].value.sum() / K


# ── PDB contact maps ───────────────────────────────────────────────────────────

def _get_pdb(pdb_code, out_dir=None, verbose=True):
    """`out_dir` defaults to $GLM2_PDB_DIR, else ./pdb beside the package."""
    if out_dir is None:
        out_dir = os.environ.get(
            "GLM2_PDB_DIR", str(Path(__file__).resolve().parent.parent / "pdb"))
    """Fetch or locate a PDB/mmCIF file, return its local path."""
    if pdb_code is None or pdb_code == "":
        raise ValueError("pdb_code must be a non-empty string.")
    if os.path.isfile(pdb_code):
        if verbose: print(f"> Loading: {pdb_code}")
        return pdb_code
    if len(pdb_code) == 4:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{pdb_code}.pdb1")
        if not os.path.isfile(path):
            os.system(f"wget -qnc https://files.rcsb.org/download/{pdb_code}.pdb1.gz -O {path}.gz")
            os.system(f"gunzip -f {path}.gz")
        return path
    os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb")
    return f"AF-{pdb_code}-F1-model_v3.pdb"


def _parse_structure(pdb_file):
    ext = pdb_file.split(".")[-1].lower()
    if ext in ("pdb", "pdb1"):
        return PDBParser(QUIET=True).get_structure("protein", pdb_file)
    if ext in ("cif", "mmcif"):
        return MMCIFParser(QUIET=True).get_structure("protein", pdb_file)
    raise ValueError(f"Unsupported format: {ext}")


def _extract_ca_coords(chain, verbose=False):
    ca_coords, residues = [], []
    for residue in chain:
        if is_aa(residue, standard=True):
            if "CA" in residue:
                ca_coords.append(residue["CA"].get_coord())
                residues.append(residue.get_id()[1])
            else:
                if verbose: print(f"Missing Cα in chain {chain.get_id()} residue {residue.get_id()[1]}")
    if not ca_coords:
        return np.empty((0, 3), dtype=float), []
    return np.vstack(ca_coords), residues


def get_contact_map_from_pdb(pdb_code, threshold=8.0, chain_id=None, verbose=True):
    """
    Intra-chain Cα contact maps from a PDB/mmCIF.

    Returns a dict {chain_id: np.ndarray} or a single array when chain_id is given.
    """
    pdb_file  = _get_pdb(pdb_code, verbose=verbose)
    structure = _parse_structure(pdb_file)
    chains    = list(structure.get_chains())
    if chain_id:
        chains = [c for c in chains if c.get_id() == chain_id]
        if not chains:
            raise ValueError(f"Chain '{chain_id}' not found.")

    contact_maps = {}
    for chain in chains:
        cid = chain.get_id()
        if verbose: print(f"Processing Chain: {cid}")
        coords, _ = _extract_ca_coords(chain, verbose=verbose)
        if coords.shape[0] == 0:
            if verbose: print(f"No valid Cα atoms in chain {cid}. Skipping.")
            continue
        n   = len(coords)
        cmap = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(coords[i] - coords[j]) <= threshold:
                    cmap[i, j] = cmap[j, i] = 1
        contact_maps[cid] = cmap
        if verbose: print(f"Chain {cid}: {n} residues, contact map generated.\n")

    return contact_maps.get(chain_id) if chain_id else contact_maps


def get_inter_contact_map_from_pdb(pdb_code, threshold=8.0, chain_id=None, verbose=True):
    """
    Inter-chain Cα contact maps from a PDB/mmCIF.

    chain_id : str like 'A-B' or tuple ('A', 'B'), or None for all pairs.
    Returns a dict {'A-B': np.ndarray shape (len_A, len_B)}.
    """
    pdb_file  = _get_pdb(pdb_code, verbose=verbose)
    structure = _parse_structure(pdb_file)
    model     = next(structure.get_models())

    chain_data = {}
    for chain in model.get_chains():
        cid    = chain.get_id()
        if verbose: print(f"Processing Chain: {cid}")
        coords, resnums = _extract_ca_coords(chain, verbose=verbose)
        if verbose: print(f"  Cα residues: {len(coords)}")
        chain_data[cid] = (coords, resnums)

    def parse_pair(ci):
        if isinstance(ci, tuple) and len(ci) == 2:
            return (str(ci[0]), str(ci[1]))
        if isinstance(ci, str) and "-" in ci:
            a, b = ci.split("-", 1)
            return (a.strip(), b.strip())
        raise ValueError("chain_id must be like 'A-B' or ('A','B').")

    if chain_id is not None:
        pairs = [parse_pair(chain_id)]
    else:
        ids   = [cid for cid, (c, _) in chain_data.items() if c.shape[0] > 0]
        pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]

    contact_maps = {}
    for a, b in pairs:
        A, _ = chain_data.get(a, (np.empty((0, 3)), []))
        B, _ = chain_data.get(b, (np.empty((0, 3)), []))
        if A.shape[0] == 0 or B.shape[0] == 0:
            if verbose: print(f"Skipping {a}-{b}: no valid Cα atoms.")
            continue
        if verbose: print(f"Computing {a}-{b}: {A.shape[0]} x {B.shape[0]}")
        dists = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
        contact_maps[f"{a}-{b}"] = (dists <= float(threshold)).astype(np.int8)
        if verbose: print(f"Inter-chain map {a}-{b} generated.\n")

    return contact_maps

def pdb_to_sequence(pdb_code, chain_id=None):
    pdb_path = get_pdb(pdb_code)
    structure = PDBParser(QUIET=True).get_structure("pdb", pdb_path)

    seqs = {}
    for chain in structure.get_chains():
        if chain_id and chain.id != chain_id:
            continue
        seqs[chain.id] = ''.join(
            seq1(r.resname) for r in chain if r.id[0] == ' '
        )

# Aligning Contact Maps (For accounting for Crystal gaps) ----------------------------------

def group_consecutive_with_padding(arr):
    """
    Group consecutive integers in a 1D array into rows and pad with NaN.

    Parameters
    ----------
    arr : array-like (1D)

    Returns
    -------
    out : np.ndarray (2D, float)
        Each row is a consecutive cluster, padded with np.nan.
    """
    arr = np.asarray(arr, dtype=int)
    if arr.size == 0:
        return np.empty((0, 0))
    arr = np.sort(arr)
    splits = np.where(np.diff(arr) != 1)[0] + 1
    clusters = np.split(arr, splits)
    max_len = max(len(c) for c in clusters)
    out = np.full((len(clusters), max_len), np.nan)
    for i, c in enumerate(clusters):
        out[i, : len(c)] = c
    return out


def extract_chain_sequences_from_pdb(pdb_path):
    """
    Extract amino-acid sequences from a PDB/mmCIF file.

    Parameters
    ----------
    pdb_path : str
        Path to .pdb, .pdb1, .ent, .cif, or .mmcif file.

    Returns
    -------
    seqs : dict
        Dictionary mapping chain_id -> sequence (1-letter AA codes).
    """
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(pdb_path)

    ext = pdb_path.split('.')[-1].lower()
    if ext in ("pdb", "pdb1", "ent"):
        parser = PDBParser(QUIET=True)
    elif ext in ("cif", "mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported structure format: {ext}")

    structure = parser.get_structure("structure", pdb_path)
    model = next(structure.get_models())  # first model only

    seqs = {}

    for chain in model:
        chain_id = chain.get_id()
        seq = []

        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            seq.append(
                protein_letters_3to1.get(residue.get_resname(), "X")
            )

        if seq:  # only keep non-empty chains
            seqs[chain_id] = "".join(seq)

    return seqs

def align_and_pad_contacts(
    ref_seq_1,
    query_seq_1,
    ref_seq_2,
    query_seq_2,
    ref_contact_map,
    query_contact_map,
):
    """
    Align two sequence pairs and pad their contact maps based on alignment gaps.
    """
    
    def align_sequences(seq_a, seq_b, mode="global"):
        """
        Align two sequences (strings) with Biopython and return aligned strings with '-' gaps.
        """
        aligner = PairwiseAligner()
        aligner.mode = mode
        aligner.match_score = 1
        aligner.mismatch_score = -1
        aligner.open_gap_score = -2
        aligner.extend_gap_score = -0.5
    
        aln = aligner.align(seq_a, seq_b)[0]
    
        return list(aln[0]), list(aln[1])
        
    def pad_contacts_with_alignment_gaps(Contact_map, SeqA_gap_sites, SeqB_gap_sites):
        Out_contact_map = Contact_map.copy()
        for g in group_consecutive_with_padding(SeqA_gap_sites)[::-1]:
            for i in g:
                if np.isnan(i):
                    continue
                idx = min(int(i), Out_contact_map.shape[0])
                Out_contact_map = np.insert(Out_contact_map, idx, 0, axis=0)
        for g in group_consecutive_with_padding(SeqB_gap_sites)[::-1]:
            for i in g:
                if np.isnan(i):
                    continue
                idx = min(int(i), Out_contact_map.shape[1])
                Out_contact_map = np.insert(Out_contact_map, idx, 0, axis=1)
        return Out_contact_map
        
    ref_aligned_1, query_aligned_1 = align_sequences(ref_seq_1, query_seq_1)
    ref_aligned_1, query_aligned_1 = np.array(ref_aligned_1), np.array(query_aligned_1)
    assert ref_aligned_1.shape == query_aligned_1.shape

    ref_aligned_2, query_aligned_2 = align_sequences(ref_seq_2, query_seq_2)
    ref_aligned_2, query_aligned_2 = np.array(ref_aligned_2), np.array(query_aligned_2)
    assert ref_aligned_2.shape == query_aligned_2.shape

    ref_inserts_2, ref_inserts_1 = (
        np.where(ref_aligned_2 == '-')[0],
        np.where(ref_aligned_1 == '-')[0],
    )
    query_inserts_2, query_inserts_1 = (
        np.where(query_aligned_2 == '-')[0],
        np.where(query_aligned_1 == '-')[0],
    )

    aligned_ref_contact = pad_contacts_with_alignment_gaps(
        ref_contact_map,
        ref_inserts_1,
        ref_inserts_2,
    )
    aligned_query_contact = pad_contacts_with_alignment_gaps(
        query_contact_map,
        query_inserts_1,
        query_inserts_2,
    )

    assert aligned_query_contact.shape == aligned_ref_contact.shape, (
        f"{aligned_query_contact.shape} != {aligned_ref_contact.shape}"
    )

    return aligned_ref_contact, aligned_query_contact
