"""Window embeddings/logits/stats workflow helpers.

Functions copied verbatim from
projects/0_context_scans/notebooks/context_logits_figures.ipynb.

The functions below depend on the following module-level names being set by the
caller before use (they are notebook globals in the original code):

    import utils.window_logits as wl
    wl.MODEL         = MODEL
    wl.TOKENIZER     = TOKENIZER
    wl.DEVICE        = DEVICE
    wl.MASK_TOKEN_ID = MASK_TOKEN_ID
    # wl.TQDM_BAR_FORMAT already has a default below; override if desired.
"""

import time

import numpy as np
import torch
from tqdm import tqdm

# Provided by the caller (see module docstring). Left as placeholders so the
# module imports cleanly; they are only needed at call time.
MODEL = None
TOKENIZER = None
DEVICE = None
MASK_TOKEN_ID = None
# Optional convenience lookups populated by load_model().
tok_dict = None
inv_tok_dict = None
ALPHABET = None
ALPHABET_MAP = None
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'


def set_globals(model=None, tokenizer=None, device=None, mask_token_id=None):
    """Inject already-loaded objects into this module's globals.

    Use this when you have already created MODEL/TOKENIZER in your notebook:

        import utils.window_logits as wl
        wl.set_globals(MODEL, TOKENIZER, DEVICE)   # mask id inferred from tokenizer
    """
    global MODEL, TOKENIZER, DEVICE, MASK_TOKEN_ID
    global tok_dict, inv_tok_dict, ALPHABET, ALPHABET_MAP
    if model is not None:
        MODEL = model
    if tokenizer is not None:
        TOKENIZER = tokenizer
        if mask_token_id is None:
            mask_token_id = tokenizer.mask_token_id
        tok_dict = tokenizer.get_vocab()
        inv_tok_dict = {v: k for k, v in tok_dict.items()}
        ALPHABET = tokenizer.convert_ids_to_tokens(np.arange(4, 24))
        ALPHABET_MAP = dict(zip(range(20), ALPHABET))
    if device is not None:
        DEVICE = device
    if mask_token_id is not None:
        MASK_TOKEN_ID = mask_token_id
    return MODEL, TOKENIZER


def load_model(model_name="tattabio/gLM2_650M", device=None):
    """Load the gLM2 model + tokenizer and populate this module's globals.

    Imports transformers lazily so simply importing this module never triggers
    a network download.

        import utils.window_logits as wl
        wl.load_model("tattabio/gLM2_650M")
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        .eval()
        .to(resolved_device)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return set_globals(model, tokenizer, resolved_device)


# Grabbing window embeddings/logits/stats workflow----------------------
def find_motif(tokens, motif):
    """
    Find the motif in the tokens.

    Parameters:
        tokens: numpy array, list, or PyTorch tensor
            The input sequence to search for the motif.
        motif: list or numpy array
            The motif to search for.

    Returns:
        List of start indices where the motif matches in the tokens.
    """
    #convert ad detach tokens
    tokens = tokens.detach().cpu().numpy()
    motif = motif.detach().cpu().numpy()
    # Find matches using a sliding window
    matches = []
    motif_len = motif.shape[0]
    for i in range(len(tokens) - motif_len + 1):
        if np.array_equal(tokens[i:i + motif_len], motif):
            matches.append(i)  # Store the start index of each match
    assert matches, "No matches found for the motif in the tokens."
    if len(matches)>1: print(f'{len(matches)} matches - reporting first hit')
    return matches[0], matches[0]+motif_len

def get_reference_logits(Full_X, Ref_X, Ref_start, Ref_end):
    """Get the Ref_Seq logits given the Seq context - only Ref_Seq masked and returned."""
    assert (Full_X[0, Ref_start:Ref_end] == Ref_X[0]).all(), 'Seq misallignment'

    f = lambda x: MODEL(x).logits[0, :, 4:24].detach().cpu().numpy()
    ln = Ref_end - Ref_start - 1
    fx_h = []
    with tqdm(total=ln, bar_format=TQDM_BAR_FORMAT) as pbar:
        for n in range(Ref_start, Ref_end):
            x_h = torch.clone(Full_X)
            x_h[0,n] = MASK_TOKEN_ID
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                fx_h.append(f(x_h)[n])
            pbar.update(1)
    return np.array(fx_h)

def get_reference_logits_batch_mask(
    Full_X,
    Ref_X,
    Ref_start,
    Ref_end,
    mask_ratio: float = 0.15,
    verbose: bool = True,
):
    """
    Get Ref_Seq logits using batch masking within the ref gene only.
    
    Context (non-ref) tokens are NEVER masked.
    Ref positions are randomly shuffled and masked in chunks of mask_ratio * ref_len.
    Each position is masked exactly once — its logit is saved at that pass.
    The final chunk may be smaller than mask_ratio * ref_len (edge case handled).
    
    Parameters
    ----------
    Full_X      : (1, L) token tensor — full context sequence
    Ref_X       : (1, R) token tensor — reference gene tokens
    Ref_start   : int — start index of ref gene in Full_X  
    Ref_end     : int — end index of ref gene in Full_X
    mask_ratio  : float — fraction of ref positions to mask per forward pass
    verbose     : bool — if False, silence the tqdm progress bar
    
    Returns
    -------
    np.ndarray of shape (ref_len, 20) — logit at each ref position
    """
    assert (Full_X[0, Ref_start:Ref_end] == Ref_X[0]).all(), 'Seq misalignment'

    f = lambda x: MODEL(x).logits[0, :, 4:24].detach().cpu().numpy()

    ref_len = Ref_end - Ref_start
    chunk_size = max(1, int(ref_len * mask_ratio))

    # Shuffle all ref positions — each will be masked exactly once
    ref_positions = np.arange(Ref_start, Ref_end)
    shuffled = ref_positions.copy()
    np.random.shuffle(shuffled)

    logit_accumulator = np.zeros((ref_len, 20), dtype=np.float32)

    # Split into chunks — last chunk may be smaller
    chunks = [shuffled[i:i + chunk_size] for i in range(0, ref_len, chunk_size)]
    n_passes = len(chunks)

    with tqdm(total=ref_len, bar_format=TQDM_BAR_FORMAT, disable=not verbose) as pbar:
        for pass_idx, chunk in enumerate(chunks):
            is_last = pass_idx == n_passes - 1

            # Clone full sequence — context untouched
            x_h = torch.clone(Full_X)

            # Mask only this chunk of ref positions
            x_h[0, chunk] = MASK_TOKEN_ID

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                logits_np = f(x_h)  # (L, 20)

            # Save logits only at masked ref positions
            for abs_pos in chunk:
                rel_pos = abs_pos - Ref_start
                logit_accumulator[rel_pos] = logits_np[abs_pos]
                pbar.update(1)

    return logit_accumulator

def get_perplexity(Probs):
    entropy = -(Probs * np.log2(Probs+1e-10)).sum(-1)
    perplixity = np.power(2, entropy)
    return perplixity

def parse_windows(Genbank_df, ref_indx, logits_type='classic'):
    t0 = time.time()
    ref_seq = Genbank_df.at[ref_indx,'Sequence']  # make this an optional input for a reference
    max_distance = int(Genbank_df.Distance_from_Center.max())
    windows = list(range(0,max_distance))
    strand_dict = {1:'<+>',-1:'<->'}
    logits = []
    for window in range(0,max_distance):
        print(f'> Window: {window+1}/{max_distance} = {(100*(window/max_distance)):.2f}% ; {(time.time() - t0):.2f} sec')
        #select window:
        Sub_genbank_df = Genbank_df[(Genbank_df.Distance_from_Center<=window)]
        #construct seq:
        seq = []
        for i, row in Sub_genbank_df.iterrows():
            dir = strand_dict.get(row.Strand,'<+>')
            row.Sequence = ''.join([dir,row.Sequence])
            seq.append(row.Sequence)
        seq = ''.join(seq)
        #tokenize -
        full_x = torch.tensor(TOKENIZER(seq)["input_ids"]).to(DEVICE)
        ref_x = torch.tensor(TOKENIZER(ref_seq)["input_ids"]).to(DEVICE)
        ref_start, ref_end = find_motif(full_x, ref_x)
        assert (full_x[ref_start:ref_end] == ref_x).all(), f'Reference Seq Error: {ref_seq}'     
        # ''.join([inv_tok_dict[int(token)] for token in full_full_x[ref_start:ref_end]trunc_x])       
        #logits
        if logits_type == 'classic':
            log = get_reference_logits(full_x.unsqueeze(0), ref_x.unsqueeze(0), ref_start, ref_end) 
        elif logits_type == 'quick':
            log = get_reference_logits_batch_mask(full_x.unsqueeze(0), ref_x.unsqueeze(0), ref_start, ref_end)
        logits.append(log)
    logits = np.array(logits)
    return logits
