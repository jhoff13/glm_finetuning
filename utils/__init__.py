# utils.model needs the gLM2 modelling source, which is not vendored in this
# repo; keep the package importable without it so the contact/scoring helpers
# below are usable on their own.
from .model import (load_glm2, vanilla_attn, parse_genbank, process_embeddings,
                    get_perplexity, distance_decay_labeller, HAVE_GLM2_SRC)
from .contacts import (
    get_categorical_jacobian,
    jac_to_contact,
    contact_to_dataframe,
    get_contact_map_from_pdb,
    get_inter_contact_map_from_pdb,
    pdb_to_sequence,
    get_offdiag_dict,
    matrix_to_df,
    Score_PatC,
    align_and_pad_contacts,
    extract_chain_sequences_from_pdb,
    group_consecutive_with_padding
)
from .plot import plot_gene_map, segments_from_df, plot_svd_summary, plot_value_norms_and_components, create_figure, draw_genetic_map
from .window_logits import find_motif, get_reference_logits, get_reference_logits_batch_mask, parse_windows, set_globals, load_model
