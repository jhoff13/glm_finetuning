import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dna_features_viewer import GraphicFeature, GraphicRecord


def draw_genetic_map(genbank_df, Title="Genetic Map", ax=None):
    """
    Draw a genetic map using the GenBank dataframe information, including arrows to indicate gene direction.

    Args:
        genbank_df (DataFrame): DataFrame containing 'Start', 'End', 'Name', 'Distance_from_Center', and 'Strand'.
        Title (str): Title for the plot.
        ax (matplotlib Axes, optional): Axis to draw on. If provided, the map is
            drawn on this axis and the caller is responsible for showing/saving
            the figure. If None, a new figure is created and shown (default).
    """
    features = []

    # Create GraphicFeature for each gene or intergenic region
    for _, row in genbank_df.iterrows():
        color = plt.cm.coolwarm(row['Distance_from_Center'] / genbank_df['Distance_from_Center'].max())
        features.append(
            GraphicFeature(
                start=row['Start'],
                end=row['End'],
                strand=row['Strand'],
                color=color,
                label=row['Name']
            )
        )

    # Create GraphicRecord
    record = GraphicRecord(sequence_length=genbank_df['End'].max(), features=features)

    # Plot the genetic map
    if ax is None:
        plt.figure(figsize=(15, 5))
        record.plot(figure_width=12)
        plt.title(Title)
        plt.show()
    else:
        record.plot(ax=ax)
        ax.set_title(Title)
    return record

def segments_from_df(genbank_df, length_col=None, gene_types=("Gene",)):
    """
    Pull gene arrow coordinates out of a genbank-style dataframe.

    Parameters
    ----------
    genbank_df : DataFrame
        Needs 'Strand', and either 'Start'/'End' columns or `length_col`.
        'Type' and 'Name' are used when present.
    length_col : str or None
        When given (e.g. 'seq_len'), coordinates are the cumulative sum of that
        column, i.e. token space rather than genomic space. The cumsum runs over
        the full frame before genes are selected, so offsets stay aligned with
        the concatenated sequence.
    gene_types : tuple[str]
        Values of 'Type' to keep. Ignored when there is no 'Type' column.

    Returns
    -------
    segments : np.ndarray, shape (N, 2)
    strands  : np.ndarray, shape (N,)
    names    : list[str] or None
    extent   : (low, high) span of the whole frame, spacers included
    """
    df = genbank_df.copy()

    if length_col is not None:
        ends = df[length_col].cumsum()
        df["_seg_start"], df["_seg_end"] = ends - df[length_col], ends
        start_col, end_col = "_seg_start", "_seg_end"
        extent = (0, int(df[length_col].sum()))
    else:
        start_col, end_col = "Start", "End"
        extent = (int(df[start_col].min()), int(df[end_col].max()))

    if "Type" in df.columns:
        df = df[df.Type.isin(gene_types)]

    segments = df[[start_col, end_col]].to_numpy()
    strands = df["Strand"].to_numpy()
    names = df["Name"].astype(str).tolist() if "Name" in df.columns else None
    return segments, strands, names, extent


def plot_gene_map(segments, strands=None, ax=None, names=None,
                  length_col=None, gene_types=("Gene",), xlim=None, Names=None):
    """
    Draw genes as directional arrows on a matplotlib Axes.

    Parameters
    ----------
    segments : np.ndarray, shape (N, 2) [[start, end], ...] or DataFrame
        Passing a genbank-style DataFrame gathers segments, strands, and names
        via `segments_from_df`, and spans the x axis over the whole frame rather
        than only the genes (which matters when the Axes shares x with others).
    strands  : np.ndarray, shape (N,)     1 = forward, -1 = reverse
    ax       : matplotlib Axes            defaults to the current Axes
    names    : list[str] or None          optional gene labels
    length_col, gene_types : see `segments_from_df`; DataFrame input only
    xlim     : (low, high) or None        overrides the computed x limits
    """
    if names is None and Names is not None:  # some notebooks pass Names=
        names = Names

    if isinstance(segments, pd.DataFrame):
        # allow plot_gene_map(df, ax) alongside plot_gene_map(df, ax=ax)
        if ax is None and strands is not None:
            ax, strands = strands, None
        segments, strands, df_names, extent = segments_from_df(
            segments, length_col=length_col, gene_types=gene_types
        )
        if names is None:
            names = df_names
        if xlim is None:
            xlim = extent

    if ax is None:
        ax = plt.gca()

    y_position, gene_height = 0, 1
    assert segments.shape[0] == strands.shape[0]
    if names is not None:
        assert len(names) == segments.shape[0], "names length must match number of genes"

    if segments.shape[0] == 0:
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.axis("off")
        return

    for i in range(segments.shape[0]):
        start, end = segments[i][0], segments[i][1]
        strand     = strands[i]

        if strand == -1:
            ax.arrow(end, y_position, start - end, 0,
                     head_width=gene_height, head_length=100,
                     length_includes_head=True, color="r")
        elif strand == 1:
            ax.arrow(start, y_position, end - start, 0,
                     head_width=gene_height, head_length=100,
                     length_includes_head=True, color="g")
        else:
            continue

        if names is not None:
            mid = (start + end) / 2
            ax.text(mid, y_position + gene_height * 0.6, str(names[i]),
                    ha="center", va="bottom", fontsize=8)

    # every gene sits on one row, so pin the limits and leave headroom for labels
    ax.set_ylim(y_position - gene_height, y_position + gene_height * 1.5)
    ax.set_xlim(*(xlim if xlim is not None else (segments[0][0], segments[-1][-1])))
    ax.set_xlabel("Position")
    ax.set_ylabel("Genes")
    ax.axis("off")


def plot_svd_summary(
    U, S, Vh,
    segments, strands, names=None,
    k=5,
    plot_gene_map_func=None,
    figsize_multi=(12, 8),
    figsize_svals=(6, 4),
    extra_sigmas=10,
    highlight_x=None,
    highlight_kwargs=None,
):
    """
    3-row multiplot (σ·v, |σ·u|, gene map) + standalone singular-values plot.

    Returns
    -------
    mid_sig : np.ndarray, shape (m, k)
    Y       : np.ndarray, shape (n, k)
    """
    r = min(len(S), U.shape[1], Vh.shape[0])
    k = int(max(1, min(k, r)))

    Y       = (Vh[:k, :] * S[:k, None]).T          # (n, k)
    mid_sig = np.abs(U[:, :k] * S[:k])             # (m, k)

    fig_multi, (ax0, ax_mid, ax1) = plt.subplots(3, 1, figsize=figsize_multi, sharex=True)

    lines0 = ax0.plot(Y, linewidth=1.2)
    ax0.set_ylabel("Loading (scaled)")
    ax0.set_title(f"Top-{k} right singular vectors (scaled by σ)")
    ax0.legend(handles=lines0, labels=[f"σ{j+1} · v{j+1}" for j in range(k)],
               loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, title="Components")
    ax0.grid(alpha=0.2)
    ax0.set_xlim(0, Y.shape[0] - 1)

    if highlight_x is not None:
        marks = np.unique(np.asarray(highlight_x, dtype=int))
        marks = marks[(marks >= 0) & (marks < Y.shape[0])]
        hk    = dict(color="r", linestyle=":", linewidth=1.0, alpha=0.6)
        if highlight_kwargs:
            hk.update(highlight_kwargs)
        for x0 in marks:
            ax0.axvline(x0, **hk)

    lines_mid = ax_mid.plot(mid_sig, linewidth=1.5)
    ax_mid.legend(handles=lines_mid, labels=[f"|σ{j+1}·u{j+1}|" for j in range(k)],
                  loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, title="Components")
    ax_mid.set_ylabel("|σ·u|")
    ax_mid.set_title(f"Top-{k} left singular vectors (magnitude, scaled)")
    ax_mid.grid(alpha=0.2)

    if plot_gene_map_func is not None:
        plot_gene_map_func(segments, strands, names=names, ax=ax1)
    else:
        ax1.text(0.5, 0.5, "plot_gene_map_func is None", ha="center", va="center")
    ax1.set_xlabel("Position / feature index")
    ax1.set_yticks([])
    fig_multi.subplots_adjust(right=0.82, hspace=0.35)

    s_top = S[:min(r, k + extra_sigmas)]
    x     = np.arange(1, len(s_top) + 1)
    fig_svals, ax_s = plt.subplots(figsize=figsize_svals)
    ax_s.plot(x, s_top, marker="o", linewidth=1.5)
    ax_s.axvline(k, color="red", linestyle="--", linewidth=1.2, label=f"k = {k}")
    ax_s.set_xlabel("Component index")
    ax_s.set_ylabel("Singular value (σ)")
    ax_s.set_title("Top singular values")
    ax_s.legend(loc="upper right")
    ax_s.grid(alpha=0.25)
    fig_svals.tight_layout()

    return mid_sig, Y


def plot_value_norms_and_components(
    l2_vals,
    components,
    K=10,
    top_per_component=3,
    suptitle=None,
    head_label="Attn Head",
    figsize=(10, 5),
    dpi=150,
):
    """
    Top panel  : L2 norms of token value vectors with top-K annotated.
    Bottom panel: right singular components with peak annotations.

    Returns
    -------
    fig, axes
    """
    top_idx  = np.argsort(np.abs(l2_vals))[-K:][::-1]
    top_vals = l2_vals[top_idx]

    fig, axes = plt.subplots(2, 1, dpi=dpi, figsize=figsize, sharex=True)
    if suptitle is not None:
        fig.suptitle(suptitle)

    axes[0].plot(l2_vals, lw=1)
    axes[0].set_title("L2 Norms of Token Value Representation")
    axes[0].set_ylabel("L2 Norm")
    for idx, val in zip(top_idx, top_vals):
        axes[0].scatter(idx, val, color="red", zorder=3)
        axes[0].annotate(f"{idx}", xy=(idx, val), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=7, color="red")

    for r in range(components.shape[-1]):
        y     = components[:, r]
        line, = axes[1].plot(y, label=f"Component {r}", alpha=0.8)
        color = line.get_color()
        top_idx_r  = np.argsort(np.abs(y))[-top_per_component:][::-1]
        top_vals_r = y[top_idx_r]
        print(f"Comp {r} - top idxs: {top_idx_r}; vals: {top_vals_r}")
        for idx, val in zip(top_idx_r, top_vals_r):
            axes[1].scatter(idx, val, color=color, zorder=3)
            axes[1].annotate(f"{idx}", xy=(idx, val), xytext=(0, 6),
                             textcoords="offset points", ha="center", fontsize=7, color=color)

    axes[1].set_title(f"Right Singular Vectors Scaled [{head_label}]")
    axes[1].set_ylabel(r"$|\sigma \cdot u|$")
    axes[1].legend(bbox_to_anchor=(1, 1), fontsize=7)
    plt.tight_layout()
    return fig, axes


def create_figure(contact_df: pd.DataFrame, tokens, title="CONSERVATION"):
    """
    Seaborn heatmap of a contact / conservation matrix. [From CatJac]

    Parameters
    ----------
    contact_df : pd.DataFrame  columns [i, j, value]
    tokens     : list[str]
    title      : str

    Returns
    -------
    fig, ax
    """
    seqlen = len(tokens)
    contact_df = contact_df.copy()
    contact_df["i"] = contact_df["i"].astype(int)
    contact_df["j"] = contact_df["j"].astype(int)
    contact_df["i_token"] = contact_df["i"].astype(str) + ": " + contact_df["i"].map(lambda x: tokens[x - 1])
    contact_df["j_token"] = contact_df["j"].astype(str) + ": " + contact_df["j"].map(lambda x: tokens[x - 1])

    heatmap_data = contact_df.pivot(index="j", columns="i", values="value").sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap    = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(heatmap_data, ax=ax, cmap=cmap, cbar=True, square=True,
                cbar_kws={"shrink": 0.75}, xticklabels=False, yticklabels=False)

    tick_pos = np.arange(0, seqlen, 20)
    ax.set_xticks(tick_pos + 0.5);  ax.set_xticklabels(tick_pos, rotation=90)
    ax.set_yticks(tick_pos + 0.5);  ax.set_yticklabels(tick_pos, rotation=0)
    ax.set_xlabel("Position (bp)")
    ax.set_ylabel("Position (bp)")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax
