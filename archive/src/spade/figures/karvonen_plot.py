"""Shared Karvonen-style Figure-4 plotting.

Both figure entrypoints build a results DataFrame their own way, then hand it
here to render the identical 2x2 loss / coverage / reconstruction grid. Columns
expected on the frame: sae_type, l0, frac_recovered, coverage, reconstruction_f1.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
})

# Canonical SAE-type display properties (Karvonen style).
SAE_CONFIG = {
    "relu":           {"name": "ReLU",           "marker": "o", "color": "#e41a1c", "size": 60},
    "topk_relu":      {"name": "TopK",           "marker": "d", "color": "#377eb8", "size": 60},
    "gated":          {"name": "Gated",          "marker": "s", "color": "#984ea3", "size": 60},
    "panneal":        {"name": "p-Anneal",       "marker": "P", "color": "#e41a1c", "size": 60},
    "gated_anneal":   {"name": "Gated+p-Anneal", "marker": "P", "color": "#984ea3", "size": 60},
    "sparsemax_dist": {"name": "SpaDE",          "marker": "^", "color": "#ff7f00", "size": 120},
    "batch_topk":     {"name": "BatchTopK",      "marker": "D", "color": "#377eb8", "size": 60},
    "cosine_spade":   {"name": "CosineSpaDE",    "marker": "v", "color": "#ff7f00", "size": 120},
}

# trainer_class strings (from trainSAE config.json) → canonical SAE_CONFIG keys.
# Canonical keys map to themselves so an already-normalized frame also works.
TRAINER_CLASS_MAP = {
    "StandardTrainer": "relu",
    "TopKTrainer": "topk_relu",
    "GatedSAETrainer": "gated",
    "PAnnealTrainer": "panneal",
    "GatedAnnealTrainer": "gated_anneal",
    "SpaDETrainer": "sparsemax_dist",
    "BatchTopKTrainer": "batch_topk",
    "CosineSpaDETrainer": "cosine_spade",
    "relu": "relu",
    "topk_relu": "topk_relu",
    "gated": "gated",
    "panneal": "panneal",
    "gated_anneal": "gated_anneal",
    "sparsemax_dist": "sparsemax_dist",
    "batch_topk": "batch_topk",
    "cosine_spade": "cosine_spade",
}

ORDERED_TYPES = ["relu", "panneal", "topk_relu", "batch_topk", "gated",
                 "gated_anneal", "sparsemax_dist", "cosine_spade"]


def get_display_info(row):
    """(display_name, color, marker, size, is_spade) for one results row.

    SpaDE is drawn larger so it stands out against the baseline scatter.
    """
    raw_type = row["sae_type"]
    canonical = TRAINER_CLASS_MAP.get(raw_type, raw_type)
    is_spade = canonical == "sparsemax_dist"
    cfg = SAE_CONFIG.get(canonical, {"name": raw_type, "marker": "o",
                                     "color": "gray", "size": 60})
    size = 120 if is_spade else cfg["size"]
    return cfg["name"], cfg["color"], cfg["marker"], size, is_spade


def add_shape_legend(ax):
    """Legend keyed by marker shape only.

    Used on the color-mapped left panels, where marker color encodes a metric
    (coverage / reconstruction F1), so shape has to carry SAE-type identity.
    """
    handles = [
        Line2D([0], [0], marker=SAE_CONFIG[t]["marker"], linestyle="None",
               markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=1.2, markersize=8, label=SAE_CONFIG[t]["name"])
        for t in ORDERED_TYPES
    ]
    ax.legend(handles=handles, title="SAE type (shape)", loc="lower right",
              fontsize=8, title_fontsize=9, framealpha=0.9)


def _error_bars(ax, df, x_col, y_col):
    """Draw seed-variation error bars (±1 std across seeds) under the markers.

    Uses <col>_std columns when present (from aggregate_seeds.py); NaN std
    (single-seed configs) draws a zero-length bar.
    """
    xerr = df[f"{x_col}_std"].fillna(0) if f"{x_col}_std" in df.columns else None
    yerr = df[f"{y_col}_std"].fillna(0) if f"{y_col}_std" in df.columns else None
    if xerr is None and yerr is None:
        return
    ax.errorbar(df[x_col], df[y_col], xerr=xerr, yerr=yerr, fmt="none",
                ecolor="dimgray", elinewidth=1.0, capsize=2.5, alpha=0.75,
                zorder=3)


def _annotate_configs(ax, df, x_col, y_col, annotate):
    """Label specific configs (df['config'] → text) with a small arrow."""
    if not annotate or "config" not in df.columns:
        return
    for cfg, text in annotate.items():
        for _, row in df[df["config"] == cfg].iterrows():
            ax.annotate(text, (row[x_col], row[y_col]),
                        xytext=(-26, -30), textcoords="offset points",
                        ha="right", fontsize=8, fontweight="bold", color="#b35900",
                        arrowprops=dict(arrowstyle="->", color="#b35900", lw=1.2),
                        zorder=12)


def plot_karvonen_subplot(ax, df, x_col, y_col, x_label, y_label, title,
                          color_col=None, cmap="viridis", annotate=None):
    """Plot one Karvonen-style scatter subplot.

    If color_col is given, points are colored by that metric and a ScalarMappable
    is returned for a figure-level colorbar; otherwise points use per-SAE-type
    colors and None is returned. SpaDE points are outlined and drawn on top.
    Frames from aggregate_seeds.py (with <metric>_std columns) get error bars.
    annotate maps df['config'] values to callout labels.
    """
    plotted_labels = set()
    scalar_mappable = None
    _error_bars(ax, df, x_col, y_col)
    _annotate_configs(ax, df, x_col, y_col, annotate)

    if color_col is not None and color_col in df.columns and df[color_col].notna().any():
        norm = mcolors.Normalize(vmin=df[color_col].min(), vmax=df[color_col].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        scalar_mappable = sm

        for _, row in df.iterrows():
            display_name, _, marker, size, is_spade = get_display_info(row)
            label = display_name if display_name not in plotted_labels else None
            if label:
                plotted_labels.add(display_name)
            ax.scatter(row[x_col], row[y_col],
                       c=[sm.to_rgba(row[color_col])], marker=marker, s=size,
                       edgecolors="black" if is_spade else "dimgray",
                       linewidths=1.5 if is_spade else 0.5,
                       zorder=10 if is_spade else 5, label=label)
    else:
        for _, row in df.iterrows():
            display_name, color, marker, size, is_spade = get_display_info(row)
            label = display_name if display_name not in plotted_labels else None
            if label:
                plotted_labels.add(display_name)
            ax.scatter(row[x_col], row[y_col],
                       c=color, marker=marker, s=size,
                       edgecolors="black" if is_spade else "none",
                       linewidths=1.5 if is_spade else 0,
                       zorder=10 if is_spade else 5, label=label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(left=0)
    return scalar_mappable


def render_grid(df, out_path, suptitle, annotate=None):
    """Render the 2x2 loss / coverage / reconstruction grid to out_path.

    annotate maps df['config'] values to callout labels drawn on every panel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sm_cov = plot_karvonen_subplot(
        axes[0, 0], df, "l0", "frac_recovered",
        r"$L_0$", "Loss Recovered (Fidelity)",
        "Loss Recovered vs $L_0$ (color = Coverage)",
        color_col="coverage", cmap="viridis", annotate=annotate)
    plot_karvonen_subplot(
        axes[0, 1], df, "l0", "coverage",
        r"$L_0$", "Coverage", "Coverage of Board States vs $L_0$",
        annotate=annotate)
    sm_f1 = plot_karvonen_subplot(
        axes[1, 0], df, "l0", "frac_recovered",
        r"$L_0$", "Loss Recovered (Fidelity)",
        "Loss Recovered vs $L_0$ (color = Recon $F_1$)",
        color_col="reconstruction_f1", cmap="plasma", annotate=annotate)
    plot_karvonen_subplot(
        axes[1, 1], df, "l0", "reconstruction_f1",
        r"$L_0$", "Reconstruction $F_1$", "Reconstruction $F_1$ vs $L_0$",
        annotate=annotate)

    # Figure-level colorbars for the left column (explicit, to avoid tight-layout
    # clipping and guarantee they export).
    if sm_cov is not None:
        fig.colorbar(sm_cov, ax=axes[0, 0], fraction=0.046, pad=0.04).set_label("Coverage", fontsize=10)
    if sm_f1 is not None:
        fig.colorbar(sm_f1, ax=axes[1, 0], fraction=0.046, pad=0.04).set_label("Reconstruction $F_1$", fontsize=10)

    axes[0, 1].legend(loc="lower right", fontsize=8, framealpha=0.9)
    add_shape_legend(axes[0, 0])
    add_shape_legend(axes[1, 0])

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")
