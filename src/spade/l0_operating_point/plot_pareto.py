"""Karvonen's four OthelloGPT L0-vs-board-metric plots, his SAEs alone and with ours added.

Every style parameter is copied from his `circuits/f1_analysis.ipynb` (`plot_l0_metric`,
`plot_3var_graph`); this project's paper_style.py and FIGURE_GUIDELINES deliberately do not
apply. KARVONEN_STYLE.md records what was copied and the three places his code had to be
extended to carry eight trainer classes instead of five.

Both figure sets come out of one run, so `renders/karvonen_only` cannot pick up our points
by accident — that directory is what `verify_match.py` checks against his published PNGs.
`load_points.py` decides which of our rows are points; run it alone to see what landed.

  python plot_pareto.py                     # every stream in load_points.OUR_STREAMS
  python plot_pareto.py --ours '<glob>'     # only these result dirs or f1_results.csv files
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize

from spade.common.paths import repo_root
from spade.l0_operating_point.load_points import (
    KARVONEN_CSV, check_physics, load, one_per_config, streams,
)

RENDERS = repo_root() / "figures/l0_operating_point"

RECON = "recon_f1"
COVERAGE = "coverage"

# label, marker, index into plt.cm.Set1. His five come first in the order his CSV's
# trainer_class.unique() returns, which is what fixed their markers and colours.
TRAINERS = {
    "GatedSAETrainer": ("Gated SAE", "o", 0),
    "GatedAnnealTrainer": ("Gated SAE w/ p-annealing", "X", 1),
    "PAnnealTrainer": ("Standard w/ p-annealing", "^", 2),
    "StandardTrainer": ("Standard", "d", 3),
    "TrainerTopK": ("TopK SAE", "s", 4),
    "SpaDETrainer": ("SpaDE", "*", 6),
    "CosineSpaDETrainer": ("Cosine SpaDE", "P", 7),
    "BatchTopKTrainer": ("BatchTopK SAE", "v", 8),
}
BIG_LEGEND_MARKERS = ("d", "*")  # shapes that read small at markersize 10

METRIC_1_LABEL = r"$L_0$ (Lower is sparser)"
METRIC_2_LABEL = "Loss Recovered (Fidelity)"

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
    }
)


def _legend_kw(location: str, below: bool) -> dict:
    """His legends sit inside the axes. Ours cannot: SpaDE and CosineSpaDE put points in
    the lower right of both 2-var panels, exactly where his legend lives, and a legend
    over data is a legibility bug however faithful the rest of the styling is."""
    if not below:
        return {"loc": location}
    return {"loc": "upper center", "bbox_to_anchor": (0.5, -0.16), "ncol": 3,
            "frameon": False, "handletextpad": 0.2, "columnspacing": 1.0}


def _finish(fig, legend_below: bool, output_filename) -> None:
    """tight_layout measures only the axes, so with the legend anchored outside it the
    y-label gets cropped by the subsequent bbox_inches="tight". Let the tight bbox do all
    the work in that case, and hand it the legend explicitly."""
    extra = []
    if legend_below:
        extra = [a.get_legend() for a in fig.axes if a.get_legend()]
    else:
        fig.tight_layout()
    if output_filename:
        fig.savefig(output_filename, bbox_inches="tight",
                    bbox_extra_artists=extra or None)
    plt.close(fig)


def present(df: pd.DataFrame) -> dict[str, tuple[str, str, int]]:
    unknown = set(df["trainer_class"]) - set(TRAINERS)
    if unknown:
        raise KeyError(f"no marker/colour assigned for {sorted(unknown)}")
    return {t: v for t, v in TRAINERS.items() if (df["trainer_class"] == t).any()}


def plot_l0_metric(
    df: pd.DataFrame,
    column_name: str,
    y_label: str,
    legend_location: str = "lower right",
    legend_below: bool = False,
    xlims: tuple[float, float] = (0.0, 400.0),
    output_filename: Optional[Path] = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for trainer, (label, marker, color_idx) in present(df).items():
        trainer_data = df[df["trainer_class"] == trainer]
        ax.scatter(
            trainer_data["l0"],
            trainer_data[column_name],
            marker=marker,
            c=plt.cm.Set1(color_idx),
            s=100,
            label=label,
        )

    ax.set_xlabel(METRIC_1_LABEL)
    ax.set_ylabel(y_label)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, **_legend_kw(legend_location, legend_below))
    ax.set_xlim(*xlims)
    _finish(fig, legend_below, output_filename)


def plot_3var_graph(
    df: pd.DataFrame,
    color_column: str,
    legend_location: str = "lower right",
    legend_below: bool = False,
    xlims: tuple[float, float] = (0.0, 400.0),
    y_lims: tuple[float, float] = (0.985, 1.001),
    colorbar_label: Optional[str] = None,
    output_filename: Optional[Path] = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = Normalize(vmin=df[color_column].min(), vmax=df[color_column].max())

    handles, labels = [], []
    for trainer, (label, marker, _) in present(df).items():
        trainer_data = df[df["trainer_class"] == trainer]
        scatter = ax.scatter(
            trainer_data["l0"],
            trainer_data["frac_recovered"],
            c=trainer_data[color_column],
            cmap="viridis",
            marker=marker,
            s=100,
            label=label,
            norm=norm,
            edgecolor="black",
        )

        _handle, _ = scatter.legend_elements(prop="sizes")
        _handle[0].set_markeredgecolor("black")
        _handle[0].set_markerfacecolor("white")
        _handle[0].set_markersize(13 if marker in BIG_LEGEND_MARKERS else 10)
        handles += _handle
        labels.append(label)

    fig.colorbar(ax.collections[0], ax=ax, label=colorbar_label)

    ax.set_xlabel(METRIC_1_LABEL)
    ax.set_ylabel(METRIC_2_LABEL)
    ax.legend(handles, labels, **_legend_kw(legend_location, legend_below))
    ax.set_xlim(*xlims)
    ax.set_ylim(*y_lims)
    _finish(fig, legend_below, output_filename)


HIS_3VAR_YLIMS = (0.996, 1.0005)


def _ylims_holding(df: pd.DataFrame, xlims: tuple[float, float]) -> tuple[float, float]:
    """His frac_recovered window is 0.9960-1.0005 because every SAE of his sits at ~1.
    Ours do not: on his window 44 of 55 SpaDE points and 62 of 117 CosineSpaDE points
    fall off the bottom and vanish without trace. A panel that silently drops 80% of an
    architecture is not a comparison, so the combined figure widens to hold what it plots.

    Floored at 0 rather than at the true minimum. Recovering less loss than zero-ablation
    is degeneracy, not a worse operating point, and the six points that do (his own
    Standard configs at L0 < 3, down to -0.81) drag the axis far enough that every real
    point collapses onto one line at the top. `render` prints what the floor excludes."""
    shown = df[(df["l0"] >= xlims[0]) & (df["l0"] <= xlims[1])]["frac_recovered"]
    low, high = max(0.0, shown.min()), shown.max()
    pad = 0.04 * (high - low)
    return (low - pad, high + pad)


def render(df: pd.DataFrame, out_dir: Path, faithful: bool = True):
    """faithful=True is his figure exactly, and verify_match holds it to that.

    faithful=False keeps every style parameter but moves the legend out of the data and
    widens the 3-var y-axis; both are forced by our points landing where his never did."""
    out_dir.mkdir(parents=True, exist_ok=True)
    below = not faithful
    y3 = HIS_3VAR_YLIMS if faithful else _ylims_holding(df, (0.0, 400.0))
    plot_l0_metric(df, RECON, r"Othello Board State Reconstruction $F_1$",
        legend_below=below, output_filename=out_dir / "othello_2var_reconstruct.png")
    plot_l0_metric(df, COVERAGE, "Coverage of Othello Board State",
        legend_below=below, output_filename=out_dir / "othello_2var_coverage.png")
    plot_3var_graph(df, RECON, y_lims=y3, legend_below=below,
        colorbar_label=r"Othello Board State Reconstruction $F_1$",
        output_filename=out_dir / "othello_3var_reconstruct.png")
    plot_3var_graph(df, COVERAGE, y_lims=y3, legend_below=below,
        colorbar_label="Coverage of Othello Board State",
        output_filename=out_dir / "othello_3var_coverage.png")
    off = df[~((df.l0 <= 400) & df.frac_recovered.between(*y3))]
    print(f"{len(df)} SAEs -> {out_dir}   3-var y {y3[0]:.4f}..{y3[1]:.4f}, "
          f"{len(off)} off the axes"
          + (f": {dict(off.trainer_class.value_counts())}" if len(off) else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--karvonen-csv", type=Path, default=KARVONEN_CSV)
    ap.add_argument("--ours", nargs="*", help="result dirs or f1_results.csv paths, globs allowed")
    ap.add_argument("--out-dir", type=Path, default=RENDERS)
    args = ap.parse_args()

    df = one_per_config(load(streams(args.ours), args.karvonen_csv))
    print(df["trainer_class"].value_counts().to_string())

    flagged = check_physics(df)
    if len(flagged):
        print(f"\n{len(flagged)} of our points are below the physics floor and are plotted anyway:")
        print(flagged[["config", "seed", "l0", RECON, COVERAGE, "flag"]].to_string(index=False))

    render(df[df["source"] == "karvonen"], args.out_dir / "karvonen_only")
    render(df, args.out_dir / "with_spade", faithful=False)


if __name__ == "__main__":
    main()
