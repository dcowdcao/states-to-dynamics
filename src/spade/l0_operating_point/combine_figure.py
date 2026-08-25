"""Karvonen's Figure 4 as one 2x2 panel, the way the paper prints it.

    python -m spade.l0_operating_point.combine_figure

His repo has no code for this: `f1_analysis.ipynb` renders four separate PNGs and the
paper assembles them in LaTeX. So this composites the same four images rather than
re-plotting, which keeps every panel byte-identical to what `verify_match.py` checks.

Layout is read off arXiv:2408.00113v2 Figure 4 -- top row coverage, bottom row board
reconstruction, left column the loss-recovered-vs-L0 scatter coloured by the metric, right
column the metric against L0. The subcaptions are bare letters: every panel already names
its own axes, and his wording repeated them.

Panels arrive at slightly different pixel sizes because each was saved with a tight bbox,
so each is centred in a common cell on white rather than stretched -- scaling them to a
shared size would resample his axes and defeat the point of compositing.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from spade.common.paths import repo_root
from spade.l0_operating_point.plot_pareto import OUR_FONTS, OUR_TYPE

RENDERS = repo_root() / "figures/l0_operating_point"

# (filename, subcaption), in the paper's reading order.
PANELS = [
    ("othello_3var_coverage.png", "(a)"),
    ("othello_2var_coverage.png", "(b)"),
    ("othello_3var_reconstruct.png", "(c)"),
    ("othello_2var_reconstruct.png", "(d)"),
]
PANEL_WIDTH_IN = 10   # plot_pareto's figsize; each panel lands in a cell narrower than that

# Which panel directory each combined figure is built from. The his-SAEs-only figure takes
# the styled render, not the byte-identical one verify_match checks.
SOURCES = {"karvonen_only": "styled_karvonen_only", "with_spade": "with_spade"}


def combine(panel_dir: Path, out_stem: Path, dpi: int = 200) -> Path:
    # The subcaptions must match the type inside the panels they label.
    with plt.rc_context(OUR_FONTS):
        return _compose(panel_dir, out_stem, dpi)


def _compose(panel_dir: Path, out_stem: Path, dpi: int) -> Path:
    images = [(mpimg.imread(panel_dir / name), caption) for name, caption in PANELS]
    cell_w = max(im.shape[1] for im, _ in images)
    cell_h = max(im.shape[0] for im, _ in images)
    # A panel is authored PANEL_WIDTH_IN wide and drawn into a cell of cell_w/dpi, so its
    # 17 pt type appears at about 8 pt here. Set the subcaption from the same scale, or it
    # reads at twice the size of everything it labels.
    caption_pt = OUR_TYPE["axes.labelsize"] * (cell_w / dpi) / PANEL_WIDTH_IN

    fig, axes = plt.subplots(
        2, 2, figsize=(2 * cell_w / dpi, 2 * cell_h / dpi), dpi=dpi
    )
    for ax, (im, caption) in zip(axes.flat, images):
        ax.imshow(im)
        ax.set_xlim(-(cell_w - im.shape[1]) / 2, im.shape[1] + (cell_w - im.shape[1]) / 2)
        ax.set_ylim(im.shape[0] + (cell_h - im.shape[0]) / 2, -(cell_h - im.shape[0]) / 2)
        ax.axis("off")
        ax.set_title(caption, fontsize=caption_pt, y=-0.11)

    fig.subplots_adjust(wspace=0.02, hspace=0.16, left=0.01, right=0.99, top=0.99, bottom=0.05)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(suffix), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_stem.with_suffix(".png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--renders", type=Path, default=RENDERS)
    args = ap.parse_args()

    for variant, source in SOURCES.items():
        panel_dir = args.renders / source
        if not panel_dir.is_dir():
            raise FileNotFoundError(f"{panel_dir} -- run spade.l0_operating_point.plot_pareto first")
        out = combine(panel_dir, args.renders / f"figure4_othello_{variant}")
        print(f"{variant}: {out} (+ .pdf)")


if __name__ == "__main__":
    main()
