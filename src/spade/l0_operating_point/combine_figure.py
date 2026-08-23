"""Karvonen's Figure 4 as one 2x2 panel, the way the paper prints it.

    python -m spade.l0_operating_point.combine_figure

His repo has no code for this: `f1_analysis.ipynb` renders four separate PNGs and the
paper assembles them in LaTeX. So this composites the same four images rather than
re-plotting, which keeps every panel byte-identical to what `verify_match.py` checks.

Layout and subcaptions are read off arXiv:2408.00113v2 Figure 4 -- top row coverage,
bottom row board reconstruction, left column the loss-recovered-vs-L0 scatter coloured by
the metric, right column the metric against L0:

    (a) Coverage of Board State            (b) Coverage of Board State vs. L0
    (c) Board Reconstruction               (d) Board Reconstruction vs. L0

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

RENDERS = repo_root() / "figures/l0_operating_point"

# (filename, subcaption), in the paper's reading order.
PANELS = [
    ("othello_3var_coverage.png", "(a) Coverage of Board State"),
    ("othello_2var_coverage.png", r"(b) Coverage of Board State vs. $L_0$"),
    ("othello_3var_reconstruct.png", "(c) Board Reconstruction"),
    ("othello_2var_reconstruct.png", r"(d) Board Reconstruction vs. $L_0$"),
]


def combine(panel_dir: Path, out_stem: Path, dpi: int = 200) -> Path:
    images = [(mpimg.imread(panel_dir / name), caption) for name, caption in PANELS]
    cell_w = max(im.shape[1] for im, _ in images)
    cell_h = max(im.shape[0] for im, _ in images)

    fig, axes = plt.subplots(
        2, 2, figsize=(2 * cell_w / dpi, 2 * cell_h / dpi), dpi=dpi
    )
    for ax, (im, caption) in zip(axes.flat, images):
        ax.imshow(im)
        ax.set_xlim(-(cell_w - im.shape[1]) / 2, im.shape[1] + (cell_w - im.shape[1]) / 2)
        ax.set_ylim(im.shape[0] + (cell_h - im.shape[0]) / 2, -(cell_h - im.shape[0]) / 2)
        ax.axis("off")
        ax.set_title(caption, fontsize=17, y=-0.11)

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

    for variant in ("karvonen_only", "with_spade"):
        panel_dir = args.renders / variant
        if not panel_dir.is_dir():
            raise FileNotFoundError(f"{panel_dir} -- run spade.l0_operating_point.plot_pareto first")
        out = combine(panel_dir, args.renders / f"figure4_othello_{variant}")
        print(f"{variant}: {out} (+ .pdf)")


if __name__ == "__main__":
    main()
