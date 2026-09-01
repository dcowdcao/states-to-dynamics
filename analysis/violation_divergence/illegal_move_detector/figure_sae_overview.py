"""Which dictionary detects a spliced move best, and what the winner's scores look like.

`detector_overview` is this figure's raw layer-5 counterpart: (c) there splits one space by
rule, (a) here splits one arm by space. Both panels are drawn by the figure that owns them --
`figure_auroc_l0_matched.bars` and `figure_histograms.panel` -- so nothing here can disagree
with either.

**(a) is the at-t0 paired statistic, not the per-position one.** The pooled illegal branch
against the SAME source game's legal control, ranked on the score at that game's own t0. That
is what `figure_auroc.py` ranks, so raw layer 5's bar here and `detector_overview` (c)'s
pooled bar are one number and the two figures read on one axis. The per-position AUROC over
every candidate in the band is a different question and lives in
`changepoint_localization/figures/threshold_detection_l0_matched.png`, where raw scores 0.716
against 0.681 here: its negatives are mostly positions nowhere near a splice, and this one's
is the matched control at the same t0.

(b) is whichever space (a) puts first, so the panels cannot disagree about the winner.

Usage: PYTHONPATH=src:analysis/violation_divergence/illegal_move_detector python
           analysis/violation_divergence/illegal_move_detector/figure_sae_overview.py
           [--camera-ready]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

import detector
from figure_auroc_l0_matched import bars
from spade.figures.paper_style import apply_style, letters_topleft, save_figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "changepoint_localization"))
from make_figures import histogram_panel  # noqa: E402  (sibling study, not a package)


def draw(table: pd.DataFrame, camera_ready: bool) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6),
                             gridspec_kw={"width_ratios": [1.0, 0.92]})
    bars(axes[0], table)
    n_games = histogram_panel(axes[1], table.space.iloc[0])
    assert (table.n_games == n_games).all(), "the two panels are not one evaluation set"
    fig.tight_layout()
    letters_topleft(fig, axes)
    return save_figure(fig, detector.FIGURES / "sae_detector_overview", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    table = pd.read_csv(detector.RESULTS / "auroc_by_space_l0_matched.csv")
    print(draw(table.sort_values("auroc", ascending=False).reset_index(drop=True),
               args.camera_ready))


if __name__ == "__main__":
    main()
