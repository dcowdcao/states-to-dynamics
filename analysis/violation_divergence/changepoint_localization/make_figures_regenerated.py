"""`changepoint_by_space` and `threshold_detection`, on the corpus whose suffix is REGENERATED
after t0.

Same panels, same nine spaces, same estimator, same sort order and styling as
`make_figures.py` -- it imports that module's `panel` and `finish` rather than restating the
drawing rules, so the two sets of figures cannot drift apart. The sort and the ceiling are its
own: this figure keeps the two-panel AUROC + balanced-accuracy layout that the spliced one
dropped, and a ceiling over both columns is meaningless once there is only one of them.

One axis label had to change in each figure and it is not cosmetic. The straightness violation
corpus carries no legal-alternative arm, so the contrast is the illegal game against **its own
source legal game**: in `changepoint_by_space` that is the parent figure's `vs_reference`
rather than its `vs_control`, and in `threshold_detection` the negatives are positions of
unspliced legal games rather than of a matched legal splice. Reading the two corpora's left
panels as the same quantity would overstate the transfer.

Usage: PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/make_figures_regenerated.py
           [--camera-ready]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from spade.figures.paper_style import apply_style

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_figures import FALSE_ALARM, finish, panel  # noqa: E402  (sibling module)


def draw_by_space(table: pd.DataFrame, camera_ready: bool) -> Path:
    table = table.sort_values("auroc_vs_control", ascending=False).reset_index(drop=True)
    row = table.iloc[0]
    alarms = table[FALSE_ALARM]
    # The parent figure's 0.50 ceiling clips here: regenerated-suffix localization runs to
    # 0.43, so the ceiling follows the data rather than the sibling figure.
    top_b = max(0.50, float(table.hit_illegal_hi.max()) * 1.22)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))

    panel(axes[0], table, "auroc_vs_control",
          "AUROC, illegal vs its own source legal game\n($\\uparrow$ better)",
          0.5, (table.auroc_null.min(), table.auroc_null.max()), 0, 1.05,
          "chance (empirical)")
    panel(axes[1], table, "hit_illegal",
          "localization: argmax lands on $t_0$\n($\\uparrow$ better)",
          row.chance_uniform, (alarms.values.min(), alarms.values.max()), 1, top_b,
          "chance")

    return finish(fig, "changepoint_by_space_regenerated", camera_ready)


def draw_threshold(table: pd.DataFrame, camera_ready: bool) -> Path:
    table = table.sort_values("auroc_position", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    # One ceiling for both panels: two probability scales sharing a chance line at 0.5.
    hi = table[["auroc_position_hi", "balanced_accuracy_hi"]].to_numpy().max()
    top = max(1.05, float(hi) * 1.12)

    panel(axes[0], table, "auroc_position",
          "per-position AUROC, illegal move vs move in an\n"
          "unspliced legal game ($\\uparrow$ better)",
          0.5, None, 0, top, "chance")
    panel(axes[1], table, "balanced_accuracy",
          "balanced accuracy at the fitted cut, out of fold\n($\\uparrow$ better)",
          0.5, None, 1, top, "chance")

    return finish(fig, "threshold_detection_regenerated", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    results = HERE / "results"
    print(draw_by_space(pd.read_csv(results / "changepoint_regenerated.csv"),
                        args.camera_ready))
    print(draw_threshold(pd.read_csv(results / "threshold_localization_regenerated.csv"),
                         args.camera_ready))


if __name__ == "__main__":
    main()
