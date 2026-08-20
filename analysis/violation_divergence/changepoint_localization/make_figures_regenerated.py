"""`changepoint_by_space`, drawn on the corpus whose suffix is REGENERATED after t0.

Same two panels, same nine spaces, same estimator, same sort order and styling as
`make_figures.py` -- it imports that module's `panel` rather than restating the drawing rule,
so the two figures cannot drift apart.

One axis label had to change and it is not cosmetic. The straightness violation corpus
carries no legal-alternative arm, so panel (a) is the illegal game against **its own source
legal game**, which is the parent figure's `vs_reference` contrast, not its `vs_control`.
Reading the two figures' left panels as the same quantity would overstate the transfer.

Usage: PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/make_figures_regenerated.py
           [--camera-ready]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from spade.figures.paper_style import MUTED, apply_style, save_figure

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_figures import FALSE_ALARM, panel  # noqa: E402  (sibling module)


def draw(table: pd.DataFrame, camera_ready: bool) -> Path:
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

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.text(0.005, 0.985, "hue = architecture family, hatch = its second dictionary; "
             "raw layer 5 is the reference space", fontsize=7, color=MUTED, va="top")
    return save_figure(fig, HERE / "figures" / "changepoint_by_space_regenerated",
                       camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    print(draw(pd.read_csv(HERE / "results" / "changepoint_regenerated.csv"),
               args.camera_ready))


if __name__ == "__main__":
    main()
