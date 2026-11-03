"""B.2: the 234x, 536x and 1.4x that justify Figs 13-16's two forms, from their committed CSVs.

`divergence_by_space/make_figures.py --raw --l0-matched` writes every drawn point of Figs 14
and 16 (in-plane distance from the reference, mean over n = 100 games, per space, series and
offset r). Three spans of each, max over min:

- control_mean: each space's legal control averaged over r >= 0, Fig 13's divisor, across the
  nine spaces. B.2's "~234x" is Fig 14's.
- every_value: the largest value drawn over the smallest drawn at r >= 0, any series and
  space; before r = 0 every distance is zero. B.2's "536x" is Fig 14's.
- panel_peak: each space's largest drawn value, across spaces. B.2's "within 1.4x" is Fig 16's.

Writes results/appendix_numbers/span_stats.csv.

Usage: python analysis/appendix_numbers/span_stats.py
"""

import pandas as pd

from spade.common.paths import results_dir

OUT = results_dir("appendix_numbers") / "span_stats.csv"
TABLES = {"fig14_displacement": results_dir("fig14_distance_displacement_raw")
                                / "divergence_by_space_n100_raw_l0_matched.csv",
          "fig16_r21": results_dir("fig16_distance_r21_raw")
                       / "divergence_by_space_r21_n100_raw_l0_matched.csv"}
CONTROL = "control_legal"


def span(figure: str, statistic: str, by_space: pd.Series) -> dict:
    return {"figure": figure, "statistic": statistic,
            "max_space": by_space.idxmax(), "max": by_space.max(),
            "min_space": by_space.idxmin(), "min": by_space.min(),
            "ratio": by_space.max() / by_space.min()}


def rows(figure: str, drawn: pd.DataFrame) -> list[dict]:
    after = drawn[drawn.r >= 0]
    top, bottom = drawn.loc[drawn.distance.idxmax()], after.loc[after.distance.idxmin()]
    return [span(figure, "control_mean",
                 after[after.series == CONTROL].groupby("space").distance.mean()),
            {"figure": figure, "statistic": "every_value",
             "max_space": top.space, "max": top.distance,
             "min_space": bottom.space, "min": bottom.distance,
             "ratio": top.distance / bottom.distance},
            span(figure, "panel_peak", drawn.groupby("space").distance.max())]


def main() -> None:
    table = pd.DataFrame([row for figure, path in TABLES.items()
        for row in rows(figure, pd.read_csv(path).query("measured_in == 'plane'"))])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False, float_format="%.6g")
    print(table.to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
