"""Which dictionary lets the detector rank an illegal move above the legal move it replaced,
once every dictionary is read at the same sparsity?

`figure_auroc.py` splits one space by rule; this splits one arm by space. The bars are the
eight `l0_operating_point` picks -- one checkpoint per architecture, each the nearest-L0 member
of its own grid to the shared target -- against raw layer 5. The canonical seed-42 roster sits
wherever each family's default hyperparameters put it on the sparsity axis, so a difference
between two of those spaces is an architecture difference *and* an L0 difference; this roster
removes the second.

Same statistic as `figure_auroc.py`: the pooled illegal branch against the SAME source game's
legal control, ranked on the score at that game's own t0, one value per branch. Same games too
-- `results/eval_games.csv`, the 500 t0-stratified games that figure wrote down, read here
rather than re-dealt, so the two panels score one set.

Which bars clear raw, and whether the top bar is separable from the next, are paired contrasts on
the bars' own draws, not two overlapping intervals (FIGURE_GUIDELINES 4.2): every dictionary minus
raw and the top bar minus every other dictionary go to results/auroc_by_space_l0_matched_paired.csv.

Bars are sorted by height, because the ordering is the claim (FIGURE_GUIDELINES 4.5), and raw
layer 5 keeps reference styling wherever it lands. The panel is drawn by
`changepoint_localization.make_figures.panel`, the same code that study's by-space bars use,
so the two cannot drift.

Usage: PYTHONPATH=src python analysis/violation_divergence/illegal_move_detector/figure_auroc_l0_matched.py
           [--camera-ready]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import detector
from spade.figures.paper_style import REFERENCE_SPACE, apply_style
from spade.l0_operating_point import roster

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "changepoint_localization"))
from make_figures import finish, panel  # noqa: E402  (sibling study, not a package)

BOOT_SEED = 42
SPACES = [REFERENCE_SPACE] + roster.SPACES


def eval_rows(data: dict, sel: dict) -> np.ndarray:
    """Positions into every arm's index array for the games of `results/eval_games.csv`.

    That file is the evaluation set, not a recipe for one: `detector.stratified` deals an
    equal quota of games at every t0 and `figure_auroc` writes the result down, so reading it
    back is what makes "the same 500 games" checkable instead of merely reproducible.
    """
    want = pd.read_csv(detector.RESULTS / "eval_games.csv")
    game, t0 = data["game_id"][sel["reference"]], data["t0"][sel["reference"]]
    order = np.argsort(game, kind="stable")
    keep = order[np.searchsorted(game[order], want.game_id.to_numpy())]
    assert np.array_equal(game[keep], want.game_id.to_numpy()), "a listed game is not eligible"
    assert np.array_equal(t0[keep], want.t0.to_numpy()), "eval_games.csv disagrees on a t0"
    return np.sort(keep)


def scored(space: str, cand: np.ndarray, sel: dict, keep: np.ndarray) -> dict[str, np.ndarray]:
    """The illegal arm and both legal controls in one space, each read at its game's own t0."""
    data = detector.load(space)
    s = detector.score(data, cand)
    return {arm: detector.at_t0(s, cand, data["t0"], sel[arm][keep])
            for arm in ("illegal", "control", "control2")}


def auroc(x: dict, draw=slice(None)) -> float:
    return detector.auroc(x["illegal"][draw], x["control"][draw])


def row(space: str, x: dict, draws: list) -> dict:
    """One space's bar: AUROC, its CI, and the empirical chance level in that space."""
    lo, hi = detector.ci(lambda draw: auroc(x, draw), draws)
    return {"space": space, "l0": roster.L0[space] if space in roster.SPACES else "",
            "n_games": len(x["illegal"]), "auroc": auroc(x),
            "auroc_lo": lo, "auroc_hi": hi,
            "auroc_null": detector.auroc(x["control2"], x["control"])}


def paired(scores: dict[str, dict], top: str, draws: list) -> pd.DataFrame:
    """Every dictionary minus raw, then the top bar minus every other dictionary."""
    pairs = ([(s, REFERENCE_SPACE) for s in roster.SPACES]
             + [(top, s) for s in roster.SPACES if s != top])
    rows = []
    for a, b in pairs:
        lo, hi = detector.ci(lambda draw: auroc(scores[a], draw) - auroc(scores[b], draw), draws)
        rows.append({"space": a, "minus": b, "n_games": len(scores[a]["illegal"]),
                     "auroc_diff": auroc(scores[a]) - auroc(scores[b]),
                     "auroc_diff_lo": lo, "auroc_diff_hi": hi})
    return pd.DataFrame(rows)


def table() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = detector.load()
    cand, sel = detector.band(), detector.arms(data)
    game_of = data["game_id"][sel["reference"]]
    for arm, rows in sel.items():
        assert np.array_equal(data["game_id"][rows], game_of), f"{arm} is not aligned by game"
    keep = eval_rows(data, sel)
    # One row per game per arm, so a cluster bootstrap over source games is a bootstrap over
    # `keep` -- the clusters are singletons here, unlike the per-position tables.
    draws = detector.resamples(len(keep), BOOT_SEED)
    scores = {s: scored(s, cand, sel, keep) for s in SPACES}
    out = pd.DataFrame([row(s, scores[s], draws) for s in SPACES])
    out = out.sort_values("auroc", ascending=False).reset_index(drop=True)
    return out, paired(scores, out.space.iloc[0], draws)


def bars(ax, table: pd.DataFrame) -> None:
    """The by-space bars, on a caller's axes, so a figure that pairs them with the score
    distributions draws this and not a copy of its argument list."""
    panel(ax, table, "auroc",
          "AUROC at $t_0$, illegal vs matched legal splice\n($\\uparrow$ better)",
          0.5, (table.auroc_null.min(), table.auroc_null.max()), None, 1.05,
          "chance (empirical)", roster.styles())


def draw(table: pd.DataFrame, camera_ready: bool) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars(ax, table)
    return finish(fig, "auroc_by_space_l0_matched", camera_ready, detector.FIGURES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    rows, leads = table()
    detector.RESULTS.mkdir(parents=True, exist_ok=True)
    csv = detector.RESULTS / "auroc_by_space_l0_matched.csv"
    rows.round(5).to_csv(csv, index=False)
    leads.round(5).to_csv(detector.RESULTS / "auroc_by_space_l0_matched_paired.csv", index=False)
    print(rows.round(4).to_string(index=False))
    print(leads.round(4).to_string(index=False))
    print(csv)
    print(draw(rows, args.camera_ready))


if __name__ == "__main__":
    main()
