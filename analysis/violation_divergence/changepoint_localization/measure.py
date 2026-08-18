"""Score every branch of the divergence tree with the reference-free change-point statistic.

The window and the paired/unpaired choice are picked on the *straightness* violation corpus
and applied here untouched, so the reported numbers are not tuned on what they report. The
two corpora are not independent -- 501 of the 573 scored games share their source legal game
with a selection-corpus violation -- but the splices differ (straightness resamples the suffix
and injects only empty-square violations; this tree keeps the suffix and carries R1 and a
matched legal control), and re-running the grid here afterwards puts the same setting on top,
so the selection is not doing the work.

The per-move-index profile that "normal" means is fit on straightness legal games minus the
ones that ARE divergence games (`meta.held_out_legal`), so no scored trajectory contributed
to the baseline it is scored against.

Arms come from `spade.move_legality.localize.arms`, which already owns the seeded
one-branch-per-game rule and makes these rows column-comparable with that study's tables.

Writes results/changepoint_localization.csv.

Usage: python analysis/violation_divergence/changepoint_localization/measure.py [--seed 42]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spade.common.study import study_dir
from spade.figures.paper_style import DICTIONARIES, REFERENCE_SPACE
from spade.move_legality import localize
from spade.straightness.build_games import CLASSES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from changepoint import (  # noqa: E402  (sibling module, not an installed package)
    auroc, candidates, ci, drop, located, profile, resamples, scan)

HERE = Path(__file__).resolve().parent
SPACES = [REFERENCE_SPACE] + DICTIONARIES
M_GRID = (4, 6, 8, 10, 12)
N_LEGAL = 2000                       # the straightness legal block, and the violation block
VIOLATION = CLASSES.index("violation")
ARMS = ["illegal", "control", "control2", "reference", *localize.RULES]
NEAR = 1                             # plies of tolerance for the within-+-1 localization rate


def npz_stem(space: str) -> str:
    return "resid_post_5" if space == REFERENCE_SPACE else f"sae_{space}"


def steps(directory: Path, space: str, suffix: str = "") -> np.ndarray:
    return np.load(directory / f"{npz_stem(space)}{suffix}.npz")["d"].astype(np.float64)


def selection_corpus(space: str, cand: np.ndarray):
    """(fit d, violation d, source-legal d, t0) from the straightness study.

    Fit on even-index legal games and score odd-index ones, so the negative arm is outside
    the baseline too. The negative arm is the *unspliced* source game -- the straightness
    corpus has no legal-splice control -- so the grid is scored on a vs-reference-shaped
    task even though the headline here is vs-control.
    """
    games = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)
    d = steps(study_dir("straightness") / "geometry", space, "_raw")
    block = np.arange(N_LEGAL) + VIOLATION * N_LEGAL
    src, t0 = games["src_game"][block].numpy(), games["t0"][block].numpy()
    keep = (src % 2 == 1) & (t0 >= cand[0] + 2) & (t0 <= cand[-1] + 2)
    return d[np.arange(0, N_LEGAL, 2)], d[block[keep]], d[src[keep]], t0[keep]


def sweep(space: str, cand: np.ndarray) -> list[dict]:
    fit, viol, legal, t0 = selection_corpus(space, cand)
    mu, sd = profile(fit)
    rows = []
    for paired in (False, True):
        for m in M_GRID:
            s = {k: scan(drop(x, mu, sd), cand, m, paired)
                 for k, x in (("violation", viol), ("legal", legal))}
            rows.append({"space": space, "paired": paired, "m": m, "n_games": len(t0),
                         "auroc": auroc(s["violation"].max(1), s["legal"].max(1)),
                         "hit_rate": float(np.mean(located(s["violation"], cand) == t0))})
    return rows


def scores(space: str, meta: dict, selection: dict, cand: np.ndarray, m: int, paired: bool):
    """Per arm: the peak of the blind scan, and whether its argmax found t0."""
    legal = steps(study_dir("straightness") / "geometry", space, "_raw")[:N_LEGAL]
    held = set(meta["held_out_legal"].tolist())
    mu, sd = profile(legal[[i for i in range(N_LEGAL) if i not in held]])

    d = steps(study_dir("divergence") / "curvature_discriminant", space)
    t0 = meta["t0"][selection["reference"]]
    s = {arm: scan(drop(d[selection[arm]], mu, sd), cand, m, paired) for arm in ARMS}
    return ({arm: v.max(1) for arm, v in s.items()},
            {arm: located(v, cand) == t0 for arm, v in s.items()},
            {arm: np.abs(located(v, cand) - t0) <= NEAR for arm, v in s.items()})


def row(space: str, peak, hit, near, ref_peak, draws, cand, m, paired) -> dict:
    """One space's line. Detection CIs are over games; the vs-raw difference reuses the same
    draws, because the claim "this dictionary clears raw" is about a difference and two
    marginal CIs that overlap do not settle it."""
    against = lambda a, b, p=peak: (lambda i: auroc(p[a][i], p[b][i]))
    out = {"space": space, "statistic": "MOSUM" if paired else "forward window", "m": m,
           "n_games": len(peak["illegal"]), "chance_uniform": 1.0 / len(cand)}
    for name, (a, b) in {"vs_control": ("illegal", "control"),
                         "vs_reference": ("illegal", "reference"),
                         "null": ("control2", "control")}.items():
        lo, hi = ci(against(a, b), draws)
        out |= {f"auroc_{name}": against(a, b)(np.arange(out["n_games"])),
                f"auroc_{name}_lo": lo, f"auroc_{name}_hi": hi}
    delta = lambda i: against("illegal", "control")(i) - against("illegal", "control", ref_peak)(i)
    lo, hi = ci(delta, draws)
    out |= {"auroc_minus_raw": delta(np.arange(out["n_games"])),
            "auroc_minus_raw_lo": lo, "auroc_minus_raw_hi": hi}
    for arm in ARMS:
        out[f"hit_{arm}"] = float(hit[arm].mean())
    out[f"hit_illegal_within_{NEAR}"] = float(near["illegal"].mean())
    out["hit_illegal_lo"], out["hit_illegal_hi"] = ci(
        lambda i: float(hit["illegal"][i].mean()), draws)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    width = steps(study_dir("divergence") / "curvature_discriminant", REFERENCE_SPACE).shape[1]
    cand = candidates(width, max(M_GRID))
    print(f"scanning move indices {cand[0] + 2}..{cand[-1] + 2} "
          f"({len(cand)} candidates, uniform chance {1 / len(cand):.4f})")

    picks = pd.DataFrame([r for space in SPACES for r in sweep(space, cand)])
    grid = picks.groupby(["paired", "m"])[["auroc", "hit_rate"]].mean()
    best = grid.sum(axis=1).idxmax()
    paired, m = bool(best[0]), int(best[1])
    margin = grid.sum(axis=1).nlargest(2)
    print(f"selection corpus ({picks.n_games.iloc[0]} straightness violations): "
          f"{'MOSUM' if paired else 'forward window'}, m = {m}; "
          f"AUROC + hit beats the runner-up by {margin.iloc[0] - margin.iloc[1]:.4f}")
    print(grid.to_string())

    meta = dict(np.load(study_dir("divergence") / "curvature_discriminant" / "meta.npz"))
    torch_meta = {k: torch.from_numpy(v) for k, v in meta.items() if k != "held_out_legal"}
    selection = {k: v.numpy() for k, v in localize.arms(torch_meta, args.seed).items()}
    draws = resamples(len(selection["reference"]), args.seed)

    computed = {s: scores(s, meta, selection, cand, m, paired) for s in SPACES}
    ref_peak = computed[REFERENCE_SPACE][0]
    table = pd.DataFrame([row(s, *computed[s], ref_peak, draws, cand, m, paired)
                          for s in SPACES])

    (HERE / "results").mkdir(exist_ok=True)
    csv = HERE / "results" / "changepoint_localization.csv"
    table.to_csv(csv, index=False)
    picks.to_csv(HERE / "results" / "window_selection.csv", index=False)
    print(f"\nwrote {csv}")
    print(table[["space", "auroc_vs_control", "auroc_minus_raw", "auroc_minus_raw_lo",
                 "auroc_minus_raw_hi", "auroc_null", "hit_illegal",
                 f"hit_illegal_within_{NEAR}", "hit_control"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
