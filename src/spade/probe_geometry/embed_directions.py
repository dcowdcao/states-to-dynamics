"""Where the 188 probe directions sit relative to each other, and how much of a
rules axis the board's own subspace already holds.

Two rules arms, and the difference between them is the study's result. The
committed probes are square-agnostic: one legality direction read off the
residual wherever the candidate move was spliced in. Every board direction is
square-specific. Pooling over squares is what makes the two families look
orthogonal, so `square_rules_directions` refits legality separately per
candidate square and both arms are carried through every table here.

Unit vectors have no meaningful mean: the origin is a real point (a zero
contrast is "no preference"), and the mean direction is itself a result rather
than a nuisance offset. So every PCA here is UNCENTRED, implemented as an SVD of
the stacked directions — `numpy.linalg.svd(X)`, no column means removed.

Two fits, because they answer different questions:

  joint  all 188 directions, for the scatter. An uncentred joint fit is a
         weighted vote and the weights are the family sizes, so the 124 board
         directions set the components and the rules directions ride along in
         whatever coordinates that basis gives them. That is fine for a picture
         of the arrangement and useless as evidence.

  board  the 124 board directions alone, for the capture curve. Each rules
         direction is *projected into* that basis, never fitted to it, which is
         the only version of the question "how much of this axis does the board
         subspace already contain" that the fit cannot answer for itself.

The capture null is empirical and it is not zero: a random unit vector in 512-d
puts k/512 of its squared norm in any fixed k-dimensional subspace, so the board
subspace captures a quarter of a random direction by the time k reaches its rank
of 124. The comparison the figure rests on is against that curve.

Usage: python src/probe_geometry/embed_directions.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


from spade.common.paths import repo_root
from spade.probe_geometry.probes import (
    BOARD_FAMILIES, D_MODEL, Direction, committed_directions, rules_corpus,
    square_rules_directions, stack)

RESULTS = repo_root() / "analysis" / "probe_geometry" / "results"
NULL_SEED = 0
N_CAPTURE_NULL = 2000   # draws behind the capture band
N_SCATTER_NULL = 128    # drawn as a cloud, so matched to the direction count
CI = (2.5, 97.5)
LAM_SQUARE = 10.0  # ridge for the per-square legality refit; ~2.9k rows per square


def basis(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uncentred principal directions of the rows of X, and their variance ratios."""
    _, s, components = np.linalg.svd(X, full_matrices=False)
    return components, s**2 / (s**2).sum()


def random_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal((n, D_MODEL))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def coords_frame(directions: list[Direction], components: np.ndarray,
                 null: np.ndarray) -> pd.DataFrame:
    xyz = stack(directions) @ components[:3].T
    named = pd.DataFrame({
        "family": [d.family for d in directions],
        "name": [d.name for d in directions],
        "square": [d.square for d in directions]})
    rows = pd.DataFrame({
        "family": "null", "name": [f"null@{i}" for i in range(len(null))],
        "square": None})
    out = pd.concat([named, rows], ignore_index=True)
    out[["pc1", "pc2", "pc3"]] = np.vstack([xyz, null @ components[:3].T])
    return out


def capture(directions: list[Direction], board_components: np.ndarray) -> np.ndarray:
    """[n, k] fraction of each direction's squared norm inside the top-k board PCs."""
    return np.cumsum((stack(directions) @ board_components.T)**2, axis=1)


def band(name: str, curves: np.ndarray) -> pd.DataFrame:
    lo, hi = np.percentile(curves, CI, axis=0)
    return pd.DataFrame({"name": name, "k": np.arange(1, curves.shape[1] + 1),
                         "captured": curves.mean(0), "lo": lo, "hi": hi,
                         "n": len(curves)})


def capture_frame(pooled: list[Direction], per_square: list[Direction],
                  board_components: np.ndarray, null: np.ndarray) -> pd.DataFrame:
    """Capture curves for both rules arms, against the random-direction null.

    The pooled arm is four single fits, so each is its own line with no band.
    The per-square arm has 60 members, so squares are the spread the band shows.
    """
    lines = [band(d.name, capture([d], board_components)) for d in pooled]
    return pd.concat(lines + [
        band("rules_square", capture(per_square, board_components)),
        band("null", np.cumsum((null @ board_components.T)**2, axis=1)),
    ], ignore_index=True)


def main():
    rules = rules_corpus()
    directions = committed_directions() + square_rules_directions(
        rules, np.arange(len(rules["acts"])), LAM_SQUARE)
    board = [d for d in directions if d.family in BOARD_FAMILIES]
    pooled = [d for d in directions if d.family in ("rules4", "rules2")]
    per_square = [d for d in directions if d.family == "rules_square"]
    rng = np.random.default_rng(NULL_SEED)

    joint, joint_var = basis(stack(directions))
    board_components, board_var = basis(stack(board))

    RESULTS.mkdir(parents=True, exist_ok=True)
    coords_frame(directions, joint, random_directions(N_SCATTER_NULL, rng)).to_csv(
        RESULTS / "pca_coords.csv", index=False)
    pd.concat([pd.DataFrame({"fit": fit, "component": np.arange(1, len(v) + 1),
                             "variance_ratio": v})
        for fit, v in [("joint", joint_var), ("board", board_var)]]).to_csv(
        RESULTS / "pca_variance.csv", index=False)
    capture_frame(pooled, per_square, board_components,
                  random_directions(N_CAPTURE_NULL, rng)).to_csv(
        RESULTS / "pca_capture.csv", index=False)

    (RESULTS / "pca_notes.json").write_text(json.dumps({
        "pca": "uncentred (SVD of the stacked unit directions, no mean removed)",
        "joint_fit": {"n": len(directions), "rank": int(len(joint_var)),
                      "note": f"dominated by the larger family: {len(board)} board "
                              f"vs {len(pooled) + len(per_square)} rules, which is "
                              "why the capture curve fits on the board alone"},
        "board_fit": {"n": len(board), "rank": int(len(board_var))},
        "per_square_rules": {"n": len(per_square), "lam": LAM_SQUARE,
                             "rows": int(len(rules["acts"]))},
        "capture_null": {"draws": N_CAPTURE_NULL, "ci_percentiles": list(CI),
                         "expected": "k/512 for a random unit direction in 512-d"},
        "scatter_null": {"draws": N_SCATTER_NULL,
                         "note": "projected into the joint basis, not fitted to it"},
        "seed": NULL_SEED,
    }, indent=2) + "\n")
    print(f"wrote pca_coords, pca_variance, pca_capture, pca_notes to {RESULTS}")


if __name__ == "__main__":
    main()
