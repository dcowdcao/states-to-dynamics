"""How far apart the six classes sit in the 21-D ReStraV descriptor, per space.

The PCA figure shows the classes overlapping; this puts a number on it. Three
things the numbers need in order to mean anything, and which the printed tables
carry:

- **Distances are standardized, never raw.** A centroid gap is divided by the
  pooled within-class spread (`sqrt` of the mean per-axis variance of the two
  classes), so it reads as a Cohen's d: 1.0 means the centroids sit one
  within-class SD apart. Raw gaps in a per-space basis are not comparable across
  the nine panels.
- **The plane is not the space.** The top-2 plane holds ~34% of the variance, so
  every pair is reported both in the plane the figure draws and in the full 21-D,
  where a separation the plane happens to miss still shows.
- **There is an empirical null.** Legal split in half by the study's own even/odd
  train mask gives two centroids of 1000 games each that differ only by sampling.
  That distance is the floor; anything near it is nothing. n = 2000 per class
  makes a centroid's own standard error ~0.02 SD, so significance is not the
  question -- size is.

Usage: python analysis/straightness/feature_pca/centroid_distances.py
"""

from itertools import combinations

import numpy as np

from spade.straightness.geometry import RESTRAV_FEATURE_NAMES

from make_figures import DRAWN, features, pca, spaces

BOOT, BOOT_SEED = 1000, 0


def standardized(x: np.ndarray, y: np.ndarray) -> float:
    """‖mean(x) − mean(y)‖ in units of the pooled within-class SD."""
    spread = np.sqrt(0.5 * (x.var(0).mean() + y.var(0).mean()))
    return float(np.linalg.norm(x.mean(0) - y.mean(0)) / spread)


def boot_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap over games, resampling each class independently."""
    g = np.random.default_rng(BOOT_SEED)
    vals = [standardized(x[g.integers(len(x), size=len(x))],
                         y[g.integers(len(y), size=len(y))]) for _ in range(BOOT)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def null(x: np.ndarray) -> float:
    """Legal against itself: even-index games vs odd, the study's own split."""
    even = np.arange(len(x)) % 2 == 0
    return standardized(x[even], x[~even])


def table(title: str, rows: dict, pairs: list) -> None:
    print(f"\n{title}")
    print(f"{'space':34}" + "".join(f"{a[:5] + '/' + b[:5]:>14}" for a, b in pairs)
          + f"{'legal null':>13}")
    for space, (dist, floor) in rows.items():
        print(f"{space:34}" + "".join(f"{dist[p]:14.2f}" for p in pairs)
              + f"{floor:13.2f}")


def main():
    pairs = list(combinations(DRAWN, 2))
    plane, full, raw_basis = {}, {}, None
    for label, entry in spaces():
        f21, labels = features(entry)
        scores, _, vecs = pca(f21, align=raw_basis)
        raw_basis = vecs if raw_basis is None else raw_basis
        z = (f21 - f21.mean(0)) / f21.std(0)
        by = {c: labels == c for c in DRAWN}
        plane[label] = ({p: standardized(scores[by[p[0]]], scores[by[p[1]]])
                         for p in pairs}, null(scores[by["legal"]]))
        full[label] = ({p: standardized(z[by[p[0]]], z[by[p[1]]]) for p in pairs},
                       null(z[by["legal"]]))

    table("Centroid distance in the top-2 PC plane (pooled within-class SD)", plane, pairs)
    table("Centroid distance in the full 21-D descriptor (z-scored)", full, pairs)

    print("\nRaw layer 5, bootstrap 95% CI over games (1000 resamples), full 21-D:")
    f21, labels = features(None)
    z = (f21 - f21.mean(0)) / f21.std(0)
    for a, b in pairs:
        lo, hi = boot_ci(z[labels == a], z[labels == b])
        print(f"  {a:16} vs {b:16} {standardized(z[labels == a], z[labels == b]):5.2f} "
              f"[{lo:.2f}, {hi:.2f}]")

    # A rule branch shares its source game's tokens before t0 >= 10, and 1998 of
    # the 2000 legal games are those source games, so the 13 leading components
    # should contribute nothing and the 8 moments should carry all of it.
    print("\nRaw layer 5, |Δ centroid| per component, legal vs each rule (z units):")
    print(f"{'component':>12}" + "".join(f"{r.split()[0]:>8}" for r in DRAWN[3:]))
    for i, name in enumerate(RESTRAV_FEATURE_NAMES):
        gaps = [abs(z[labels == r][:, i].mean() - z[labels == "legal"][:, i].mean())
                for r in DRAWN[3:]]
        print(f"{name:>12}" + "".join(f"{g:8.2f}" for g in gaps))

    rules = [p for p in pairs if p[0] in DRAWN[3:] and p[1] in DRAWN[3:]]
    to_legal = [p for p in pairs if p[0] == "legal" and p[1] in DRAWN[3:]]
    for name, rows in (("plane", plane), ("21-D", full)):
        among = np.mean([[d[p] for p in rules] for d, _ in rows.values()])
        against = np.mean([[d[p] for p in to_legal] for d, _ in rows.values()])
        floor = np.mean([f for _, f in rows.values()])
        print(f"\nmean over the nine spaces ({name}): rule-to-rule {among:.2f}, "
              f"rule-to-legal {against:.2f}, legal-vs-legal null {floor:.2f}")


if __name__ == "__main__":
    main()
