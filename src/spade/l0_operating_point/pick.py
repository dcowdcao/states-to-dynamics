"""One shared L0 band, and each type's best config inside it.

    python -m spade.l0_operating_point.pick

The band is searched, not asserted: every (lo, hi) on a log grid is scored by the mean of
its per-type winners, and the best one wins. A band that cannot seat all eight types is not
a common band and is skipped, which is what keeps the answer honest about CosineSpaDE — it
cannot leave L0 [19.4, 178.4] at any gamma, so any band above that ceiling would be a
comparison with a type missing.

**The width cap is load-bearing, not a tuning knob.** The mean of per-type winners is
monotone in width — a wider band only ever adds candidates, so it can never score lower —
and without a cap the search returns the widest band that seats eight types (L0 55-506,
a 9x spread). That is the L0 confound this study exists to remove. MAX_WIDTH fixes what
"matched sparsity" is allowed to mean and the search picks only the location.

Filtered to one dictionary size by default. His sweep runs both expansions and ours are all
16x = 8192; a 4096 winner against our 8192 points swaps one confound for another.

Reports the min-max pick and the rank-mean pick side by side. A type where they disagree is
not resolved by this table.
"""

import argparse
import numpy as np
import pandas as pd

from spade.common.paths import repo_root
from spade.l0_operating_point import score
from spade.l0_operating_point.aggregate import OUT as POOLED, SCORED

PICKS = repo_root() / "results/l0_operating_point/picks.csv"

# Karvonen's axis runs L0 10-571.
EDGES = np.geomspace(10, 600, 41)
# 2.5x. Wide enough to seat eight architectures whose sparsity knobs land where they land,
# narrow enough that "matched L0" still means something; comparable to the +/-25% (1.67x)
# window karvonen_comparison used for its matched-L0 tables.
MAX_WIDTH = 2.5


def winners(df: pd.DataFrame, lo: float, hi: float, column: str = score.MINMAX) -> pd.DataFrame:
    band = df[(df["l0"] >= lo) & (df["l0"] <= hi)]
    if band["trainer_class"].nunique() < df["trainer_class"].nunique():
        return pd.DataFrame()
    return band.loc[band.groupby("trainer_class")[column].idxmax()]


def per_type_best(df: pd.DataFrame, column: str = score.MINMAX) -> pd.DataFrame:
    """Each type's own best config, ignoring every other type. Stage 1 of `consensus`."""
    return df.loc[df.groupby("trainer_class")[column].idxmax()].sort_values("l0")


def consensus_target(df: pd.DataFrame, column: str = score.MINMAX) -> tuple[float, pd.DataFrame]:
    """The mean of the eight per-type optimal L0s.

    Not searched, derived: every type first says where it would like to sit if nothing else
    constrained it, and the shared target is the average of those eight answers. So the
    target is a compromise the types themselves define, rather than the point that happens
    to maximise a pooled objective -- which is what `search_target` does, and which lets a
    type with a flat score curve drag the target toward a type with a peaked one.

    Arithmetic mean, per the stated rule. Note it is not scale-free: on an axis this wide a
    single dense optimum pulls the mean more than a sparse one, and `main` prints the eight
    inputs so that pull is visible rather than buried in one number."""
    best = per_type_best(df, column)
    return float(best["l0"].mean()), best


def nearest_winners(df: pd.DataFrame, target: float,
                    max_width: float | None = MAX_WIDTH) -> pd.DataFrame:
    """Each type's config whose L0 sits closest to `target`, in log space.

    The other rule: `winners` takes each type's best-scoring config anywhere in a band,
    which lets the eight sit at different L0s inside it. This one holds L0 as fixed as
    each grid allows and accepts whatever score that config happens to have — a tighter
    matched-sparsity claim, and a worse-scoring one, because a type is not free to move to
    its own optimum.

    With max_width set, every type must have a config within sqrt(max_width) of the target
    or the target is rejected, so no type is seated by something nowhere near it. With
    max_width None the seating is unconditional -- `consensus` uses that, because its target
    is derived from the types rather than searched, so there is no candidate to reject and a
    type that cannot reach it is a finding. `l0_over_target` in the output is what exposes
    that, and `main` sorts by it."""
    if max_width is None:
        distance = (np.log(df["l0"]) - np.log(target)).abs()
        return df.loc[distance.groupby(df["trainer_class"]).idxmin()]
    reach = np.sqrt(max_width)
    band = df[(df["l0"] >= target / reach) & (df["l0"] <= target * reach)]
    if band["trainer_class"].nunique() < df["trainer_class"].nunique():
        return pd.DataFrame()
    distance = (np.log(band["l0"]) - np.log(target)).abs()
    return band.loc[distance.groupby(band["trainer_class"]).idxmin()]


def search_target(df: pd.DataFrame, max_width: float = MAX_WIDTH,
                  column: str = score.MINMAX) -> tuple[float, pd.DataFrame]:
    """Only the target depends on the score here — given a target, each type's config is
    fixed by L0 alone. So the rank-mean twin can only move the target, not the seating."""
    best = None
    for target in EDGES:
        won = nearest_winners(df, target, max_width)
        if won.empty:
            continue
        key = won[column].mean()
        if best is None or key > best[0]:
            best = (key, target, won)
    if best is None:
        raise ValueError(f"no target L0 seats all eight types within sqrt({max_width})x")
    _, target, won = best
    return target, won


def search(df: pd.DataFrame, max_width: float = MAX_WIDTH) -> tuple[float, float, pd.DataFrame]:
    best = None
    for i, lo in enumerate(EDGES):
        for hi in EDGES[i + 1:]:
            if hi / lo > max_width:
                break
            won = winners(df, lo, hi)
            if won.empty:
                continue
            # Ties go to the narrower band: it is the tightest honest claim about matched
            # sparsity that still holds the same winners.
            key = (round(won[score.MINMAX].mean(), 6), -hi / lo)
            if best is None or key > best[0]:
                best = (key, lo, hi, won)
    if best is None:
        raise ValueError(f"no band of width <= {max_width}x seats all eight types")
    _, lo, hi, won = best
    return lo, hi, won


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooled", type=argparse.FileType("r"), default=str(POOLED))
    ap.add_argument("--out", default=str(PICKS))
    ap.add_argument("--max-width", type=float, default=MAX_WIDTH,
                    help="widest hi/lo the band may take; see the module docstring")
    ap.add_argument("--dict-size", type=int, default=8192,
                    help="0 to pool both of his expansion factors with our 16x points")
    ap.add_argument("--rule", choices=("consensus", "band", "nearest"), default="consensus",
                    help="consensus: target = mean of the eight per-type optimal L0s, then "
                         "each type seats at its config nearest that. band: each type's "
                         "best-scoring config inside a shared band. nearest: searches for "
                         "the target that maximises the mean of the nearest-seated configs.")
    args = ap.parse_args()

    df = score.per_config(pd.read_csv(args.pooled))
    if args.dict_size:
        df = df[df["dict_size"] == args.dict_size]
    df = score.add_scores(df)

    cols = ["family", "trainer_class", "config", "l0", "l0_std", *SCORED,
            score.MINMAX, score.RANK, "n_seeds", "eval_n_activations"]
    if args.rule == "consensus":
        target, best = consensus_target(df)
        print("stage 1 - each type's own optimum, ignoring the others:\n")
        print(best[["family", "trainer_class", "config", "l0", score.MINMAX, score.RANK]]
              .to_string(index=False))
        print(f"\nmean of those {len(best)} L0s -> shared target L0 {target:.1f}"
              f"   (median {best['l0'].median():.1f}, range {best['l0'].min():.1f}"
              f"-{best['l0'].max():.1f})\n")
        won = nearest_winners(df, target, max_width=None)
        won = won.assign(l0_over_target=won["l0"] / target)
        cols.insert(5, "l0_over_target")
        lo, hi = won["l0"].min(), won["l0"].max()
        print("stage 2 - each type's config nearest that target:\n")
    elif args.rule == "nearest":
        target, won = search_target(df, args.max_width)
        lo, hi = target / np.sqrt(args.max_width), target * np.sqrt(args.max_width)
        won = won.assign(l0_over_target=won["l0"] / target)
        cols.insert(5, "l0_over_target")
        print(f"shared target L0 {target:.1f}  (each type's nearest config, "
              f"admissible within {np.sqrt(args.max_width):.2f}x)\n")
    else:
        lo, hi, won = search(df, args.max_width)
        n = df[(df.l0 >= lo) & (df.l0 <= hi)].shape[0]
        target = float("nan")
        print(f"band: L0 {lo:.1f} - {hi:.1f}  ({hi / lo:.2f}x, {n} configs "
              f"at dict {args.dict_size})\n")

    sort_by = "l0_over_target" if args.rule == "consensus" else score.MINMAX
    print(won[cols].sort_values(sort_by, ascending=(sort_by != score.MINMAX))
          .to_string(index=False))
    print(f"\nmean score across the eight: {won[score.MINMAX].mean():.4f}")

    if args.rule == "consensus":
        rank_target, rank_best = consensus_target(df, score.RANK)
        rank_won = nearest_winners(df, rank_target, max_width=None) \
            .set_index("trainer_class")["config"]
        print(f"rank-mean twin derives target L0 {rank_target:.1f} (vs {target:.1f})")
    elif args.rule == "nearest":
        rank_target, rank_frame = search_target(df, args.max_width, score.RANK)
        print(f"rank-mean twin puts the target at L0 {rank_target:.1f} (vs {target:.1f})")
        rank_won = rank_frame.set_index("trainer_class")["config"]
    else:
        rank_won = winners(df, lo, hi, score.RANK).set_index("trainer_class")["config"]
    minmax = won.set_index("trainer_class")["config"]
    disagree = minmax.index[minmax != rank_won.reindex(minmax.index)]
    print("rank-mean twin picks a different config for: "
          + (", ".join(f"{t} ({minmax[t]} -> {rank_won[t]})" for t in disagree) or "nothing"))

    print("\nwhere the score's spread comes from:")
    print(score.spread(df).to_string())

    out = won.assign(rule=args.rule, band_lo=lo, band_hi=hi, target_l0=target,
                     agrees_with_rank_mean=~won["trainer_class"].isin(disagree))
    out.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
