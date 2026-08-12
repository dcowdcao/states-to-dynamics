"""Step 4: the tidy CSVs the figures draw. Every statistic in this study is computed here.

The bootstrap unit is the source game (2000 percentile resamples, seed 0): a position is
not an independent draw, since 56 of them share one game's prefix.

Two things worth knowing before reading a skew number.

**The null is per-position matched.** A rule's *availability* swings by an order of
magnitude across a game -- R1 owns 5 of 60 squares at ply 1 and ~55 at ply 56 -- so a rule
share pooled against a pooled square count would mostly measure that drift. Every `expected`
column here is instead the model's own illegal mass at each position, redistributed
uniformly over that position's illegal squares, and summed. Ratio 1.0 is "the model spreads
its illegal mass the way the board offers it".

**The two survival arms are not the same experiment.** `teacher_forced` composes the
per-position mass of a model that never sees its own output; `free_running` samples whole
games and lets the error compound. The gap between them is the cost of self-conditioning,
and neither one is a substitute for the other.

Usage: python -m spade.generation_legality.aggregate
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spade.common.config import CTX_LEN
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.generation_legality.positions import N_SQUARES, POSITIONS
from spade.generation_legality.rollout_scan import N_GAMES
from spade.othello.rollout_bank import load as load_bank
from spade.othello.taxonomy import CATEGORIES, RULE_MEMBERS, RULE_OF

N_BOOT = 2000
BOOT_SEED = 0
PLAYED_BIN = 8  # plies per bin for the greedy/sampled rates; per-t counts are ~1
RULES = [rule.split()[0] for rule in RULE_MEMBERS]
RULE_CATEGORIES = {rule.split()[0]: members for rule, members in RULE_MEMBERS.items()}
ARMS = ["mass", "greedy", "sampled"]


def weights(n_games: int, rng: np.random.Generator) -> np.ndarray:
    """[N_BOOT, n_games] multinomial resample counts, as in legality_probe.readout."""
    return rng.multinomial(n_games, np.full(n_games, 1 / n_games), size=N_BOOT).astype(float)


def mean_ci(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    draws = w @ x / x.shape[0]
    return x.mean(0), *np.percentile(draws, [2.5, 97.5], axis=0)


def ratio_ci(num: np.ndarray, den: np.ndarray,
             w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A rule with no available square at this ply, or an arm that never plays an illegal
    # move, has an empty denominator: that is a NaN skew, not a bug.
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (w @ num) / (w @ den)
        return num.sum(0) / den.sum(0), *np.percentile(draws, [2.5, 97.5], axis=0)


def nan_mean_ci(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`mean_ci` where a game contributes to some columns and not others.

    The rollout table is ragged: a game leaves at its first illegal move, so column t is a
    mean over the games still alive at t. One weight matrix is still used across the whole
    curve -- the same resampled games, with dead cells simply not contributing -- rather
    than an independent resample per column, which would decouple neighbouring points.
    """
    alive = np.isfinite(x).astype(float)
    filled = np.nan_to_num(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (w @ filled) / (w @ alive)
    return np.nanmean(x, axis=0), *np.percentile(draws, [2.5, 97.5], axis=0)


def pivot(frame: pd.DataFrame, column: str) -> np.ndarray:
    """[n_games, len(POSITIONS)] view of a per-position column."""
    return frame.pivot(index="game_id", columns="t", values=column).to_numpy()


def per_game(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every per-position quantity as an [n_games, n_moves] array, plus the matched null."""
    names = RULES + CATEGORIES[1:]
    out = {c: pivot(frame, c) for c in
           ["p_legal", "p_pass", "n_legal", *(f"p_{n}" for n in names),
            *(f"n_{n}" for n in names)]}
    out["p_illegal"] = sum(out[f"p_{r}"] for r in RULES)
    assert (out["p_illegal"] > 0).all(), "an illegal mass underflowed to zero"
    n_illegal = N_SQUARES - out["n_legal"]
    out["uniform_illegal"] = n_illegal / N_SQUARES
    for name in names:
        out[f"expected_{name}"] = out["p_illegal"] * out[f"n_{name}"] / n_illegal
    for arm in ["greedy", "sampled"]:
        category = pivot(frame, f"{arm}_category")
        out[f"{arm}_pass"] = (category == "pass").astype(float)
        out[f"{arm}_illegal"] = (~np.isin(category, ["legal", "pass"])).astype(float)
        for name in names:
            members = RULE_CATEGORIES[name] if name in RULES else [name]
            out[f"{arm}_{name}"] = np.isin(category, members).astype(float)
            out[f"{arm}_expected_{name}"] = (out[f"{arm}_illegal"]
                                             * out[f"n_{name}"] / n_illegal)
    return out


def by_move(game: dict, w: np.ndarray) -> pd.DataFrame:
    n_games = len(w[0])
    rows = []
    for metric in ["p_legal", "p_illegal", "p_pass", "uniform_illegal",
                   *(f"p_{r}" for r in RULES), *(f"p_{c}" for c in CATEGORIES[1:])]:
        rows.append((metric, *mean_ci(game[metric], w)))
    for name in RULES + CATEGORIES[1:]:
        rows.append((f"skew_{name}", *ratio_ci(game[f"p_{name}"], game[f"expected_{name}"], w)))
    return pd.DataFrame([
        {"t": int(t), "metric": metric, "mean": point[i], "lo": lo[i], "hi": hi[i],
         "n_games": n_games}
        for metric, point, lo, hi in rows for i, t in enumerate(POSITIONS)])


def rollout_by_move(frame: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    """`by_move` over the self-generated prefixes of `rollout_scan`, with n alive per ply.

    Same metrics and same estimator as `by_move`; only the prefix distribution differs. The
    skew columns are not recomputed here -- the by-rule figure draws mass, and a matched-null
    ratio conditional on survival is a different statistic that nothing plots.
    """
    n_games = len(w[0])

    def grid(metric: str) -> np.ndarray:
        """[n_games, len(POSITIONS)], NaN where the game had already gone illegal.

        Reindexed rather than pivoted straight: a game whose first move is illegal has no
        rows at all and would otherwise drop out of the array the weights index into.
        """
        return (frame.pivot(index="game_id", columns="t", values=metric)
                .reindex(index=range(n_games), columns=list(POSITIONS)).to_numpy())

    values = {m: grid(m) for m in ["p_legal", *(f"p_{n}" for n in RULES + CATEGORIES[1:])]}
    values["p_illegal"] = sum(values[f"p_{r}"] for r in RULES)
    alive = np.isfinite(values["p_legal"]).sum(0)
    rows = [(metric, *nan_mean_ci(value, w)) for metric, value in values.items()]
    return pd.DataFrame([
        {"t": int(t), "metric": metric, "mean": point[i], "lo": lo[i], "hi": hi[i],
         "n_games": int(alive[i]), "n_games_total": n_games}
        for metric, point, lo, hi in rows for i, t in enumerate(POSITIONS)])


def played_rate(game: dict, w: np.ndarray) -> pd.DataFrame:
    """Greedy/sampled illegal rates in PLAYED_BIN-ply bins; per-t counts are ~1."""
    n_games, edges = len(w[0]), range(POSITIONS.start, POSITIONS.stop, PLAYED_BIN)
    rows = []
    for arm in ["greedy", "sampled"]:
        binned = np.stack([game[f"{arm}_illegal"][:, lo : lo + PLAYED_BIN].mean(1)
                           for lo in edges], axis=1)
        point, lo_ci, hi_ci = mean_ci(binned, w)
        rows += [{"arm": arm, "t_lo": lo, "t_hi": min(lo + PLAYED_BIN, POSITIONS.stop) - 1,
                  "rate": point[i], "lo": lo_ci[i], "hi": hi_ci[i], "n_games": n_games,
                  "n_illegal": int(game[f"{arm}_illegal"][:, lo : lo + PLAYED_BIN].sum())}
                 for i, lo in enumerate(edges)]
    return pd.DataFrame(rows)


def rule_skew(game: dict, w: np.ndarray) -> pd.DataFrame:
    """Observed share of illegal mass (or of illegal plays) against the matched null."""
    n_games, rows = len(w[0]), []
    for arm in ARMS:
        for name in RULES + CATEGORIES[1:]:
            observed = game[f"p_{name}"] if arm == "mass" else game[f"{arm}_{name}"]
            expected = (game[f"expected_{name}"] if arm == "mass"
                        else game[f"{arm}_expected_{name}"])
            total = game["p_illegal"] if arm == "mass" else game[f"{arm}_illegal"]
            ratio, lo, hi = ratio_ci(observed.sum(1, keepdims=True),
                                     expected.sum(1, keepdims=True), w)
            events = total.sum()
            rows.append({
                "arm": arm, "level": "rule" if name in RULES else "category",
                "name": name,
                "rule": name if name in RULES else RULE_OF[name].split()[0],
                "share_observed": observed.sum() / events if events else np.nan,
                "share_expected": expected.sum() / events if events else np.nan,
                "ratio": ratio[0], "lo": lo[0], "hi": hi[0], "n_games": n_games,
                "n_events": events, "n_positions": observed.size})
    return pd.DataFrame(rows)


def spread(share: np.ndarray, rule: np.ndarray) -> float:
    """Within-rule / between-rule spread of a per-category quantity, in log10 (GUIDELINES 1.7).

    Near 1 the three rules are a label rather than a finding, and the figure drawing them has
    to say so. The skew answers it one way and the raw mass share another, because a
    category's matched null carries its own availability; both are reported.
    """
    value = np.log10(share)
    means = np.array([value[rule == r].mean() for r in RULES])
    within = np.sqrt(sum(((value[rule == r] - m) ** 2).sum() for r, m in zip(RULES, means))
                     / (len(value) - len(RULES)))
    return within / means.std(ddof=1)


def category_shares(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(share of illegal mass per category, its rule), pooled over every row of a scan."""
    mass = np.array([frame[f"p_{c}"].sum() for c in CATEGORIES[1:]])
    rules = np.array([RULE_OF[c].split()[0] for c in CATEGORIES[1:]])
    return mass / mass.sum(), rules


def survival(game: dict, rollouts: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    """P(at least one illegal move in the first k plies), both arms."""
    n_games = len(w[0])
    legal_run = np.cumprod(1 - game["p_illegal"], axis=1)
    point, lo, hi = mean_ci(1 - legal_run, w)
    rows = [{"arm": "teacher_forced", "plies": 1, "value": 0.0, "lo": 0.0, "hi": 0.0,
             "n_games": n_games}]
    rows += [{"arm": "teacher_forced", "plies": t + 2, "value": point[i], "lo": lo[i],
              "hi": hi[i], "n_games": n_games} for i, t in enumerate(POSITIONS)]

    first = rollouts.first_illegal.to_numpy()
    hit = np.stack([((first >= 0) & (first < k)).astype(float)
                    for k in range(1, CTX_LEN + 1)], axis=1)
    point, lo, hi = mean_ci(hit, weights(len(first), np.random.default_rng(BOOT_SEED)))
    rows += [{"arm": "free_running", "plies": k, "value": point[k - 1], "lo": lo[k - 1],
              "hi": hi[k - 1], "n_games": len(first)} for k in range(1, CTX_LEN + 1)]
    return pd.DataFrame(rows)


def overall(game: dict, rollouts: pd.DataFrame, skew: pd.DataFrame, rollout_share: np.ndarray,
            w: np.ndarray) -> pd.DataFrame:
    n_games, rows = len(w[0]), []
    for metric in ["p_legal", "p_illegal", "p_pass", "uniform_illegal",
                   *(f"p_{r}" for r in RULES), "greedy_illegal", "sampled_illegal",
                   "greedy_pass", "sampled_pass"]:
        point, lo, hi = mean_ci(game[metric].mean(1, keepdims=True), w)
        rows.append({"metric": metric, "value": point[0], "lo": lo[0], "hi": hi[0],
                     "n_games": n_games, "n": game[metric].size})
    index = np.random.default_rng(BOOT_SEED).integers(0, n_games, (N_BOOT, n_games))
    for label, quantile in [("median", 50), ("q25", 25), ("q75", 75)]:
        for name in ["p_illegal", "uniform_illegal"]:
            value = game[name]
            draws = [np.percentile(value[i], quantile) for i in index]
            rows.append({"metric": f"{name}_{label}", "value": np.percentile(value, quantile),
                         "lo": np.percentile(draws, 2.5), "hi": np.percentile(draws, 97.5),
                         "n_games": n_games, "n": value.size})
    clean = (rollouts.first_illegal.to_numpy() < 0).astype(float)[:, None]
    point, lo, hi = mean_ci(clean, weights(len(clean), np.random.default_rng(BOOT_SEED)))
    rows.append({"metric": "free_running_fully_legal", "value": point[0], "lo": lo[0],
                 "hi": hi[0], "n_games": len(clean), "n": len(clean)})
    members = skew[(skew.arm == "mass") & (skew.level == "category")]
    rule = members.rule.to_numpy()
    for metric, share in [("rule_skew_within_over_between", members.ratio.to_numpy()),
                          ("rule_mass_within_over_between", members.share_observed.to_numpy()),
                          ("rollout_mass_within_over_between", rollout_share)]:
        rows.append({"metric": metric, "value": spread(share, rule), "lo": np.nan,
                     "hi": np.nan, "n_games": n_games, "n": len(CATEGORIES) - 1})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=repo_root() / "analysis" / "generation_legality" / "results")
    args = parser.parse_args()

    base = study_dir("generation")
    frame = pd.read_csv(base / "scan" / "positions.csv")
    _, first_illegal = load_bank(N_GAMES)
    rollouts = pd.DataFrame({"first_illegal": first_illegal.numpy()})
    rollout_frame = pd.read_csv(base / "rollout" / "positions.csv")
    game = per_game(frame)
    rng = np.random.default_rng(BOOT_SEED)
    w = weights(len(game["p_illegal"]), rng)

    skew = rule_skew(game, w)
    tables = {
        "by_move": by_move(game, w),
        "rollout_by_move": rollout_by_move(
            rollout_frame, weights(len(rollouts), np.random.default_rng(BOOT_SEED))),
        "played_rate": played_rate(game, w),
        "rule_skew": skew,
        "survival": survival(game, rollouts, w),
        "overall": overall(game, rollouts, skew, category_shares(rollout_frame)[0], w),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.results / f"{name}.csv", index=False)
    print(f"wrote {len(tables)} CSVs to {args.results}")
    print(tables["overall"].to_string(index=False))
    print(skew[skew.level == "rule"].to_string(index=False))


if __name__ == "__main__":
    main()
