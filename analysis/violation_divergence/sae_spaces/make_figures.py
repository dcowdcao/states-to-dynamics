"""How far an illegal branch travels from its game's reference, in SAE latents.

M4 (sparsity of the divergence) over the eight seed-42 dictionaries. Latent
norms differ by ~100x across families and the SpaDE codes live on a simplex, so
nothing here is a bare distance: every quantity is a support overlap, an energy
share, or a ratio to that space's own control_legal branch.

  sae_feature_recurrence.png   M4, is it the same features on every event

  results/violation_divergence/sae_divergence_by_space.csv
  results/violation_divergence/sae_feature_recurrence.csv

Only the recurrence figure survives. The magnitude panels it shipped with — δ̂
across all nine spaces, the Jaccard/top-10/PR triptych, and the space x category
plateau matrix — all said the same thing, that every space lands in a narrow
0.91–1.03 band around the legal control, and thirty panels is an expensive way
to show a null. That band is a sentence in the README; the CSVs still carry
every number. What is worth a figure is the one place the dictionaries do
separate from the residual stream: a handful of features that fire on a
violation and almost never on either legal branch.

Reads $SPADE_DIVERGENCE_DIR[/rollout]/divergence/*.npz written by
src/spade/violation_divergence/compute_divergence.py — no re-encoding.

`--variant rollout` reads the tree where every branch, the reference included,
samples its own continuation, and writes `*_rollout` names beside the spliced
ones. That tree holds K rollouts per source branch as contiguous rows, so every
per-branch score the figure draws is averaged over them first (`per_branch`);
otherwise a branch is one item K times in the bootstrap and the interval is ~2x
too tight. The CSVs stay row-level and say so in `n_rows`.

Usage: python analysis/violation_divergence/sae_spaces/make_figures.py
       [--variant rollout]
"""

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spade.common.paths import results_dir
from spade.common.study import variant_dir
from spade.figures.paper_style import (
    CATEGORY_STYLE, MUTED, NULL, OKABE_ITO, apply_style,
    rule_key, sae_label, save_figure)
from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence.rollout_branches import K_ROLLOUTS

FIG_DIR = Path(__file__).resolve().parent / "figures"
CSV_DIR = results_dir("violation_divergence")
VARIANTS = ("spliced", "rollout")

RAW = "resid_post_5"                 # the dense space the dictionaries encode
ILLEGAL = CATEGORIES[1:]             # the six violation categories, taxonomy order
SERIES = ILLEGAL + ["control_legal"]
EMPTY_SQUARE = ILLEGAL[2:]           # the four violations that place on an empty square
POOLED = "illegal_pooled"
DICT_WIDTH = 8192                    # every seed-42 dictionary; config.json dict_size
TOPK = 10                            # topk_idx width, and k in topk_share
BOOT, SEED = 1000, 0

# Local stand-in for a paper_style entry (requested in RESULTS.md): hue =
# architecture family, marker + line style = the variant inside it, so eight
# dictionaries stay separable without color and without spending eight hues.
DASH = (0, (4, 1.5))
SPACE_STYLE = {
    "sae_StandardTrainer-othello":    {"color": OKABE_ITO[0], "marker": "o", "ls": "-"},
    "sae_PAnnealTrainer-othello":     {"color": OKABE_ITO[0], "marker": "s", "ls": DASH},
    "sae_GatedSAETrainer-othello":    {"color": OKABE_ITO[1], "marker": "o", "ls": "-"},
    "sae_GatedAnnealTrainer-othello": {"color": OKABE_ITO[1], "marker": "s", "ls": DASH},
    "sae_TopKTrainer-othello":        {"color": OKABE_ITO[2], "marker": "o", "ls": "-"},
    "sae_BatchTopKTrainer-othello":   {"color": OKABE_ITO[2], "marker": "s", "ls": DASH},
    "sae_SpaDE-lam1e-03-gam1e-04":    {"color": OKABE_ITO[3], "marker": "o", "ls": "-"},
    "sae_CosineSpaDETrainer-othello": {"color": OKABE_ITO[3], "marker": "s", "ls": DASH},
}
SPACES = list(SPACE_STYLE)


def label(space: str) -> str:
    return sae_label(space.removeprefix("sae_").removesuffix("Trainer-othello"))


@lru_cache
def load(space: str, variant: str):
    """Every array of one space, materialized once — the figures read the same
    npz a few dozen times each."""
    tree = variant_dir("divergence", variant="" if variant == "spliced" else variant)
    return dict(np.load(tree / "divergence" / f"{space}.npz"))


def named(name: str, variant: str) -> str:
    """Today's names stay with the spliced tree; a variant gets its own."""
    return name if variant == "spliced" else f"{name}_{variant}"


def per_branch(values: np.ndarray, k: int) -> np.ndarray:
    """Average a per-row score over the K rollouts of its source branch.

    The rollout tree stores them as contiguous rows (`row = source_row * K + k`)
    and every mask here selects on columns constant within a branch, so a reshape
    is the grouping. k = 1 in the spliced tree, where this is the identity.
    """
    return values if k == 1 else values.reshape(-1, k, *values.shape[1:]).mean(1)


def masks(meta) -> dict[str, np.ndarray]:
    """Branch rows per condition. Categories are unequally available, so every
    n differs and nothing is pooled without saying so."""
    m = {c: (meta["category"] == i + 1) & (meta["is_control"] == 0)
         for i, c in enumerate(ILLEGAL)}
    m["control_legal"] = meta["is_control"] == 1
    m[POOLED] = meta["is_control"] == 0
    return m


def style_of(name: str) -> dict:
    return NULL if name == "control_legal" else CATEGORY_STYLE[name]


# --- statistics ------------------------------------------------------------
# One branch row per category per game, so a per-category CI over rows is a CI
# over source games. The pooled-illegal conditions put six rows in one game;
# by_game collapses them first, otherwise n is inflated 6x.

def by_game(values: np.ndarray, gid: np.ndarray) -> np.ndarray:
    """Per-game mean over the rows given, in ascending game_id order."""
    order = np.argsort(gid, kind="stable")
    v, g = values[order], gid[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ok = ~np.isnan(v)
    total = np.add.reduceat(np.where(ok, v, 0.0), starts, axis=0)
    count = np.add.reduceat(ok, starts, axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def mean_ci(per_game: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and 95% CI half-width over source games."""
    n = (~np.isnan(per_game)).sum(0)
    return (np.nanmean(per_game, 0),
            1.96 * np.nanstd(per_game, 0) / np.sqrt(np.maximum(n, 1)))


def boot_ci(per_game: np.ndarray, statistic=np.nanmean) -> tuple[float, float]:
    """95% percentile bootstrap over source games."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per_game), (BOOT, len(per_game)))
    return tuple(np.percentile(statistic(per_game[idx], axis=1), [2.5, 97.5]))


def boot_ratio_ci(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """95% CI of mean(num)/mean(den), resampling games paired so the control
    moves with the illegal branches of the same game."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(num), (BOOT, len(num)))
    ratio = np.nanmean(num[idx], 1) / np.nanmean(den[idx], 1)
    return tuple(np.percentile(ratio, [2.5, 97.5]))


# --- M4 recurrence: are the top features the same across events? ------------

def recurrence(idx: np.ndarray) -> np.ndarray:
    """Fraction of events whose top-10 |Δ| at r = 0 contains each feature."""
    return np.bincount(idx.ravel(), minlength=DICT_WIDTH) / len(idx)


def split_half(idx: np.ndarray, gid: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-branch overlap with the 10 features most recurrent on the other games.

    Fitting and scoring on the same events would make any ranking look
    predictive; splitting on game parity is what turns "these ten indices came
    up often" into "the next event's top-10 is predictable". Returns the
    per-branch scores of each half, so the CI is over held-out games.
    """
    half = gid % 2 == 0
    out = []
    for fit, score in [(half, ~half), (~half, half)]:
        top = np.argsort(-recurrence(idx[fit]))[:TOPK]
        out.append(per_branch(np.isin(idx[score], top).sum(1) / TOPK, k))
    return tuple(out)


def feature_recurrence(meta, mask, variant: str, k: int) -> None:
    gid = meta["game_id"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    x = np.arange(len(SPACES))
    dodge = np.linspace(-0.3, 0.3, len(SERIES))
    empty_square_rates = []
    for i, cond in enumerate(SERIES):
        style, rows = style_of(cond), mask[cond]
        top_rate, top_ci, gen, gen_ci = [], [], [], []
        for space in SPACES:
            idx = load(space, variant)["topk_idx"][rows]
            rec = recurrence(idx)
            hit = per_branch(np.isin(idx, int(rec.argmax())).any(1).astype(float), k)
            top_rate.append(hit.mean())
            top_ci.append(boot_ci(hit))
            a, b = split_half(idx, gid[rows], k)
            scores = np.r_[a, b]
            gen.append(scores.mean())
            gen_ci.append(boot_ci(scores))
        if cond in EMPTY_SQUARE:
            empty_square_rates += top_rate
        for ax, mu, ci in [(axes[0], top_rate, top_ci), (axes[1], gen, gen_ci)]:
            mu, ci = np.array(mu), np.array(ci).T
            ax.errorbar(x + dodge[i], mu, yerr=np.abs(ci - mu), fmt=style["marker"],
                        ms=5, lw=0, elinewidth=1.0, capsize=2, color=style["color"],
                        markeredgecolor="white", markeredgewidth=0.5,
                        zorder=2 if cond == "control_legal" else 3,
                        label=style.get("label", "control_legal (null)"))
    for ax, chance, ylabel, title in [
            (axes[0], TOPK / DICT_WIDTH,
             "events holding that category's single\nmost recurrent feature in their "
             "top-10",
             "(a) one feature, how often it recurs"),
            (axes[1], TOPK ** 2 / DICT_WIDTH,
             "overlap of an event's top-10 with the\ntop-10 fitted on the other games",
             "(b) held-out generalization of the top-10")]:
        ax.axhline(chance, color=NULL["color"], lw=0.9, ls=NULL["ls"], zorder=1,
                   label="uniform chance over 8192 features")
        ax.text(len(SPACES) - 0.5, chance * 1.15, f"uniform chance = {chance:.4f}",
                fontsize=7, color=MUTED, ha="right", va="bottom")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([label(s) for s in SPACES], rotation=30, ha="right",
                           fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlim(-0.6, len(SPACES) - 0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    order = list(range(1, len(handles))) + [0]  # the chance null goes last
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("One feature carries the divergence for 43-99% of empty-square "
                 "violations in every dictionary, and for almost no legal move"
                 if variant == "spliced" else
                 f"The same features still carry it when the suffix is sampled: "
                 f"{min(empty_square_rates):.0%}-{max(empty_square_rates):.0%} of "
                 "empty-square violations",
                 fontsize=11, y=1.02)
    fig.text(0.5, 0.955,
             "Top-$|\\Delta|$ features at $r = 0$. Error bars are 95% percentile "
             "bootstrap over source games (1000 resamples); n per category is "
             "1636-2000 games, 2000 for the control."
             if variant == "spliced" else
             "Top-$|\\Delta|$ features at $r = 0$. Error bars are 95% percentile "
             "bootstrap over source branches (1000 resamples),\neach branch's K="
             f"{K_ROLLOUTS} scores averaged first; n per category is 1630-2000 "
             "branches, 2000 for the control.",
             fontsize=8, color=MUTED, style="italic", ha="center", va="top")
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    rule_key(fig, ["R1", "R2", "R3"], y=-0.12)
    save_figure(fig, FIG_DIR / named("sae_feature_recurrence", variant))


# --- CSVs ------------------------------------------------------------------

DIVERGENCE_COLS = ["space", "condition", "n_rows", "n_games", "delta_r0",
                   "peak_per_branch", "peak_r_per_branch_mean", "peak_r_per_branch_median",
                   "frac_peak_at_r0", "peak_of_mean_curve", "peak_r_of_mean_curve",
                   "half_life_median", "half_life_censored", "plateau",
                   "plateau_ratio_control", "nan_frac_delta", "cos_control_r0",
                   "cos_control_r24", "jaccard_r0", "jaccard_min", "jaccard_min_r",
                   "topk10_share_r0", "pr_r0", "dl0_r0"]


def divergence_csv(meta, mask, variant: str) -> None:
    """Both peak conventions, labelled. `persistence()` takes each branch's own
    argmax and the mean of those lands at r ~ 4; the mean curve peaks at r = 0 in
    every space. The per-branch distribution is right-skewed (median 0), so its
    mean is the weaker summary — but it is the one the npz stores, so both ship.

    Row-level, and `n_rows` says so: in the rollout tree a row is one (branch,
    rollout) pair. The means are unaffected -- K is the same for every branch --
    but the medians and fractions here are over the pooled K samples, and nothing
    in this file carries an interval."""
    window, gid = meta["window"], meta["game_id"]
    i0 = int(np.flatnonzero(window == 0)[0])
    post = window >= 0
    with (CSV_DIR / f"{named('sae_divergence_by_space', variant)}.csv").open(
            "w", newline="") as fh:
        out = csv.DictWriter(fh, DIVERGENCE_COLS)
        out.writeheader()
        for space in [RAW] + SPACES:
            z = load(space, variant)
            control = np.nanmean(z["plateau"][mask["control_legal"]])
            for cond in ILLEGAL + ["control_legal", POOLED]:
                rows = mask[cond]
                mean_curve = np.nanmean(z["delta"][rows], 0)
                at = int(np.nanargmax(np.where(post, mean_curve, np.nan)))
                row = {"space": space, "condition": cond, "n_rows": int(rows.sum()),
                       "n_games": len(np.unique(gid[rows])),
                       "delta_r0": mean_curve[i0],
                       "peak_per_branch": np.nanmean(z["peak"][rows]),
                       "peak_r_per_branch_mean": np.mean(z["peak_r"][rows]),
                       "peak_r_per_branch_median": np.median(z["peak_r"][rows]),
                       "frac_peak_at_r0": (z["peak_r"][rows] == 0).mean(),
                       "peak_of_mean_curve": mean_curve[at],
                       "peak_r_of_mean_curve": int(window[at]),
                       "half_life_median": np.median(z["half_life"][rows]),
                       "half_life_censored": np.isinf(z["half_life"][rows]).mean(),
                       "plateau": np.nanmean(z["plateau"][rows]),
                       "plateau_ratio_control": np.nanmean(z["plateau"][rows]) / control,
                       "nan_frac_delta": np.isnan(z["delta"][rows]).mean(),
                       "cos_control_r0": np.nanmean(z["cos_control"][rows, i0]),
                       "cos_control_r24": np.nanmean(z["cos_control"][rows, -1])}
                if "jaccard" in z:
                    j = np.nanmean(by_game(z["jaccard"][rows], gid[rows]), 0)
                    row |= {"jaccard_r0": j[i0], "jaccard_min": np.nanmin(j[post]),
                            "jaccard_min_r": int(window[post][np.nanargmin(j[post])]),
                            "dl0_r0": np.nanmean(z["dl0"][rows, i0])}
                row |= {"topk10_share_r0": np.nanmean(z["topk_share"][rows, i0]),
                        "pr_r0": np.nanmean(z["pr"][rows, i0])}
                out.writerow(row)


RECURRENCE_COLS = ["space", "condition", "n_rows", "top_feature", "top_recurrence",
                   "top_recurrence_in_control", "feature_2", "recurrence_2",
                   "feature_3", "recurrence_3", "splithalf_overlap",
                   "splithalf_chance", "n_features_half_mass"]


def recurrence_csv(meta, mask, variant: str, k: int) -> None:
    gid = meta["game_id"]
    with (CSV_DIR / f"{named('sae_feature_recurrence', variant)}.csv").open(
            "w", newline="") as fh:
        out = csv.DictWriter(fh, RECURRENCE_COLS)
        out.writeheader()
        for space in SPACES:
            idx_all = load(space, variant)["topk_idx"]
            in_control = idx_all[mask["control_legal"]]
            for cond in ILLEGAL + ["control_legal", POOLED]:
                rows = mask[cond]
                idx = idx_all[rows]
                rec = recurrence(idx)
                order = np.argsort(-rec)[:3]
                a, b = split_half(idx, gid[rows], k)
                cum = np.cumsum(np.sort(rec)[::-1]) / TOPK
                out.writerow({
                    "space": space, "condition": cond, "n_rows": int(rows.sum()),
                    "top_feature": int(order[0]), "top_recurrence": rec[order[0]],
                    "top_recurrence_in_control":
                        np.isin(in_control, int(order[0])).any(1).mean(),
                    "feature_2": int(order[1]), "recurrence_2": rec[order[1]],
                    "feature_3": int(order[2]), "recurrence_3": rec[order[2]],
                    "splithalf_overlap": np.r_[a, b].mean(),
                    "splithalf_chance": TOPK ** 2 / DICT_WIDTH,
                    "n_features_half_mass": int(np.searchsorted(cum, 0.5)) + 1})


def contract(meta, mask, variant: str) -> None:
    """The cross-validation keys of the design doc, printed so RESULTS.md quotes
    a run instead of a memory. A3 and A5 report the same names.

    `by_game` collapses each game's rows before the bootstrap, which absorbs the
    rollout tree's K rows per branch as well as its six illegal rows per game."""
    window, gid = meta["window"], meta["game_id"]
    i0 = int(np.flatnonzero(window == 0)[0])
    print(f"\nn_games {len(np.unique(gid))}  n_branches {len(meta['row'])} non-reference "
          + ("rows (15531 sequences)" if variant == "spliced"
             else f"rows = {len(meta['row']) // K_ROLLOUTS} branches x K={K_ROLLOUTS}"))
    pl = load(RAW, variant)["plateau"]
    ill, con = mask[POOLED], mask["control_legal"]
    lo, hi = boot_ratio_ci(by_game(pl[ill], gid[ill]), by_game(pl[con], gid[con]))
    print(f"ratio_illegal_control_L5 {np.nanmean(pl[ill]) / np.nanmean(pl[con]):.4f}"
          f"  [{lo:.4f}, {hi:.4f}] 95% CI over games")
    print(f"{'space':32}{'jaccard_min':>12}{'at r':>6}{'topk10_share':>14}{'pr(0)':>9}")
    for space in SPACES:
        z = load(space, variant)
        j = np.nanmean(by_game(z["jaccard"][ill], gid[ill]), 0)[window >= 0]
        print(f"{space:32}{np.nanmin(j):12.4f}{window[window >= 0][int(np.nanargmin(j))]:6d}"
              f"{np.nanmean(z['topk_share'][ill, i0]):14.4f}"
              f"{np.nanmean(z['pr'][ill, i0]):9.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="spliced", choices=VARIANTS,
                    help="rollout: every branch samples its own continuation "
                         "($DIR/rollout/), K per branch, averaged")
    variant = ap.parse_args().variant
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    meta = load(RAW, variant)
    for space in SPACES:
        assert np.array_equal(load(space, variant)["row"], meta["row"]), \
            f"{space}: row order"
    k = K_ROLLOUTS if variant == "rollout" else 1
    assert k == 1 or np.array_equal(meta["rollout"], meta["row"] % k), \
        "rows are not source_row * K + k"
    mask = masks(meta)
    rows = len(meta["row"])
    print(f"{len(np.unique(meta['game_id']))} games, {rows} branch rows"
          + (f" ({rows // k} branches x K={k})" if k > 1 else ""))
    for cond in ILLEGAL + ["control_legal", POOLED]:
        print(f"  {cond:18} {mask[cond].sum():6d}")

    feature_recurrence(meta, mask, variant, k)
    divergence_csv(meta, mask, variant)
    recurrence_csv(meta, mask, variant, k)
    contract(meta, mask, variant)
    print(f"\n{len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}")
    print(f"2 csvs in {CSV_DIR}")


if __name__ == "__main__":
    main()
