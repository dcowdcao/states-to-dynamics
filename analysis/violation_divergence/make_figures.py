"""Raw-space results for the violation-divergence study (M1, M2, M3).

Reads `$SPADE_DIVERGENCE_DIR[/rollout]/divergence/{token_embed,resid_post_0..7}.npz`
and writes the summary CSVs and figures F1 and F3. One entry point, because every
number here is a groupby-mean over the same nine arrays that the figures draw.

    results/violation_divergence/depth_sweep.csv         space x role, M1/M3
    results/violation_divergence/divergence_curves.csv   role x r, layer 5 + W_E
    results/violation_divergence/contract_keys.csv       the cross-validation keys

    figures/divergence_layer5.{png,pdf}   F1  event-locked delta-hat, raw and vs control
    figures/depth_sweep.{png,pdf}         F3  peak/plateau/offset/censoring vs depth

Two peak conventions are reported side by side and must not be confused. The
per-branch peak (`persistence`'s `peak`/`peak_r`, the contract's definition)
takes each branch's own argmax over r >= 0; the category-mean curve's maximum
takes the argmax after averaging. They disagree: the mean curve is maximal at
r = 0 in every space and every category, while the per-branch offset averages
5.9 at layer 5, because an argmax over 25 offsets of a noisy row is biased late
and upward. Both columns are in depth_sweep.csv.

`--variant rollout` reads the tree where every branch, the reference included,
samples its own continuation from OthelloGPT instead of replaying the source
game's, and writes `*_rollout` names beside the spliced ones. That tree holds
K = 4 rollouts per source branch; `reduce_k` averages them, so a branch is one
item in every n and every interval. The rollout tree carries its own caveat:
with a sampled suffix both classes drift for reasons the branch move did not
cause, and the plateau saturates near the unrelated-activation ceiling, where a
ratio of 1 means nothing.

Usage: python analysis/violation_divergence/make_figures.py [--variant rollout]
                                                            [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spade.common.paths import results_dir
from spade.common.study import variant_dir
from spade.figures.paper_style import (
    CATEGORY_STYLE, GRID, INK, MUTED, NULL, apply_style, save_figure)
from spade.othello.taxonomy import CATEGORIES

HERE = Path(__file__).resolve().parent
SPACES = ["token_embed"] + [f"resid_post_{i}" for i in range(8)]
DEPTH_LABELS = ["$W_E$"] + [f"L{i}" for i in range(8)]
# W_E sits at its own tick with a gap before the residual stream (§4.1): the
# input space is not layer -1, and the gap is what the "computed, not
# inherited" claim is read off.
DEPTH_X = np.array([0.0] + [1.5 + i for i in range(8)])
DEPTH_RULE_X = 0.75
HEADLINE = "resid_post_5"
CONTROL = "control_legal"
ILLEGAL = CATEGORIES[1:]
ROLES = [CONTROL] + ILLEGAL
POOLED = "illegal_pooled"
REPORTED = ROLES + [POOLED]
BOOT, BOOT_SEED = 1000, 0

VARIANTS = ("spliced", "rollout")
# Columns that name the branch rather than measure it, so they are read off one
# rollout instead of averaged over K.
PER_BRANCH = ("game_id", "t0", "category", "is_control", "n_flipped", "role")
# delta-hat between two *different* games' layer-5 rollouts at the same absolute
# position, mean over r in [20, 24], n = 2000 games at seed 0. The rollout curves
# are read against this: it is what "these activations are unrelated" measures.
UNRELATED_CEILING = 1.312
SATURATED_R = 8  # from here on the rollout illegal/control ratio is within 0.5% of 1


def tree(variant: str) -> Path:
    return variant_dir("divergence", variant="" if variant == "spliced" else variant)


def named(name: str, variant: str) -> str:
    """Today's names stay with the spliced tree; a variant gets its own."""
    return name if variant == "spliced" else f"{name}_{variant}"


def load(space: str, variant: str) -> dict:
    """The npz plus a `role` column that folds is_control into the category."""
    z = dict(np.load(tree(variant) / "divergence" / f"{space}.npz"))
    role = np.array(CATEGORIES, dtype=object)[z["category"]]
    role[z["is_control"] == 1] = CONTROL
    z["role"] = role
    z["censored"] = np.isinf(z["half_life"])
    return reduce_k(z) if "rollout" in z else z


def mean_k(g: np.ndarray, empty: float = np.nan) -> np.ndarray:
    """Mean over axis 1 ignoring non-finite entries. Not `np.nanmean`: cos_t0 is
    undefined for every rollout at r < 0, and an all-NaN slice is expected there."""
    ok = np.isfinite(g)
    n = ok.sum(1)
    return np.where(n > 0, np.where(ok, g, 0).sum(1) / np.maximum(n, 1), empty)


def reduce_k(z: dict) -> dict:
    """One row per source branch, averaging its K rollouts.

    The rollout tree stores K samples of each branch's continuation as contiguous
    rows (`row = source_row * K + k`), so without this a branch is one item K times
    in every n and every interval. `half_life` averages over the rollouts that did
    halve and stays +inf only when none did; `censored` carries the share that did
    not, which is what F3 draws.
    """
    k = int(z["rollout"].max()) + 1
    assert np.array_equal(z["rollout"], z["row"] % k), "rows are not source_row * K + k"
    out = {"window": z["window"], "row": z["row"][::k] // k}
    for key, v in z.items():
        if key in ("window", "row", "rollout"):
            continue
        g = v.reshape(-1, k, *v.shape[1:])
        if key in PER_BRANCH:
            assert (g == g[:, :1]).all(), f"{key} differs across a branch's rollouts"
            out[key] = g[:, 0]
        else:
            out[key] = mean_k(g, empty=np.inf if key == "half_life" else np.nan)
    return out


def rows_of(z: dict, role: str) -> np.ndarray:
    """Row mask for `role`; the pooled scope is every non-control row."""
    return z["role"] != CONTROL if role == POOLED else z["role"] == role


def mean_ci(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 95% CI over rows. One row per source game within a role, so the
    resampling unit is the game and no clustering correction is needed.

    Columns with nothing in them stay NaN rather than warning: `cos_t0` is
    undefined for r < 0, where Delta is exactly zero and has no direction.
    """
    n = (~np.isnan(x)).sum(0)
    mu, half = np.full(x.shape[1], np.nan), np.full(x.shape[1], np.nan)
    seen = n > 1
    mu[seen] = np.nanmean(x[:, seen], 0)
    half[seen] = 1.96 * np.nanstd(x[:, seen], 0, ddof=1) / np.sqrt(n[seen])
    return mu, mu - half, mu + half


def against_control(z: dict, role: str, key: str) -> np.ndarray:
    """`key` for `role`, minus the same game's control_legal row. Paired, so the
    game-to-game spread in how far any branch travels drops out."""
    is_control = z["role"] == CONTROL
    control = np.full(int(z["game_id"].max()) + 1, np.nan)
    control[z["game_id"][is_control]] = np.arange(len(is_control))[is_control]
    rows = z["role"] == role
    return z[key][rows] - z[key][control[z["game_id"][rows]].astype(int)]


def plateau_ratio_ci(z: dict) -> tuple[float, float, float]:
    """Pooled-illegal over control plateau, with a bootstrap over source games.

    Resampling games rather than rows is the point: a game contributes up to six
    illegal rows that share its prefix, so a row bootstrap would understate the
    interval.
    """
    is_control = z["role"] == CONTROL
    ratio = z["plateau"][~is_control].mean() / z["plateau"][is_control].mean()
    # A resample's ratio is (sum of its illegal plateaus / count) over the same
    # for control, so per-game sums and counts are a sufficient summary.
    game, plateau = z["game_id"], z["plateau"]
    totals = [(np.bincount(game[m], plateau[m]), np.bincount(game[m]).astype(float))
              for m in (~is_control, is_control)]
    rng = np.random.default_rng(BOOT_SEED)
    pick = rng.integers(0, len(totals[0][0]), size=(BOOT, len(totals[0][0])))
    (si, ni), (sc, nc) = ((s[pick].sum(1), n[pick].sum(1)) for s, n in totals)
    lo, hi = np.percentile((si / ni) / (sc / nc), [2.5, 97.5])
    return ratio, lo, hi


def depth_table(variant: str) -> pd.DataFrame:
    """Per space and role, M1's level and M3's four persistence statistics.

    `excess_over_we` is each role's largest gap above the input floor over
    r >= 0. The floor is a delta function, so the maximiser is always r = 1 and
    the value is just the mean curve there; computing it as a max anyway keeps
    the column matching the contract's wording for `peak_layer`.
    """
    floor = load("token_embed", variant)
    floor_curve = {role: floor["delta"][rows_of(floor, role)].mean(0) for role in REPORTED}
    rows = []
    for space in SPACES:
        z = load(space, variant)
        w = z["window"]
        control_plateau = z["plateau"][z["role"] == CONTROL].mean()
        # token_embed's plateau is exactly 0 for every branch, so the ratio to
        # the control is 0/0 -- undefined, and the interesting fact itself.
        ratio_defined = control_plateau > 0
        for role in REPORTED:
            m = rows_of(z, role)
            curve = z["delta"][m].mean(0)
            post = np.where(w < 0, -np.inf, curve)
            half_life = z["half_life"][m]
            rows.append({
                "space": space, "role": role, "n": int(m.sum()),
                "delta_r0": curve[w == 0][0],
                "delta_peak_row": z["peak"][m].mean(),
                "delta_peak_row_ci": 1.96 * z["peak"][m].std(ddof=1) / np.sqrt(m.sum()),
                "peak_r_row": z["peak_r"][m].mean(),
                "peak_r_row_ci": 1.96 * z["peak_r"][m].std(ddof=1) / np.sqrt(m.sum()),
                "peak_r_row_median": np.median(z["peak_r"][m]),
                "delta_peak_curve": curve[post.argmax()],
                "peak_r_curve": int(w[post.argmax()]),
                "excess_over_we": np.where(w < 0, -np.inf,
                                           curve - floor_curve[role]).max(),
                "plateau": z["plateau"][m].mean(),
                "plateau_ci": 1.96 * z["plateau"][m].std(ddof=1) / np.sqrt(m.sum()),
                "plateau_ratio_to_control": (z["plateau"][m].mean() / control_plateau
                                             if ratio_defined else np.nan),
                "half_life_median": np.median(half_life),
                "half_life_censored": z["censored"][m].mean(),
            })
    return pd.DataFrame(rows)


def curve_table(variant: str) -> pd.DataFrame:
    """Event-locked mean and 95% CI per role, for the two spaces F1 draws."""
    rows = []
    for space in ["token_embed", HEADLINE]:
        z = load(space, variant)
        for role in ROLES:
            m = z["role"] == role
            out = {"space": space, "role": role, "r": z["window"], "n": int(m.sum())}
            for key in ["delta", "cos_t0", "cos_control"]:
                mu, lo, hi = mean_ci(z[key][m])
                out |= {key: mu, f"{key}_lo": lo, f"{key}_hi": hi}
            if role != CONTROL:
                mu, lo, hi = mean_ci(against_control(z, role, "delta"))
                out |= {"delta_vs_control": mu, "delta_vs_control_lo": lo,
                        "delta_vs_control_hi": hi}
            rows.append(pd.DataFrame(out))
    return pd.concat(rows, ignore_index=True)


def contract_table(depth: pd.DataFrame, variant: str) -> pd.DataFrame:
    """The design doc's cross-validation keys, one row each, for a mechanical diff."""
    z = load(HEADLINE, variant)
    at = depth.set_index(["space", "role"])
    we = depth[depth.space == "token_embed"].set_index("role")
    ratio, lo, hi = plateau_ratio_ci(z)
    pooled = depth[(depth.role == POOLED) & (depth.space != "token_embed")]
    peak_layer = SPACES.index(pooled.loc[pooled.excess_over_we.idxmax(), "space"]) - 1

    # The npz drops the 2000 reference rows, whose divergence against themselves
    # is zero; n_branches is the branch set DATA_REPORT counts, so add them back.
    n_games = len(np.unique(z["game_id"]))
    per_role = {"delta_peak_L5": "delta_peak_row", "delta_peak_r_L5": "peak_r_row",
                "delta_plateau_L5": "plateau"}
    rows = [{"key": "n_games", "scope": "-", "value": n_games},
            {"key": "n_branches", "scope": "-", "value": len(z["game_id"]) + n_games}]
    rows += [{"key": key, "scope": role, "value": at.loc[(HEADLINE, role), column]}
             for role in REPORTED for key, column in per_role.items()]
    rows += [{"key": "delta_peak_WE", "scope": role, "value": we.loc[role, "delta_peak_row"]}
             for role in REPORTED]
    rows += [
        {"key": "ratio_illegal_control_L5", "scope": "-", "value": ratio},
        {"key": "ratio_illegal_control_L5_lo", "scope": "-", "value": lo},
        {"key": "ratio_illegal_control_L5_hi", "scope": "-", "value": hi},
        {"key": "peak_layer", "scope": "-", "value": peak_layer},
    ]
    return pd.DataFrame(rows)


def style_of(role: str) -> dict:
    """CATEGORY_STYLE for a violation; NULL for the control, which is the null."""
    if role != CONTROL:
        return CATEGORY_STYLE[role]
    return {"color": NULL["color"], "marker": NULL["marker"], "label": "legal control (null)"}


def series(ax, x, mu, role: str, n: int, offset: int, every: int = 4) -> None:
    """The mean curve alone. Seven of these per axes, and seven shaded CIs on top
    of them hid the curves they annotated; F3 draws its interval as error bars,
    which seven series can carry, and F1's are in divergence_curves.csv."""
    style = style_of(role)
    null = role == CONTROL
    ax.plot(x, mu, color=style["color"], marker=style["marker"], ms=4,
            lw=NULL["lw"] if null else 1.7, ls=NULL["ls"] if null else "-",
            markevery=(offset % every, every), markeredgecolor="white",
            markeredgewidth=0.5, zorder=NULL["zorder"] if null else 4,
            label=f"{style['label']} (n={n})")


def event_axes(ax, w: np.ndarray) -> None:
    ax.axvspan(w[0] - 0.5, -0.5, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.set_xlabel("position relative to the branch move ($r = s - t_0$)")
    ax.set_xticks(range(int(w[0]), int(w[-1]) + 1, 4))


def legend_order(ax, **kw):
    """Categories in taxonomy order, the null last (§3)."""
    handles, labels = ax.get_legend_handles_labels()
    order = list(range(1, len(handles))) + [0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], **kw)


def saturated_band(ax, w: np.ndarray) -> None:
    """Shade the offsets where the rollout curves stop carrying a comparison.

    Both classes climb toward the divergence of unrelated activations, so from
    r = 8 the illegal/control ratio is within 0.5% of 1 whatever legality does.
    The band says which part of the x-axis is not evidence.
    """
    ax.axvspan(SATURATED_R, w[-1] + 0.5, color=GRID, alpha=0.4, lw=0, zorder=0)


F1_TITLE = {
    "spliced": "An illegal move displaces layer 5 no further than a different legal "
               "move does",
    "rollout": "Once every branch samples its own future, illegal and legal displace "
               "layer 5 alike",
}

def f1_divergence_layer5(variant: str, camera_ready: bool) -> None:
    z = load(HEADLINE, variant)
    w = z["window"]
    # Two panels, not four. The magnitude story is the result; the two cosine
    # panels it shipped with said "no illegal-specific direction", which is a
    # null and now a sentence in the README. cos_t0 and cos_control stay in
    # divergence_curves.csv.
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharex=True)
    axes = [[axes[0], axes[1]]]

    for i, role in enumerate(ROLES):
        m = z["role"] == role
        n = int(m.sum())
        series(axes[0][0], w, mean_ci(z["delta"][m])[0], role, n, i)
        if role != CONTROL:
            series(axes[0][1], w, mean_ci(against_control(z, role, "delta"))[0], role, n, i)

    for ax in axes[0]:
        event_axes(ax, w)
    rollout = variant == "rollout"
    axes[0][0].set_ylabel(r"$\hat\delta(r) = \|\Delta\| \,/\, \|z^{\mathrm{ref}}\|$")
    axes[0][0].set_title("(a) divergence from the reference "
                         + ("branch's own rollout" if rollout else "game"), loc="left")
    top = ILLEGAL[np.argmax([z["delta"][z["role"] == c][:, w == 0].mean() for c in ILLEGAL])]
    peak = z["delta"][z["role"] == top][:, w == 0].mean()
    axes[0][0].annotate(f"{peak:.3f} at $r$=0 ({CATEGORY_STYLE[top]['label']})",
                        xy=(0.3, peak), xytext=(4.0, peak + 0.02),
                        fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    if rollout:
        saturated_band(axes[0][0], w)
        axes[0][0].axhline(UNRELATED_CEILING, color=NULL["color"], lw=0.9,
                           ls=(0, (1, 2)), zorder=1,
                           label="unrelated-activation ceiling")
    axes[0][0].set_ylim(-0.03, 1.45 if rollout else 0.95)
    legend_order(axes[0][0], fontsize=7, loc="lower right", ncol=2, columnspacing=0.8)

    axes[0][1].axhline(0, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axes[0][1].set_ylabel(r"$\hat\delta_{\mathrm{cat}} - \hat\delta_{\mathrm{control}}$"
                          "\n(paired, same source game)")
    axes[0][1].set_title("(b) excess over a different legal move", loc="left")
    if rollout:
        saturated_band(axes[0][1], w)
    axes[0][1].set_ylim(-0.155, 0.115)

    if not camera_ready:
        fig.suptitle(F1_TITLE[variant], fontsize=11, y=1.0)
    # 0.95 is where the removed estimator note ended; the band above it is the suptitle's.
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, HERE / "figures" / named("divergence_layer5", variant), camera_ready)


# The two per-branch-peak panels this figure used to carry are gone. `peak` and
# `peak_r` average a per-row argmax over 33 offsets, which estimates E[max] not
# max E: an offset permutation within rows reproduces `peak` to 0.4% at every
# layer and 79-83% of `peak_r`'s depth drift (RED_TEAM.md F2). Both remain in
# depth_sweep.csv for anyone who wants them next to that null. What is left are
# two means, neither selection-biased.
DEPTH_PANELS = [
    ("plateau", "plateau_ci", r"plateau $\hat\delta$, $r \in [20, 24]$",
     "(a) how much is still there 20+ plies later"),
    ("half_life_censored", None, r"branches with no half-decay in 24 plies",
     "(b) how often the gap never halves"),
]


F3_TITLE = {
    "spliced": "The input difference is a delta function; every layer converts more of "
               "it into a lasting offset",
    "rollout": "A sampled suffix moves the token embedding too, so no layer is reading "
               "against a silent input",
}

def f3_depth_sweep(depth: pd.DataFrame, variant: str, camera_ready: bool) -> None:
    at = depth.set_index(["space", "role"])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1), sharex=True)

    for ax, (key, err, ylabel, title) in zip(axes, DEPTH_PANELS):
        ax.axvline(DEPTH_RULE_X, color=MUTED, lw=0.8, zorder=1)
        for i, role in enumerate(ROLES):
            y = np.array([at.loc[(s, role), key] for s in SPACES])
            n = at.loc[(HEADLINE, role), "n"]
            if err is None:  # a proportion: normal-approximation interval
                half = 1.96 * np.sqrt(y * (1 - y) / n)
            else:
                half = np.array([at.loc[(s, role), err] for s in SPACES])
            series(ax, DEPTH_X, y, role, int(n), i, every=1)
            ax.errorbar(DEPTH_X, y, yerr=half, fmt="none", elinewidth=0.9, capsize=2,
                        ecolor=style_of(role)["color"], zorder=2)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.set_xticks(DEPTH_X, DEPTH_LABELS)
        ax.grid(axis="x", visible=False)
    for ax in axes:
        ax.set_xlabel("representation space (residual stream, 0 = first block)")

    spliced = variant == "spliced"
    floor = at.loc[("token_embed", POOLED), "plateau"]
    axes[0].annotate("exactly 0: the token embedding\ncarries nothing past $r=0$" if spliced
                     else f"{floor:.2f}, not 0: the branches'\nsuffixes are different "
                          "tokens here",
                     xy=(DEPTH_X[0], floor),
                     xytext=(DEPTH_X[2], floor + 0.13) if spliced
                     else (DEPTH_X[5], floor - 0.085),
                     fontsize=8, color=INK,
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    axes[1].set_ylim(-0.03, 0.82 if spliced else 1.09)
    legend_order(axes[1], fontsize=7, loc="upper left" if spliced else "lower right",
                 ncol=2, columnspacing=0.8)

    if not camera_ready:
        fig.suptitle(F3_TITLE[variant], fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, HERE / "figures" / named("depth_sweep", variant), camera_ready)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="spliced", choices=VARIANTS,
                    help="rollout: every branch samples its own continuation "
                         "($DIR/rollout/), K per branch, averaged")
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)
    variant = args.variant

    out = results_dir("violation_divergence")
    out.mkdir(parents=True, exist_ok=True)
    depth = depth_table(variant)
    depth.to_csv(out / f"{named('depth_sweep', variant)}.csv", index=False,
                 float_format="%.6g")
    curve_table(variant).to_csv(out / f"{named('divergence_curves', variant)}.csv",
                                index=False, float_format="%.6g")
    contract = contract_table(depth, variant)
    contract.to_csv(out / f"{named('contract_keys', variant)}.csv", index=False,
                    float_format="%.6g")

    f1_divergence_layer5(variant, args.camera_ready)
    f3_depth_sweep(depth, variant, args.camera_ready)
    ratio = contract.set_index("key")["value"]
    print(f"{variant}: ratio_illegal_control_L5 {ratio['ratio_illegal_control_L5']:.4f} "
          f"[{ratio['ratio_illegal_control_L5_lo']:.4f}, "
          f"{ratio['ratio_illegal_control_L5_hi']:.4f}]")
    print(f"csvs in {out}, figures in {HERE / 'figures'}")


if __name__ == "__main__":
    main()
