"""How long a step the branch takes, against the step the played move takes.

The fork family draws where a branch goes; this draws how fast it gets there.
For each branch b of a source game, at the study's sampled positions,

    s_b(t) = ||z_b(t) - z_b(t - STRIDE)||

and the panels draw the paired ratio s_b(t) / s_ref(t) -- the branch's step
length against the reference branch's at the same position in the same game.
The nine spaces differ ~580x in absolute step length (raw layer 5 steps 18.6,
SpaDE 0.032), so a raw step length is not a quantity nine cells can share; the
within-game ratio is, the reference is then exactly 1, and a second legal move's
own ratio is the empirical null. The absolute lengths are in the CSV and the raw
per-space scale is in README.md, so nothing the division removes is lost.

**The sampling convention is load-bearing.** Positions come from
`fork_plane.positions` at STRIDE = 2, the fork family's convention, so one step
spans four plies inside one mover's frame. `../rule_curvature` measures per ply
instead, and the two are different quantities, not two estimates of one: on the
same layer-5 corpus the legal class steps 29.2 per ply against 18.9 at stride 2
(`../theta_comparability`), and the class contrasts change sign between them.
The two conventions must never share an axis.

Before t0 every branch is token-identical, so the ratio is exactly 1 there.
`step_ratio` asserts it rather than assuming it, and the panels draw those
positions -- shaded, with the legend saying why they are flat -- instead of
starting at the fork.

**Two figures, one quantity and one panel grid.** Without `--n_games` this is one
game (n = 1): an illustration of the construction, not a claim, with the three
rule curves drawn over their member categories and the within/between-rule spread
on the canvas. With `--n_games` it is `measure.py`'s population -- the same ratio
averaged over source games, with a percentile bootstrap CI over games, both legal
controls drawn as the two draws of one null that they are, and no member traces,
because a member there is itself a mean over games and would read as
game-to-game spread. The reduction and the interval are the whole difference;
the quantity, the panel and the grid are shared.

Usage: python analysis/violation_divergence/step_length_fork/make_figures.py
       [--game 1235 | --n_games 100] [--camera-ready]
"""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.figures.branch_panels import (
    ratio_series, rule_ratio_panel, rule_ratio_population_panel)
from spade.figures.branch_style import below, ordered_legend, rule_member_key
from spade.figures.paper_style import MUTED, apply_style, save_figure
from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence.branches import (
    FIGURE_GAME, Game, complete_games, load_game, reference_row, role)
from spade.violation_divergence.fork_plane import (
    RAW_SPACE, STRIDE, positions, space_features, spaces)
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

FIG_DIR = Path(__file__).resolve().parent / "figures"
RESULTS = (Path(__file__).resolve().parents[3] / "results" / "violation_divergence")
CSV = RESULTS / "step_length_fork.csv"
NCOL = 3
CELL_W, CELL_H = 3.2, 2.6
BAND_H = 1.1          # the legend and the notes under the grid
TICK_EVERY = 6        # x ticks in plies; a multiple of STRIDE through t0
BOOT, BOOT_SEED = 2000, 0
SELECTION_NOTE = (f"game {FIGURE_GAME} is selected, not typical: the argmax over 1550 "
                  f"branch-complete games of its illegal branches' mutual separation "
                  f"(README)")
SAMPLING_NOTE = (f"one step spans {2 * STRIDE} plies: positions are sampled every "
                 f"{STRIDE} plies at $t_0$'s parity (fork_plane.positions), the fork "
                 f"family's convention. ../rule_curvature samples every ply instead,\n"
                 f"where the same layer-5 legal corpus steps 29.2 against 18.9 here "
                 f"(../theta_comparability) — the two conventions measure different "
                 f"quantities and must never share an axis.")
NULL_NOTE = ("the two grey curves are two draws of one null — a different legal token at "
             "$t_0$, measured exactly as the illegal branches are. Neither is a divisor "
             "here (the reference is), so the gap\nbetween them is the size of a "
             "difference that means nothing, and each panel's number is that rule minus "
             "control_legal on the same games, not two point estimates side by side.")


def step_ratio(F: np.ndarray, pos: np.ndarray, t0: int,
               ref: int) -> tuple[np.ndarray, np.ndarray]:
    """([n, T-1], [n, T-1]): each branch's step length over the sampled positions,
    and that length divided by the reference branch's at the same position.

    A step is attributed to the position it ends on, so it is branch-free exactly
    when that position precedes t0 -- and there the two rows are bitwise equal,
    which is what the assert checks.
    """
    s = np.linalg.norm(F[:, 1:] - F[:, :-1], axis=-1)
    ratio = s / s[ref]
    assert (ratio[:, pos[1:] < t0] == 1).all(), \
        "branches do not step identically over the shared prefix"
    return s, ratio


def series(rows: list[dict]) -> list[tuple[str, str, list[int]]]:
    """(kind, name, member row indices) per tabulated series, in legend order.

    Every legal control the caller kept, not the first one: the population figure
    draws both, and one game's figure drops the second before it gets here."""
    return ([("reference", "reference", [reference_row(rows)])]
            + [("rule", rule, members) for rule, members in rule_rows(rows).items()]
            + [("category", CATEGORIES[rows[i]["category"]], [i])
               for members in rule_rows(rows).values() for i in members]
            + [("control", role(r["category"], r["is_control"]), [i])
               for i, r in enumerate(rows) if r["is_control"]])


def table(g: Game, pos: np.ndarray, measured: list[tuple[str, np.ndarray, np.ndarray]]
          ) -> list[dict]:
    """One row per (space, series, position): what the panels draw, plus the
    absolute step length the ratio divides away."""
    return [{"game": g.game, "t0": g.t0, "space": space, "kind": kind,
             "series": name, "n_members": len(members),
             "t": int(t), "r": int(t) - g.t0,
             "step_length": f"{s[members, j].mean():.6g}",
             "ratio": f"{ratio[members, j].mean():.6g}"}
            for space, s, ratio in measured
            for kind, name, members in series(g.rows)
            for j, t in enumerate(pos[1:])]


def annotate_peak(ax, rows: list[dict], pos: np.ndarray, ratio: np.ndarray,
                  t0: int) -> None:
    """The panel's largest rule-mean excursion, as text: a nine-cell grid has no
    room for an arrow per curve, and the value is what a reader wants off it."""
    means = {rule: ratio[members].mean(axis=0)
             for rule, members in rule_rows(rows).items()}
    rule, curve = max(means.items(), key=lambda kv: np.abs(kv[1] - 1).max())
    j = int(np.abs(curve - 1).argmax())
    ax.text(0.97, 0.93, f"{rule.split()[0]} {curve[j]:.2f} at $t_0${pos[j] - t0:+d}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)


def spread_note(spread: float) -> str:
    return (f"within ÷ between-rule spread of the drawn ratio over $t \\geq t_0$, in "
            f"{RAW_SPACE.split(' (')[0]}, one game: {spread:.3f}. Near 1 the three "
            f"curves are a label rather than a finding,\nand the faint member traces "
            f"carry the evidence; the nine spaces' values are in README.md.")


def panel_grid(fig, cell, names: list[str], pos: np.ndarray, t0: int, draw) -> list:
    """The nine-cell grid both figures draw in: one ratio panel per space on
    shared axes, x ticks every TICK_EVERY plies through t0, and the axis labels
    on the outer cells only. `draw(ax, j)` fills cell j."""
    nrow = -(-len(names) // NCOL)
    grid = cell.subgridspec(nrow, NCOL, wspace=0.10, hspace=0.40)
    axes = []
    for j, name in enumerate(names):
        ax = fig.add_subplot(grid[j // NCOL, j % NCOL],
                             sharex=axes[0] if axes else None,
                             sharey=axes[0] if axes else None)
        axes.append(ax)
        draw(ax, j)
        ax.set_title(name, loc="left", fontsize=8.5)
        ax.set_xticks([t for t in pos if (t - t0) % TICK_EVERY == 0])
        ax.tick_params(labelsize=8, labelbottom=j >= len(names) - NCOL,
                       labelleft=j % NCOL == 0)
    for ax in axes[-NCOL:]:
        ax.set_xlabel(f"move index $t$ (branch at $t_0$ = {t0})", fontsize=9)
    for ax in axes[::NCOL]:
        ax.set_ylabel("step length $\\div$ the reference\nbranch's, "
                      "$s_b(t) / s_{ref}(t)$", fontsize=9)
    return axes


def figure(g: Game, pos: np.ndarray,
           measured: list[tuple[str, np.ndarray, np.ndarray]], spread: float,
           camera_ready: bool) -> Path:
    nrow = -(-len(measured) // NCOL)
    fig = plt.figure(figsize=(CELL_W * NCOL, CELL_H * nrow + BAND_H))
    gs = fig.add_gridspec(2, 1, height_ratios=[CELL_H * nrow, BAND_H],
                          left=0.07, right=0.98, top=0.90, bottom=0.03, hspace=0.16)

    head = fig.add_subplot(gs[0])
    head.axis("off")
    head.patch.set_visible(False)
    # The estimator and n ride above the grid, not in the suptitle: camera-ready
    # drops the suptitle and both are what keeps the curves readable (§1, §5).
    head.set_title("step length at the fork, as a multiple of the reference branch's at "
                   "the same position — one curve per\nconstraint rule over its member "
                   "categories, drawn faint; $n$ = 1 game, so no interval over games",
                   loc="left", fontsize=10.5, pad=16)

    def draw(ax, j):
        rule_ratio_panel(ax, measured[j][2], g.rows, pos[1:], g.t0, legend=(j == 0))
        annotate_peak(ax, g.rows, pos[1:], measured[j][2], g.t0)

    axes = panel_grid(fig, gs[0], [space for space, _, _ in measured], pos[1:], g.t0, draw)

    band = fig.add_subplot(gs[1])
    band.axis("off")
    leg = ordered_legend(band, axes[0], 3, loc="upper left", bbox_to_anchor=(0.0, 1.05))
    # Each note hangs off the bottom of the one above it: how many lines the legend
    # and each note wrapped to is what moves everything under them.
    y = below(fig, leg, 0.012)
    y = below(fig, fig.text(0.005, y, SAMPLING_NOTE, ha="left", va="top", fontsize=7.5,
                            color=MUTED, linespacing=1.5), 0.006)
    y = below(fig, fig.text(0.005, y, spread_note(spread), ha="left", va="top",
                            fontsize=7.5, color=MUTED, linespacing=1.5), 0.006)
    if g.game == FIGURE_GAME:
        y = below(fig, fig.text(0.005, y, SELECTION_NOTE, ha="left", va="top",
                                fontsize=7.5, color=MUTED), 0.006)
    rule_member_key(fig, x=0.005, y=y)

    if not camera_ready:
        fig.suptitle(f"Step Length at the Fork in Each Representation Space — "
                     f"game {g.game}, layer 5", fontsize=12)
    return save_figure(fig, FIG_DIR / f"game_{g.game}_step_length_fork", camera_ready)


def one_game(game: int, camera_ready: bool) -> None:
    device = select_device()
    base = study_dir("divergence")   # the spliced tree; the rollout set forks its suffix too
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)

    assert game in complete_games(meta), f"game {game} does not carry every branch type"
    g = load_game(meta, game)
    # One control, not two: control 2 is the null *for* control 1 and belongs where
    # control 1 is measured, not in an illustration of the construction.
    keep = [i for i, r in enumerate(g.rows) if r["is_control"] != 2]
    g = replace(g, idx=g.idx[keep], rows=[g.rows[i] for i in keep])
    pos = positions(g.t0)
    Z = acts[g.idx][:, pos, :].float().numpy()
    ref = reference_row(g.rows)

    post = pos[1:] >= g.t0
    measured, spread = [], {}
    for space, entry in spaces():
        s, ratio = step_ratio(space_features(Z, entry, device), pos, g.t0, ref)
        measured.append((space, s, ratio))
        spread[space] = spread_ratio(ratio[:, post], g.rows)

    # Printed, not read off the figure: README.md quotes these.
    drawn = [(name, m) for kind, name, m in series(g.rows) if kind in ("rule", "control")]
    print(f"game {g.game}, t0 {g.t0}, r = {pos[1] - g.t0} … {pos[-1] - g.t0} at stride "
          f"{STRIDE}; cells are the mean ratio over r >= 0 (the value at r = 0)")
    print(f"{'space':34} {'s_ref range':>17} {'spread':>7}  "
          + "  ".join(f"{name.split()[0]:>15}" for name, _ in drawn))
    for space, s, ratio in measured:
        print(f"{space:34} {s[ref].min():7.4g}–{s[ref].max():<9.4g} {spread[space]:7.3f}  "
              + "  ".join(f"{ratio[m][:, post].mean():.3f} "
                          f"({ratio[m][:, pos[1:] == g.t0].mean():.2f})".rjust(15)
                          for _, m in drawn))

    rows = table(g, pos, measured)
    with CSV.open("w", newline="") as fh:
        out = csv.DictWriter(fh, list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    out_fig = figure(g, pos, measured, spread[RAW_SPACE], camera_ready)
    print(f"wrote {out_fig}\nwrote {CSV}")


def rows_of(z) -> list[dict]:
    """The branch rows `series`, `ratio_series` and `rule_rows` read, out of the
    npz columns."""
    return [{"category": int(c), "is_control": int(i)}
            for c, i in zip(z["category"], z["is_control"])]


def boot_ci(x: np.ndarray, draws: np.ndarray):
    """(lo, hi) of the 95% percentile bootstrap CI of the mean of `x`
    [n_games, ...] over source games.

    `draws` is one index matrix shared by every series, every panel and every
    contrast, so the intervals are read off the same resamples of the same games.
    Games and not branches: a game's branches share a prefix and the reference
    they are divided by."""
    return tuple(np.percentile(x[draws].mean(axis=1), [2.5, 97.5], axis=0))


def drawn_series(z, si: int) -> np.ndarray:
    """[n_series, n_games, T] for one space: each drawn series' per-game ratio,
    a rule's taken over its member categories inside the game."""
    return np.stack([z["ratio"][si][:, m].mean(axis=1) for m in ratio_series(rows_of(z))])


def annotate_contrast(ax, x: np.ndarray, draws: np.ndarray, at: int,
                      names: list[str]) -> None:
    """The panel's largest rule contrast against the legal control, at r = 0.

    A fixed offset and a named rule, not the largest excursion anywhere in the
    window: the n = 1 finding lives at the fork, and an argmax over 13 offsets
    and 3 rules would be a different, unstated statistic."""
    control = len(names)              # the first control follows the three rules
    gaps = x[:control, :, at] - x[control, :, at]
    k = int(np.abs(gaps.mean(axis=1)).argmax())
    lo, hi = boot_ci(gaps[k], draws)
    ax.text(0.97, 0.93, f"{names[k].split()[0]} {gaps[k].mean():+.2f} "
            f"[{lo:+.2f}, {hi:+.2f}] vs control at $t_0$", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=MUTED)


def population_notes(z, n_games: int) -> list[str]:
    """What the population is, and what its spread ratio says (§1.7).

    The spread is raw layer 5's, the space `rule_groups` measures the study's own
    ratio in, and it is a median over games rather than the one game's value: at
    this n the statistic has a distribution and the IQR is what a reader needs."""
    g = z["spread_game"][0]
    where = f"in {RAW_SPACE.split(' (')[0]}, per game over {n_games} games"
    pop = (f"the first {n_games} of the {int(z['n_pool'])} branch-complete games forking "
           f"at $t_0$ = {int(z['t0'])}, ascending by game id: no game-level criterion "
           f"enters this figure")
    if FIGURE_GAME in z["game"]:
        pop += (f", and game {FIGURE_GAME} — the argmax the\nsingle-game figure draws — "
                f"is in it as one ordinary member of {n_games}")
    return [f"within ÷ between-rule spread of the drawn ratio over $t \\geq t_0$, {where}: "
            f"median {np.median(g):.3f} [IQR {np.percentile(g, 25):.3f}–"
            f"{np.percentile(g, 75):.3f}], above 1 in {(g > 1).mean():.0%} of them.\nNear "
            f"1 the three curves are a label rather than a finding; the nine spaces' "
            f"values are in README.md.", NULL_NOTE, pop + " (README)."]


def population_figure(z, draws: np.ndarray, camera_ready: bool) -> Path:
    rows, names = rows_of(z), list(z["space"])
    rules = list(rule_rows(rows))
    pos, t0, n_games = z["pos"], int(z["t0"]), len(z["game"])
    at = int(np.flatnonzero(pos == t0)[0])
    nrow = -(-len(names) // NCOL)
    fig = plt.figure(figsize=(CELL_W * NCOL, CELL_H * nrow + BAND_H))
    gs = fig.add_gridspec(2, 1, height_ratios=[CELL_H * nrow, BAND_H],
                          left=0.07, right=0.98, top=0.90, bottom=0.03, hspace=0.16)

    head = fig.add_subplot(gs[0])
    head.axis("off")
    head.patch.set_visible(False)
    head.set_title("step length at the fork, as a multiple of the reference branch's at "
                   "the same position — one curve per constraint\nrule, the mean over "
                   f"$n$ = {n_games} source games of that game's own ratio; bands are a "
                   f"95% percentile bootstrap CI over games", loc="left", fontsize=10.5,
                   pad=16)

    def draw(ax, j):
        x = drawn_series(z, j)
        lo, hi = boot_ci(np.moveaxis(x, 0, 1), draws)
        rule_ratio_population_panel(ax, x.mean(axis=1), lo, hi, rows, pos, t0,
                                    legend=(j == 0))
        annotate_contrast(ax, x, draws, at, rules)

    axes = panel_grid(fig, gs[0], names, pos, t0, draw)

    band = fig.add_subplot(gs[1])
    band.axis("off")
    leg = ordered_legend(band, axes[0], 3, loc="upper left", bbox_to_anchor=(0.0, 1.05))
    y = below(fig, leg, 0.012)
    for note in [SAMPLING_NOTE, *population_notes(z, n_games)]:
        y = below(fig, fig.text(0.005, y, note, ha="left", va="top", fontsize=7.5,
                                color=MUTED, linespacing=1.5), 0.006)
    rule_member_key(fig, x=0.005, y=y)

    if not camera_ready:
        fig.suptitle(f"Step Length at the Fork in Each Representation Space — "
                     f"{n_games} games at $t_0$ = {t0}, layer 5", fontsize=12)
    return save_figure(fig, FIG_DIR / f"n{n_games}_step_length_fork", camera_ready)


def population_table(z, draws: np.ndarray) -> list[dict]:
    """One row per (space, series, position): the drawn mean with its bootstrap CI
    and its median, the absolute step length the ratio divides away, and the
    paired contrast against the legal control on the same games.

    The contrast is a column and not a second table because an unpaired ordering
    of two point estimates is what this study already had to correct once: a
    game's branches share a prefix and the reference they are divided by, so the
    difference has to be taken inside the game."""
    rows, t0 = rows_of(z), int(z["t0"])
    control = next(i for i, r in enumerate(rows) if r["is_control"] == 1)
    out = []
    for si, space in enumerate(z["space"]):
        c = z["ratio"][si][:, control]
        for kind, name, members in series(rows):
            x = z["ratio"][si][:, members].mean(axis=1)
            s = z["step"][si][:, members].mean(axis=1)
            lo, hi = boot_ci(x, draws)
            gap_lo, gap_hi = boot_ci(x - c, draws)
            out += [{"space": space, "t0": t0, "n_games": len(z["game"]), "kind": kind,
                     "series": name, "n_members": len(members),
                     "t": int(t), "r": int(t) - t0,
                     "step_length": f"{s[:, j].mean():.6g}",
                     "ratio": f"{x[:, j].mean():.6g}",
                     "ratio_lo": f"{lo[j]:.6g}", "ratio_hi": f"{hi[j]:.6g}",
                     "ratio_median": f"{np.median(x[:, j]):.6g}",
                     "vs_control": f"{(x - c)[:, j].mean():.6g}",
                     "vs_control_lo": f"{gap_lo[j]:.6g}",
                     "vs_control_hi": f"{gap_hi[j]:.6g}"}
                    for j, t in enumerate(z["pos"])]
    return out


def cells(x: np.ndarray, draws: np.ndarray, keep) -> str:
    """One printed row of `mean [lo, hi]` per series, over the positions `keep`."""
    return "  ".join(f"{v.mean():6.3f} [{lo:.3f}, {hi:.3f}]"
                     for v, (lo, hi) in ((s, boot_ci(s, draws))
                                         for s in x[:, :, keep].mean(axis=2)))


def report(z, draws: np.ndarray) -> None:
    rows, names = rows_of(z), [n.split(" (")[0] for n in z["space"]]
    labels = list(rule_rows(rows)) + ["control_legal", "control_legal_2"]
    pos, t0 = z["pos"], int(z["t0"])
    at, post = pos == t0, pos >= t0
    X = np.stack([drawn_series(z, si) for si in range(len(names))])   # [S, K, G, T]
    control = len(rule_rows(rows))

    print(f"{len(z['game'])} of the {int(z['n_pool'])} branch-complete games forking at "
          f"t0 = {t0}; r = {pos[0] - t0} … {pos[-1] - t0} at stride {STRIDE}\n"
          f"mean over games of the per-game ratio, [95% percentile bootstrap CI over "
          f"games, {BOOT} resamples, seed {BOOT_SEED}]")
    for where, keep in (("at r = 0", at), ("mean over r >= 0", post)):
        print(f"\n{where}\n{'space':<14} " + "  ".join(f"{l.split()[0]:^21}" for l in labels))
        for si, name in enumerate(names):
            print(f"  {name:<12} {cells(X[si], draws, keep)}")

    gaps = np.concatenate([X[:, :control] - X[:, control:control + 1],
                           X[:, control + 1:] - X[:, control:control + 1]], axis=1)
    for where, keep in (("at r = 0", at), ("mean over r >= 0", post)):
        print(f"\npaired on the same games, series − control_legal, {where}\n"
              f"{'space':<14} " + "  ".join(f"{l.split()[0]:^21}"
                                            for l in labels[:control] + labels[-1:]))
        for si, name in enumerate(names):
            print(f"  {name:<12} {cells(gaps[si], draws, keep)}")

    print(f"\ndecay of each rule's mean ratio over r = 0, 2, 4, 6, 8")
    for si, name in enumerate(names):
        print(f"  {name:<12} " + "  ".join(
            f"{labels[k].split()[0]} "
            + " ".join(f"{v:.3f}" for v in X[si, k][:, post].mean(0)[:5])
            for k in range(control)))
    excursion = np.abs(X[:, control][:, :, post].mean(1) - 1).max(axis=1)
    late = pos > t0 + 4
    outside = (np.abs(X[:, :control][:, :, :, late].mean(2) - 1)
               > excursion[:, None, None]).sum(axis=(1, 2))
    print(f"\ncontrol_legal's own largest |mean − 1| over r >= 0, and how many of the "
          f"{control} x {late.sum()} rule-mean values past r = +4 leave that band")
    for si, name in enumerate(names):
        print(f"  {name:<12} band ±{excursion[si]:.3f}   outside {outside[si]}")

    print("\nwithin/between rule spread ratio, per game: median [IQR], above 1 in")
    for si, name in enumerate(names):
        g = z["spread_game"][si]
        print(f"  {name:<12} {np.median(g):.3f} [{np.percentile(g, 25):.3f}–"
              f"{np.percentile(g, 75):.3f}]  {(g > 1).mean():.0%}")


def population(n_games: int, camera_ready: bool) -> None:
    npz = study_dir("divergence") / "step_length_fork" / f"n{n_games}.npz"
    z = dict(np.load(npz))
    assert len(z["game"]) == n_games, f"{npz.name} holds {len(z['game'])} games"
    draws = np.random.default_rng(BOOT_SEED).integers(n_games, size=(BOOT, n_games))

    report(z, draws)
    out = RESULTS / f"step_length_fork_n{n_games}.csv"
    rows = population_table(z, draws)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {population_figure(z, draws, camera_ready)}\nwrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=FIGURE_GAME)
    parser.add_argument("--n_games", type=int, default=None,
                        help="draw measure.py's population twin instead of one game")
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.n_games:
        population(args.n_games, args.camera_ready)
    else:
        one_game(args.game, args.camera_ready)


if __name__ == "__main__":
    main()
