"""Curvature over the board, forked at t0 into the three rules a branch violates.

Three panels. (a) is the population clock: the straightness study's many-game
class profiles -- legal, OthelloGPT rollout, championship, 2000 games each -- and
the three rules measured over every game that forks at one position (--t0,
default 20, which is FIGURE_GAME's own fork: ~80 games of the 2000).
(b) is one game's violation-locked response -- game 1235, branched at t0 = 20 into
R1 placement, R2 bracketing and R3 turn -- as Δ against the matched source game,
which here is the game's own reference branch. (c) is the board the branches fork
on, with each branch move on its square.

**(a) carries six series, so line style is load-bearing.**
`CLASS_STYLE["rollout"]` and `RULE_STYLE["R1 placement"]` are the same blue --
two style tables built for two figures -- so the rule curves take a long dash and
their rule marker, and no series on that axes is named by colour alone
(FIGURE_GUIDELINES 1). The dash also separates the two constructions: a class is
a whole game, a rule is one branched move inside one, at the dashed t0.

**(a)'s population is one fork position, not all of them.** t0 is round-robin
over [10, 34] across the 2000 rule-complete games, and where a violation happens
moves curvature on its own, so pooling them would put the fork position inside
the rule curves. Restricting to t0 = 20 costs a factor of 25 in n -- the CI is
5x wider and the legend and the caption say the real number -- and buys a panel
whose dashed line is the same fork (b) and (c) draw.

**The sampling convention is load-bearing.** Both sides are measured per ply over
positions 1..58, one angle at its own vertex, no stride and no trailing window --
`rule_curvature` explains why that is not a free choice and README.md carries the
check. The fork figures of this study sample every 2 plies instead; the same
branches read 107.9 deg there against 142.9 deg here, and the class contrasts
change sign between the two, so the two conventions must never share an axis.

Two trees, one code path. `--tree branches` reads the spliced set, where every
branch replays the source game's tokens[t0+1:]; `--tree rollout` reads the set
whose continuations are resampled from OthelloGPT. Same game, same t0, same three
rules both times, so the difference between the two figures is attributable to
the suffix and to nothing else. What the spliced tree's suffix costs the reader is
in README.md under Captions, and it is not optional.

Usage: python analysis/violation_divergence/rule_curvature/make_figures.py
       [--tree branches|rollout] [--game 1235] [--t0 20|all]
       [--space raw|<family>|all] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.figures.branch_panels import board_note, board_panel, mover_name
from spade.figures.branch_style import (
    below, branch_style, ordered_legend, rule_member_key)
from spade.figures.paper_style import (
    GRID, MUTED, RULE_STYLE, apply_style, save_figure)
from spade.figures.straightness_panels import (
    ci_bars, class_blocks, event_axes, profile_panel)
from spade.othello.boards import flip_set, replay
from spade.othello.taxonomy import LEGAL
from spade.violation_divergence import rule_curvature as rc
from spade.violation_divergence.fork_plane import spaces
from spade.violation_divergence.rule_groups import rule_rows

FIG_DIR = Path(__file__).resolve().parent / "figures"
GEOM = study_dir("straightness") / "geometry"
# The straightness study's own violation class is a different construction -- a
# no-flip placement with the turn passed, over 2000 games -- so drawing it beside
# this fork would put two unrelated violations under one word.
CLASSES = ["legal", "rollout", "championship"]
TREE_TITLE = {"branches": "spliced suffix", "rollout": "resampled suffix"}
NCOL = 3
PANEL_H, BOARD_H, CELL_W, CELL_H = 3.9, 3.6, 3.2, 2.8
CONTROL = 1                    # one legal-move null; a second said nothing more
POPULATION_LS = (0, (6, 1.6))  # (a)'s rules: a branch population, not a game class


def space_stem(entry) -> str:
    return "raw_L5" if entry is None else entry.config.removesuffix("Trainer-othello")


def rule_style(fork: rc.Fork, members: list[int]) -> dict:
    return branch_style(fork.rows[members[0]]["category"], 0, by_rule=True)


def draw_rule(ax, x: np.ndarray, y: np.ndarray, st: dict, label: str | None) -> None:
    """One rule's mean curve in (b), with no uncertainty display at all.

    n = 1 game there, and nothing honest fits: +/-1.96 SEM ribbons swallowed the
    curves they annotated, and the faint per-member traces that replaced them were
    clutter. Panel (a) is where the interval lives, over games. What a single
    game's between-member spread was is printed by `main` and quoted in the study
    README, which is where an n = 1 number belongs.
    """
    ax.plot(x, y, ls=st["ls"], color=st["color"], lw=st["lw"],
            marker=st["marker"], ms=4, markevery=(2, 5), markerfacecolor=st["mfc"],
            markeredgecolor=st["mec"], markeredgewidth=0.5, zorder=st["zorder"],
            label=label)


def rule_n(pop: rc.Population) -> str:
    """(a)'s rule line, now that the legend carries only the rule name. The three
    n need not match: a rule takes every selected game with a member available."""
    lo, hi = pop.t0_range
    where = f"$t_0$ = {lo}" if lo == hi else f"$t_0 \\in$ [{lo}, {hi}]"
    n = sorted(set(int(g) for g in pop.n_games))
    each = f"$n$ = {n[0]}" if len(n) == 1 else f"$n$ = {n[0]}-{n[-1]}"
    per = "" if pop.n_rollouts == 1 else f" $\\times$ {pop.n_rollouts} rollouts"
    return f"forked branches, {each} games{per} at {where}"


def t0_line(ax, t0: int) -> None:
    ax.axvline(t0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)


def class_profile(ax, fork: rc.Fork, entry, tree: str, blocks: dict,
                  t0: int | None, note: bool = True) -> rc.Population:
    """(a) the population clock: three game classes and the three rules.

    The classes are the straightness study's committed npz; the rules are measured
    here on the same tree the fork panels read, per ply and over the games forking
    at `t0` -- one position, so the dashed line is where every rule curve forks and
    is the fork (b) locks to. Under `--t0 all` the games fork all over [10, 34]
    and that range is shaded instead. The classes get neither: they are whole
    games and do not fork.
    """
    pop = rc.population(tree, entry, t0)
    lo, hi = pop.t0_range
    if lo == hi:
        where_note = f"every rule forks here, at $t_0$ = {lo}"
        t0_line(ax, lo)
    else:
        where_note = f"the rules fork anywhere in $t_0 \\in$ [{lo}, {hi}]"
        ax.axvspan(lo, hi, color=GRID, alpha=0.5, lw=0, zorder=0)
    if note:
        ax.annotate(where_note, xy=(lo + 0.6, 0.97), xycoords=("data", "axes fraction"),
                    fontsize=7.5, color=MUTED, ha="left", va="top")
    profile_panel(ax, np.load(GEOM / f"{rc.space_key(entry)}_raw.npz")["theta"], blocks)
    for j, rule in enumerate(rc.RULES):
        st = RULE_STYLE[rule]
        at = np.arange(6 + j, len(pop.positions), 9)
        ci_bars(ax, pop.positions[at], pop.mean[j][at], 1.96 * pop.sem[j][at], st["color"])
        ax.plot(pop.positions, pop.mean[j], ls=POPULATION_LS, color=st["color"], lw=1.5,
                marker=st["marker"], ms=4, markevery=(6 + j, 9),
                markerfacecolor=st["color"], markeredgecolor="white",
                markeredgewidth=0.5, zorder=4, label=st["label"])
    if lo != fork.t0:  # the drawn game's own fork, when the population is elsewhere
        t0_line(ax, fork.t0)
    ax.set_xlabel("board position $t$")
    return pop


def control_row(fork: rc.Fork, control: int) -> int:
    return next(i for i, r in enumerate(fork.rows) if r["is_control"] == control)


def locked(ax, fork: rc.Fork, legend: bool) -> None:
    """The violation-locked Δ per rule, against the legal control as the null.

    (b) in the per-space figure and every cell of the combined one, so it carries
    no panel letter of its own.
    """
    event_axes(ax)
    rels, rules = rc.rule_locked(fork)
    for control in (CONTROL,):
        st = branch_style(LEGAL, control)
        draw_rule(ax, rels, rc.locked(fork, [control_row(fork, control)])[1].mean, st,
                  st["label"] if legend else None)
    for rule, members in rule_rows(fork.rows).items():
        st = rule_style(fork, members)
        draw_rule(ax, rels, rules[rule].mean, st, st["label"] if legend else None)
    ax.set_xlabel("position relative to the branch ($t - t_0$)")
    ax.set_ylabel("Δ turning angle vs. the game's\nreference branch (deg)")


def apparatus(fig, cell, fork: rc.Fork, keyed, ncol: int, letter: str,
              board_w: float = BOARD_H):
    """The board the branches fork on, and the band beside it: one legend for the
    whole figure, the board's reading key, and the R1/R2/R3 key under everything.

    The board is square and the panels above are wide, so the band is space the
    layout already spends; FIGURE_GUIDELINES 5 says that is where the shared
    apparatus goes rather than a second legend.
    """
    lower = cell.subgridspec(1, 2, wspace=0.02,
                             width_ratios=[board_w, ncol * CELL_W - board_w])
    board = replay(fork.tokens[:fork.t0])
    ax = fig.add_subplot(lower[0])
    board_panel(ax, board, fork.rows, flip_set(board.state,
                fork.rows[fork.ref]["square"], board.next_hand_color), by_rule=True)
    ax.set_title(f"({letter}) the position at $t_0$ = {fork.t0} "
                 f"({mover_name(board).capitalize()} to move),\nand the move each "
                 f"branch plays there", loc="left", fontsize=11)

    band = fig.add_subplot(lower[1])
    band.axis("off")
    leg = ordered_legend(band, keyed, 2, loc="center left", bbox_to_anchor=(0.0, 0.62))
    band.text(0.0, below(band, leg, 0.06), board_note(board, by_rule=True),
              ha="left", va="top", fontsize=8, color=MUTED, linespacing=1.5)
    rule_member_key(fig, x=0.005, y=-0.05)


def title(entry, label: str, tree: str, fork: rc.Fork) -> str:
    where = label if entry is None else f"{label} latents (layer-5 SAE, raw)"
    return (f"Curvature in {where} — game {fork.game} forked at $t_0$ = {fork.t0}, "
            f"{TREE_TITLE[tree]}")


def per_space(fork: rc.Fork, entry, label: str, tree: str, blocks: dict,
              t0: int | None, camera_ready: bool) -> Path:
    fig = plt.figure(figsize=(9.6, PANEL_H + BOARD_H + 1.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[PANEL_H, BOARD_H],
                          left=0.07, right=0.98, top=0.87, bottom=0.06, hspace=0.32)
    # Board position and offset-from-the-branch are unrelated axes, so the two
    # panels sit side by side rather than stacking on a shared x.
    top = gs[0].subgridspec(1, 2, wspace=0.30, width_ratios=[1.3, 1])

    ax_a = fig.add_subplot(top[0])
    pop = class_profile(ax_a, fork, entry, tree, blocks, t0)
    ax_a.set_title("(a) the population's curvature clock\nclasses solid: whole games, "
                   "no fork, $n$ = 2000 each\n"
                   f"rules dashed: {rule_n(pop)}\n"
                   "mean $\\pm$ 95% CI over games, at the markers",
                   loc="left", fontsize=9.5)

    ax_b = fig.add_subplot(top[1])
    locked(ax_b, fork, legend=True)
    ax_b.set_title("(b) violation-locked response\n$n$ = 1 game — no uncertainty "
                   "display", loc="left", fontsize=9.5)

    apparatus(fig, gs[1], fork, [ax_a, ax_b], NCOL, "c")
    if not camera_ready:
        fig.suptitle(title(entry, label, tree, fork), fontsize=12)
    return save_figure(fig, FIG_DIR / f"curvature_rules_{tree}_{space_stem(entry)}",
                       camera_ready)


def combined(forks: list[tuple[str, object, rc.Fork]], tree: str,
             camera_ready: bool) -> Path:
    """The violation-locked panel in every representation space, shared axes.

    Only the violation-locked panel: it is the measurement, and nine copies of the
    population clock would repeat what the per-space files already carry. Turning
    angles are scale-free, so the nine cells really do share a y-axis, which
    latent norms and step lengths would not.
    """
    nrow = -(-len(forks) // NCOL)
    fig = plt.figure(figsize=(NCOL * CELL_W, CELL_H * nrow + BOARD_H + 0.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[CELL_H * nrow, BOARD_H],
                          left=0.07, right=0.98, top=0.90, bottom=0.02, hspace=0.28)
    grid = gs[0].subgridspec(nrow, NCOL, wspace=0.12, hspace=0.42)

    head = fig.add_subplot(gs[0])
    head.axis("off")
    head.patch.set_visible(False)
    head.set_title("(a) the violation-locked response in every representation "
                   "space — $n$ = 1 game, no uncertainty display",
                   loc="left", fontsize=11, pad=16)

    axes = []
    for j, (label, _, fork) in enumerate(forks):
        ax = fig.add_subplot(grid[j // NCOL, j % NCOL],
                             sharex=axes[0] if axes else None,
                             sharey=axes[0] if axes else None)
        axes.append(ax)
        locked(ax, fork, legend=(j == 0))
        ax.set_title(label, loc="left", fontsize=8.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[-NCOL:]:
        ax.set_xlabel("position relative to the branch ($t - t_0$)")
    for ax in axes[::NCOL]:
        ax.set_ylabel("Δ turning angle vs. the\nreference branch (deg)")

    apparatus(fig, gs[1], forks[0][2], axes[0], NCOL, "b")
    if not camera_ready:
        fig.suptitle(f"Curvature response to a rule violation across nine spaces "
                     f"(layer 5, raw, {TREE_TITLE[tree]})", fontsize=12)
    return save_figure(fig, FIG_DIR / f"curvature_rules_{tree}_combined", camera_ready)


def combined_population(forks: list[tuple[str, object, rc.Fork]], tree: str,
                        blocks: dict, t0: int | None, camera_ready: bool) -> Path:
    """The population clock in every representation space, shared axes.

    The companion to `combined`: that one carries the violation-locked response of
    a single game, this one the many-game population the response is read against.
    No board here -- (a) is a population forking at one position, and drawing one
    game's board beside nine population panels would invite reading the population
    as that game.
    """
    nrow = -(-len(forks) // NCOL)
    fig = plt.figure(figsize=(NCOL * CELL_W, CELL_H * nrow + 1.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[CELL_H * nrow, 1.0],
                          left=0.07, right=0.98, top=0.90, bottom=0.03, hspace=0.30)
    grid = gs[0].subgridspec(nrow, NCOL, wspace=0.12, hspace=0.42)

    head = fig.add_subplot(gs[0])
    head.axis("off")
    head.patch.set_visible(False)
    head.set_title("(a) the population's curvature clock in every representation "
                   "space", loc="left", fontsize=11, pad=16)

    axes = []
    for j, (_, entry, fork) in enumerate(forks):
        ax = fig.add_subplot(grid[j // NCOL, j % NCOL],
                             sharex=axes[0] if axes else None,
                             sharey=axes[0] if axes else None)
        axes.append(ax)
        pop = class_profile(ax, fork, entry, tree, blocks, t0, note=False)
        ax.set_title(forks[j][0], loc="left", fontsize=8.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[-NCOL:]:
        ax.set_xlabel("board position $t$")
    for ax in axes[::NCOL]:
        ax.set_ylabel("mean turning angle (deg)")

    band = fig.add_subplot(gs[1])
    band.axis("off")
    ordered_legend(band, axes[0], 3, loc="upper left", bbox_to_anchor=(0.0, 1.0))

    if not camera_ready:
        fig.suptitle(f"The curvature clock across nine spaces "
                     f"(layer 5, {rule_n(pop)}, {TREE_TITLE[tree]})", fontsize=12)
    return save_figure(fig, FIG_DIR / f"curvature_rules_{tree}_population_combined",
                       camera_ready)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", default="branches", choices=list(rc.TREE_VARIANT))
    parser.add_argument("--game", type=int, default=rc.FIGURE_GAME)
    parser.add_argument("--t0", default=str(rc.FIGURE_T0),
                        help="(a)'s fork position, or 'all' to pool every position")
    parser.add_argument("--space", default="all",
                        help="'all', 'raw', or an SAE family (e.g. TopK)")
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()
    t0 = None if args.t0 == "all" else int(args.t0)

    apply_style(args.camera_ready)
    device = select_device()
    cls = torch.load(study_dir("straightness") / "games" / "games.pt",
                     weights_only=True)["cls"]
    blocks = {c: b for c, b in class_blocks(len(cls)).items() if c in CLASSES}

    cells = [(label, entry) for label, entry in spaces()
             if args.space == "all" or space_stem(entry).startswith(
                 "raw" if args.space == "raw" else args.space)]
    assert cells, f"no representation space matches --space {args.space!r}"

    forks, written = [], []
    for label, entry in cells:
        fork = rc.load_fork(args.tree, args.game, entry, device)
        rc.check_prefix(fork)
        forks.append((label, entry, fork))
        written.append(per_space(fork, entry, label, args.tree, blocks, t0,
                                 args.camera_ready))
        # Printed, not drawn: the README quotes these, and they have to be
        # re-derivable without measuring a figure by eye (as ../fork_and_board does).
        rels, rules = rc.rule_locked(fork)
        cols = "  ".join(
            f"{r.split()[0]} {v.mean.min():+6.2f}@{rels[v.mean.argmin()]:+3d}"
            f" tail {v.mean[rels >= 2].mean():+5.2f}"
            f"+-{np.nan if v.sem is None else np.nanmean(v.sem):.2f}"
            for r, v in rules.items())
        print(f"{label:34} spread {rc.spread(fork):.3f}  {cols}")
        # (a)'s gap is small on an axis spanning thirty, so it is not measurable off
        # that panel; here are its extremum and its own CI. The pre-fork residual is
        # the check: no branch exists there yet, so the rules and the legal class are
        # the same games measured twice, and whatever they differ by is sampling. The
        # turn at t0-1 already sees the branch token, so pre-fork ends at t0-2.
        pop = rc.population(args.tree, entry, t0)
        legal = np.nanmean(np.load(GEOM / f"{rc.space_key(entry)}_raw.npz"
                                   )["theta"][blocks["legal"]], axis=0)
        gap = pop.mean - legal
        k = np.nanargmin(gap, axis=1)
        print(f"{'':34} vs legal  pre-fork |Δ| <= "
              f"{np.nanmax(np.abs(gap[:, pop.positions <= pop.t0_range[0] - 2])):.2f}  "
              + "  ".join(f"{r.split()[0]} n={pop.n_games[j]} "
                          f"{gap[j, k[j]]:+5.2f}@{pop.positions[k[j]]}"
                          f"+-{1.96 * pop.sem[j, k[j]]:.2f}"
                          for j, r in enumerate(rc.RULES)))
    if len(cells) > 1:
        written.append(combined(forks, args.tree, args.camera_ready))
        written.append(combined_population(forks, args.tree, blocks, t0,
                                           args.camera_ready))
    print("\n".join(f"wrote {p}" for p in written))


if __name__ == "__main__":
    main()
