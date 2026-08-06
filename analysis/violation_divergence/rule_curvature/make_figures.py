"""Curvature over the board, forked at t0 into the three rules a branch violates.

Four panels. (a) is the straightness study's many-game class profiles -- legal,
OthelloGPT rollout, championship, 2000 games each. (b) is one game's fork on (a)'s
scale: game 1235 branching at t0 = 20 into R1 placement, R2 bracketing and R3
turn, over its own reference and two legal controls. (c) is the violation-locked
response, Δ against the matched source game, which here is the game's own
reference branch. (d) is the board the branches fork on, with each branch move on
its square. The vertical line at t0 in (a) and (b) is only well defined because
the violation condition is a single game with a single t0.

**Why (a) and (b) are two axes and not one overlay.** Two reasons, and either
alone is decisive. `CLASS_STYLE["rollout"]` and `RULE_STYLE["R1 placement"]` are
the same blue, so one axes carrying both would give one colour two meanings. And
`profile_panel`'s classes are population means where the fork is n = 1, so the
overlay invites reading a single game's swing as a class difference. They share
x and y, so the comparison a reader wants is still exact.

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
       [--tree branches|rollout] [--game 1235] [--space raw|<family>|all]
       [--camera-ready]
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
    MEMBER, below, branch_style, legend_proxy, ordered_legend, rule_member_key)
from spade.figures.paper_style import (
    INK, MUTED, REFERENCE, apply_style, save_figure)
from spade.figures.straightness_panels import (
    class_blocks, event_axes, profile_panel)
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


def space_key(entry) -> str:
    """The straightness geometry npz this space's class baselines live in."""
    return f"resid_post_{rc.LAYER}" if entry is None else f"sae_{entry.config}"


def space_stem(entry) -> str:
    return "raw_L5" if entry is None else entry.config.removesuffix("Trainer-othello")


def rule_style(fork: rc.Fork, members: list[int]) -> dict:
    return branch_style(fork.rows[members[0]]["category"], 0, by_rule=True)


def rule_label(fork: rc.Fork, st: dict, members: list[int]) -> str:
    """The legend entry, carrying what the mean is over (FIGURE_GUIDELINES 1.7).

    On the rollout tree a member category is K sampled continuations, so the
    count a reader needs is both numbers and not their product.
    """
    n = (f"{len(members)}" if fork.n_rollouts == 1
         else f"{len(members)} $\\times$ {fork.n_rollouts} rollouts")
    return f"{st['label']} (mean of {n})"


def draw_rule(ax, x: np.ndarray, rule: rc.Rule, st: dict, label: str | None,
              lo: int = 0) -> None:
    """One rule's mean over the sequences it averages, drawn faint behind it.

    No ribbon. Five series per axes -- three rules and two controls -- each with a
    +/-1.96 SEM fill made the overlaps unreadable, which is the failure
    FIGURE_GUIDELINES 1.7 warns about: the uncertainty display swallowed the
    curves it was annotating. The member traces show the same spread, sequence by
    sequence rather than summarised, and are what a reader should be looking at.
    R3 has one member and so never had a band to lose.
    """
    for member in rule.members:
        ax.plot(x[lo:], member[lo:], color=st["color"], **MEMBER)
    ax.plot(x[lo:], rule.mean[lo:], ls=st["ls"], color=st["color"], lw=st["lw"],
            marker=st["marker"], ms=4, markevery=(2, 5), markerfacecolor=st["mfc"],
            markeredgecolor=st["mec"], markeredgewidth=0.5, zorder=st["zorder"],
            label=label)


def t0_line(ax, fork: rc.Fork) -> None:
    ax.axvline(fork.t0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)


def class_profile(ax, fork: rc.Fork, entry, blocks: dict) -> None:
    """(a) the many-game class profiles: the population clock every game rides.

    Drawn on its own axes rather than under the fork. `CLASS_STYLE["rollout"]`
    and `RULE_STYLE["R1 placement"]` are the same blue -- two style tables built
    for two figures -- so one axes carrying both would name one colour twice
    (FIGURE_GUIDELINES 1, 5). The x and y axes are shared with (b), so the two
    registers are still read off one scale.
    """
    profile_panel(ax, np.load(GEOM / f"{space_key(entry)}_raw.npz")["theta"], blocks)
    t0_line(ax, fork)
    ax.set_xlabel("")


def fork_profile(ax, fork: rc.Fork) -> None:
    """(b) the same quantity for one game's nine branches, on (a)'s scale."""
    t0_line(ax, fork)
    pos, curves, profiles = fork.positions, rc.branch_curves(fork), rc.rule_profiles(fork)
    ax.plot(pos, curves[fork.ref], color=REFERENCE["color"], ls=(0, (3, 1.5)), lw=1.3,
            marker=REFERENCE["marker"], ms=4, markevery=(4, 9), markerfacecolor="white",
            markeredgecolor=REFERENCE["color"], markeredgewidth=0.8, zorder=4,
            label=f"reference: the move as played at $t_0$" + (
                "" if fork.n_rollouts == 1 else f" (mean of {fork.n_rollouts})"))
    # From the last shared turn, so a branch visibly leaves the reference rather
    # than appearing out of nowhere; before it every branch is that same curve.
    for control in (1, 2):
        st = branch_style(LEGAL, control)
        ax.plot(pos[fork.last_shared:], curves[control_row(fork, control), fork.last_shared:],
                ls=st["ls"], color=st["color"], lw=st["lw"], marker=st["marker"], ms=4,
                markevery=(3, 9), markerfacecolor=st["mfc"], markeredgecolor=st["mec"],
                markeredgewidth=0.5, zorder=st["zorder"], label=st["label"])
    for rule, members in rule_rows(fork.rows).items():
        st = rule_style(fork, members)
        draw_rule(ax, pos, profiles[rule], st,
                  rule_label(fork, st, members), fork.last_shared)
    legend_proxy(ax)
    ax.set_xlabel("board position $t$")
    ax.set_ylabel("mean turning angle (deg)")
    lo, hi = ax.get_ylim()
    ax.annotate(f"$t_0$ = {fork.t0}: the fork\nbranches coincide to its left",
                xy=(fork.t0 - 1.2, lo + 0.02 * (hi - lo)), fontsize=7.5, color=INK,
                ha="right", va="bottom")


def control_row(fork: rc.Fork, control: int) -> int:
    return next(i for i, r in enumerate(fork.rows) if r["is_control"] == control)


def locked(ax, fork: rc.Fork, legend: bool) -> None:
    """(b) the violation-locked Δ per rule, against both legal controls as nulls."""
    event_axes(ax)
    rels, rules = rc.rule_locked(fork)
    for control in (1, 2):
        st = branch_style(LEGAL, control)
        draw_rule(ax, rels, rc.locked(fork, [control_row(fork, control)])[1], st,
                  st["label"] if legend else None)
    for rule, members in rule_rows(fork.rows).items():
        st = rule_style(fork, members)
        draw_rule(ax, rels, rules[rule], st,
                  rule_label(fork, st, members) if legend else None)
    if legend:
        legend_proxy(ax)
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
              camera_ready: bool) -> Path:
    fig = plt.figure(figsize=(9.6, 2 * PANEL_H + BOARD_H + 0.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2 * PANEL_H, BOARD_H],
                          left=0.07, right=0.98, top=0.92, bottom=0.05, hspace=0.30)
    # (a) and (b) stack on one shared position axis; (c) has its own, so it takes
    # the full height beside them rather than leaving a band of white.
    top = gs[0].subgridspec(2, 2, wspace=0.34, hspace=0.24, width_ratios=[1.3, 1])

    ax_a = fig.add_subplot(top[0, 0])
    class_profile(ax_a, fork, entry, blocks)
    ax_a.tick_params(labelbottom=False)
    ax_a.set_title("(a) the population's curvature clock\nmean $\\pm$ 95% CI, "
                   "$n$ = 2000 games per class", loc="left", fontsize=9.5)
    ax_a.legend(fontsize=7, loc="lower center", ncol=3, columnspacing=1.0)

    ax_b = fig.add_subplot(top[1, 0], sharex=ax_a, sharey=ax_a)
    fork_profile(ax_b, fork)
    ax_b.set_title("(b) one game's fork, on (a)'s scale\n$n$ = 1 game; every "
                   "sequence a rule averages is drawn faint", loc="left", fontsize=9.5)

    ax_c = fig.add_subplot(top[:, 1])
    locked(ax_c, fork, legend=True)
    ax_c.set_title("(c) violation-locked response\nevery sequence a rule averages "
                   "is drawn faint; $n$ = 1 game", loc="left", fontsize=9.5)

    apparatus(fig, gs[1], fork, [ax_a, ax_b, ax_c], NCOL, "d")
    if not camera_ready:
        fig.suptitle(title(entry, label, tree, fork), fontsize=12)
    return save_figure(fig, FIG_DIR / f"curvature_rules_{tree}_{space_stem(entry)}",
                       camera_ready)


def combined(forks: list[tuple[str, object, rc.Fork]], tree: str,
             camera_ready: bool) -> Path:
    """The violation-locked panel in every representation space, shared axes.

    Only panel (b): it is the measurement, and nine copies of the class profiles
    would be nine copies of a figure the per-space files already carry. Turning
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
    head.set_title("(a) the violation-locked response in every representation space",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", default="branches", choices=list(rc.TREE_VARIANT))
    parser.add_argument("--game", type=int, default=rc.FIGURE_GAME)
    parser.add_argument("--space", default="all",
                        help="'all', 'raw', or an SAE family (e.g. TopK)")
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

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
        written.append(per_space(fork, entry, label, args.tree, blocks,
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
    if len(cells) > 1:
        written.append(combined(forks, args.tree, args.camera_ready))
    print("\n".join(f"wrote {p}" for p in written))


if __name__ == "__main__":
    main()
