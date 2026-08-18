"""How long a step the branch takes, against the step the played move takes.

The fork family draws where a branch goes; this draws how fast it gets there.
For each branch b of one source game, at the study's sampled positions,

    s_b(t) = ||z_b(t) - z_b(t - STRIDE)||

and the panels draw the paired ratio s_b(t) / s_ref(t) -- the branch's step
length against the reference branch's at the same position in the same game.
The nine spaces differ ~580x in absolute step length (raw layer 5 steps 18.6,
SpaDE 0.032), so a raw step length is not a quantity nine cells can share; the
within-game ratio is, the reference is then exactly 1, and the legal control's
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

One game (n = 1): an illustration of the construction, not a claim. The three
rule curves are means over 2, 3 and 1 member categories, drawn faint behind
them, and the within/between-rule spread ratio rides on the figure.

Usage: python analysis/violation_divergence/step_length_fork/make_figures.py
       [--game 1235] [--camera-ready]
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
from spade.figures.branch_panels import rule_ratio_panel
from spade.figures.branch_style import below, ordered_legend, rule_member_key
from spade.figures.paper_style import MUTED, apply_style, save_figure
from spade.othello.taxonomy import CATEGORIES
from spade.violation_divergence.branches import (
    FIGURE_GAME, Game, complete_games, load_game, reference_row)
from spade.violation_divergence.fork_plane import (
    RAW_SPACE, STRIDE, positions, space_features, spaces)
from spade.violation_divergence.rule_groups import rule_rows, spread_ratio

FIG_DIR = Path(__file__).resolve().parent / "figures"
CSV = (Path(__file__).resolve().parents[3] / "results" / "violation_divergence"
       / "step_length_fork.csv")
NCOL = 3
CELL_W, CELL_H = 3.2, 2.6
BAND_H = 1.1          # the legend and the notes under the grid
TICK_EVERY = 6        # x ticks in plies; a multiple of STRIDE through t0
SELECTION_NOTE = (f"game {FIGURE_GAME} is selected, not typical: the argmax over 1550 "
                  f"branch-complete games of its illegal branches' mutual separation "
                  f"(README)")
SAMPLING_NOTE = (f"one step spans {2 * STRIDE} plies: positions are sampled every "
                 f"{STRIDE} plies at $t_0$'s parity (fork_plane.positions), the fork "
                 f"family's convention. ../rule_curvature samples every ply instead,\n"
                 f"where the same layer-5 legal corpus steps 29.2 against 18.9 here "
                 f"(../theta_comparability) — the two conventions measure different "
                 f"quantities and must never share an axis.")


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
    """(kind, name, member row indices) per drawn series, in legend order."""
    return ([("reference", "reference", [reference_row(rows)])]
            + [("rule", rule, members) for rule, members in rule_rows(rows).items()]
            + [("category", CATEGORIES[rows[i]["category"]], [i])
               for members in rule_rows(rows).values() for i in members]
            + [("control", "control_legal",
                [next(i for i, r in enumerate(rows) if r["is_control"] == 1)])])


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


def figure(g: Game, pos: np.ndarray,
           measured: list[tuple[str, np.ndarray, np.ndarray]], spread: float,
           camera_ready: bool) -> Path:
    nrow = -(-len(measured) // NCOL)
    fig = plt.figure(figsize=(CELL_W * NCOL, CELL_H * nrow + BAND_H))
    gs = fig.add_gridspec(2, 1, height_ratios=[CELL_H * nrow, BAND_H],
                          left=0.07, right=0.98, top=0.90, bottom=0.03, hspace=0.16)
    grid = gs[0].subgridspec(nrow, NCOL, wspace=0.10, hspace=0.40)

    head = fig.add_subplot(gs[0])
    head.axis("off")
    head.patch.set_visible(False)
    # The estimator and n ride above the grid, not in the suptitle: camera-ready
    # drops the suptitle and both are what keeps the curves readable (§1, §5).
    head.set_title("step length at the fork, as a multiple of the reference branch's at "
                   "the same position — one curve per\nconstraint rule over its member "
                   "categories, drawn faint; $n$ = 1 game, so no interval over games",
                   loc="left", fontsize=10.5, pad=16)

    axes = []
    for j, (space, _, ratio) in enumerate(measured):
        ax = fig.add_subplot(grid[j // NCOL, j % NCOL],
                             sharex=axes[0] if axes else None,
                             sharey=axes[0] if axes else None)
        axes.append(ax)
        rule_ratio_panel(ax, ratio, g.rows, pos[1:], g.t0, legend=(j == 0))
        annotate_peak(ax, g.rows, pos[1:], ratio, g.t0)
        ax.set_title(space, loc="left", fontsize=8.5)
        ax.set_xticks([t for t in pos[1:] if (t - g.t0) % TICK_EVERY == 0])
        ax.tick_params(labelsize=8, labelbottom=j >= len(measured) - NCOL,
                       labelleft=j % NCOL == 0)
    for ax in axes[-NCOL:]:
        ax.set_xlabel(f"move index $t$ (branch at $t_0$ = {g.t0})", fontsize=9)
    for ax in axes[::NCOL]:
        ax.set_ylabel("step length $\\div$ the reference\nbranch's, "
                      "$s_b(t) / s_{ref}(t)$", fontsize=9)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=FIGURE_GAME)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    device = select_device()
    base = study_dir("divergence")   # the spliced tree; the rollout set forks its suffix too
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)

    assert args.game in complete_games(meta), \
        f"game {args.game} does not carry every branch type"
    g = load_game(meta, args.game)
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

    CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = table(g, pos, measured)
    with CSV.open("w", newline="") as fh:
        out = csv.DictWriter(fh, list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    out_fig = figure(g, pos, measured, spread[RAW_SPACE], args.camera_ready)
    print(f"wrote {out_fig}\nwrote {CSV}")


if __name__ == "__main__":
    main()
