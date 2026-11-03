# Panel (b) is gone (branch `fig/drop-panel-b`)

`analysis/violation_divergence/rule_curvature`'s per-space figure drew one game's
nine branches as θ-vs-position on panel (a)'s scale. It is out. What is left is
the population clock and the violation-locked response, and the board renumbers
from (d) to (c).

- `analysis/violation_divergence/rule_curvature/make_figures.py` — `fork_profile`
  deleted; the layout reworked; (a) given its x axis back.
- `src/spade/violation_divergence/rule_curvature.py` — four things lost their
  last caller with it.
- `analysis/violation_divergence/rule_curvature/README.md` — captions, panel
  inventory, and the two passages that argued for (b)'s existence.
- 18 of the 20 figures re-rendered. The two `_combined` PNGs came back
  **byte-identical**, which is the check that the combined figure never touched
  (b): it draws only the violation-locked panel and passes its own board letter.

## What died with it

`fork_profile` was the only caller of five things, and every one was checked
against the whole tree before deleting:

| gone | why it had no other caller |
|---|---|
| `rc.branch_curves` | `rule_profiles` was its only other user, and `_game_rule_means` computes the same `nanmean(theta, axis=1)` inline |
| `rc.rule_profiles` | drew (b)'s three rule paths; `Rule` survives, `locked` still builds it |
| `Fork.n_rollouts` | only (b)'s "(mean of K)" legend suffix read it. `Population.n_rollouts` is a different field and still used |
| `Fork.positions` | only (b)'s x. Every other `.positions` in the tree is `Population.positions` |
| `draw_rule`'s `lo` argument | (b) started its curves at `last_shared`; the locked panel always passed 0 |

Also out: the `REFERENCE` import (no user left in the file — checked, the board
takes its styling from `branch_style` inside `branch_panels`), the `INK` import,
and the in-panel "$t_0$ = 20: the fork / branches coincide to its left"
annotation, which named a vertical line that no longer has a panel.

**`fork.ref` stays.** The board reads `fork.rows[fork.ref]["square"]` to compute
the flip set, and `aligned_delta`/`check_prefix` read it to build Δ. Only the
*drawn reference curve* went.

**`Rule.n_members` stays, and is dead.** Nothing reads it, but that was already
true before this change — `locked` set it and no one looked — so it is not a
vestige of (b) and it is not this commit's business.

## The trap: (a) had no x axis of its own

Panel (a) sat above (b) on a shared position axis and (b) carried the tick labels
and the "board position *t*" label for both, so (a) set
`tick_params(labelbottom=False)` and `set_xlabel("")`. Deleting (b) would have
shipped an unlabelled axis (FIGURE_GUIDELINES §5). `class_profile` now sets
`board position $t$` instead of clearing it — the mathtext form, matching
(b)'s `($t - t_0$)`, rather than `profile_panel`'s plain-text default — and the
`labelbottom` call is gone. Confirmed on the render, not on the call.

The change also gave (a) a **tighter y range**: it used to share y with (b), whose
single-game curve reached 156°, so the population's 2–4° spread was drawn on a
32° axis. Alone it autoscales to 123–147° and the rule/class gap is visible.

## Layout

Was: outer `height_ratios=[2 * PANEL_H, BOARD_H]` over a 9.6 × 12.3 in canvas,
with (a) stacked over (b) in `gs[0].subgridspec(2, 2, ...)` and the locked panel
spanning both rows on the right.

Now: **9.6 × 8.7 in**, `height_ratios=[PANEL_H, BOARD_H]`, and one row of two
panels side by side. Side by side rather than stacked because board position and
offset-from-the-branch are unrelated x axes — there is nothing to share. The
column widths (1.3 : 1) are unchanged, so both panels keep their width to within
0.08 in and only their height grows (2.79 → 3.09 in). `top`/`bottom` were retuned
to keep the same absolute headroom for the suptitle over (a)'s four-line title on
a figure 3.6 in shorter. The board row is 2.85 in against 2.87 in before, so the
board and its band are untouched. Judged off the render at three spaces per tree,
not off the numbers.

## One thing I would not have removed, and did

**The board's reference square now has no legend entry.** (b) supplied
`reference: the move as played at $t_0$`, and with (b) gone the black-outlined
cell on the board panel is unnamed. This is not a regression I introduced so much
as one I propagated: the **combined figure has always looked like this** (it
harvests handles from the locked panel only), and the board's own label —
`reference (the game as played)`, from `branch_style` — never matched (b)'s
wording anyway. It cannot be fixed by adding the board axes to `ordered_legend`'s
sources: `ordered_legend` dedupes by label and later handles win, so the board's
marker-only handles would replace the rules' and the control's line handles and
the legend would lose its line styles. If you want the reference named, it wants
a proxy handle, which is a decision rather than a cleanup. Flagged, not done.

## Two stale sentences I left alone, from the previous commit

`1143f73` moved the rule *n* out of the legend and into (a)'s title, and dropped
control 2 from the curves, but did not update the README. Two claims about
apparatus are now false and neither is about (b):

- line ~267: "their legend entries read **80 games × 4 rollouts, forked at
  *t0* = 20**" — that string is in (a)'s title now, not the legend.
- line ~271: "A rule now averages over member categories × 4 rollouts, **and the
  legend says so**" — it does not; the title does.

I did fix the third one, because it was inside the caption I had to renumber and
it contradicted the panel it described: "**Both legal controls are drawn, and
they are the point**" became "the legal control is drawn", with both measured
numbers kept and a note that the panel draws the first of each pair. Verified
against the data that control 1 is the −7.99 / −5.14 of those pairs, not the
−12.38 / −9.25.

One drive-by: `class_profile`'s return annotation said `-> None` while returning
a `Population`. Now `-> rc.Population`.

## Mechanics

`/n/home10/dcao/.conda/envs/spade_env/bin/python` with `PYTHONPATH=$PWD/src`.
Both trees rendered end to end on the login node, **exit 0** each — no exit 137,
so nothing hit the 8 GB cap. `checkpoints/**/ae.pt` is gitignored and absent from
a fresh worktree; symlinked read-only to `/n/home10/dcao/spade_sae/checkpoints/`,
as `single-t0.md` records for the previous one. They stay.

Every printed number is unchanged from the pre-change run (the measurement did
not move; only what is drawn did).

## Not done

- `PROCESS.md` untouched, as instructed. Nothing pushed.
- No docs outside `analysis/violation_divergence/rule_curvature/` needed editing:
  grepped every `.md` in the tree for references to this figure's panels, and
  nothing outside that directory names a panel past (a).
- Everything the owner asked to keep is kept — no faint member traces, no shaded
  ribbons, one legend entry per rule, one control, per-ply θ, and the population
  at a single *t0* with its n in (a)'s title.
