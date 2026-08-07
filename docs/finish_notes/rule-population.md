# Panel (a) becomes a population, and the faint member traces come out (branch `fig/rule-population`)

Two changes to `analysis/violation_divergence/rule_curvature`, and they interact.
Taking the faint per-member traces out left the figure with no uncertainty
display anywhere; putting the three rules into panel (a) at population n is where
an interval comes back, and it is the only panel that now has one.

*Renumbered since — panel (b), one game's fork on (a)'s scale, was removed: (a)
stays (a), the violation-locked response (c) → (b), the board (d) → (c). Every
panel letter below is the four-panel layout as it stood. See
`docs/finish_notes/panelb-docs.md`.*

- `src/spade/violation_divergence/rule_curvature.py` — the measurement, plus the
  new population sweep and its cache.
- `analysis/violation_divergence/rule_curvature/make_figures.py` — draws only.
- `analysis/violation_divergence/rule_curvature/README.md` — captions and numbers.
- `scripts/slurm/studies/violation_divergence/submit_rule_population.sbatch` — the
  sweep on a GPU; it does not fit a login node.
- 20 figures re-rendered under `.../rule_curvature/figures/`.

## 1. The member traces are gone, and so is the uncertainty display in (b) and (c)

`draw_rule` drew one faint line per sequence a rule averages, in panel (b), panel
(c), and every cell of the combined figure. All of it is out; `MEMBER` and
`legend_proxy` are no longer imported here. Both stay in `branch_style` —
`branch_panels` and `fork_curvature_space` still draw them — so nothing was
deleted from the shared module; the `one category inside a rule` legend entry is
gone from this figure only.

**State this plainly: panels (b) and (c) now carry no uncertainty of any
kind.** The ±1.96 SEM ribbons went in an earlier change because five of them per
axes swallowed the curves; the member traces were what carried the spread
afterwards; both are now gone. No substitute was added there and none is
available — n = 1 game, and a CI bar over 2, 3 or 1 sequences of one selected
game would assert a precision the design cannot deliver. The panel titles say
`n = 1 game — no uncertainty display`, the README's *Reading it honestly* section
says it in full, and the between-member spread a reader might have eyeballed off
the faint lines is still printed by `make_figures` (`tail -0.42+-0.47`) and quoted
in the README's spliced-vs-resampled table. **The figure is strictly less
informative about spread at n = 1 than it was, and that is the trade the owner
asked for.**

Panel (a)'s rule curves do get CI bars — `straightness_panels.ci_bars`, 95% over
games at the marker positions, the same device the class curves already used.

## 2. Panel (a) now carries R1/R2/R3 over the whole population

### Where the data had to come from

Not the straightness study. Its violation class is built from
`build_games.EMPTY_CATEGORIES = [3, 4, 5, 6]` — empty-square violations only — so
it contains **no R1 at all**, and no R1/R2/R3 split at n = 2000 exists there to
read. The curves are measured here off `branches.pt`, one branch per illegal
category per game, per ply over positions 1..58, no stride and no trailing
window. That is the same convention as panels (b)–(c) and the same as the class
baselines, which is the only reason the six series may share an axis
(`../theta_comparability` documents the −0.508 correlation that a stride-2 curve
would bring instead).

### n is per rule: 2000, 2000, 1997

Every source game affords an R1 and an R2 branch. **Three games afford no
`opponent_only` square at their *t0*, so an R3 violation cannot be constructed
there at all** — a property of those positions, not a filter, and no relaxation
recovers them. Intersecting to 1997 across the board would have thrown away real
R1 and R2 data to buy a matching number, so each rule uses every game it has and
its own n is in its legend entry. Cross-rule comparisons rest on sets differing
by 3 games in 2000.

This is **not** the 1550-game branch-complete set that `game_selection` and
`rule_groups` use, and the difference has a methodological cost worth knowing:
on the full set a rule's **member composition varies by game**. R1 averages
exactly 2 categories in every game and R3 exactly 1, but **R2 averages 2.77** —
some games offer only one or two of `empty_detached`, `empty_own_only`,
`empty_no_bracket`. On the branch-complete set it is always 3. The order of
operations is unchanged regardless, and is what φ's nonlinearity forces: over a
branch's rollouts, then over the rule's member categories inside one game, then
across games. `test_curvature_plane.py` still passes.

### The curves differ by tree, and they should

Rebuilt per tree. The class baselines belong to the straightness study and know
nothing about either tree, so they are identical in both figures; the rule curves
are not, because the spliced tree replays one fixed suffix and the resampled tree
samples K = 4 continuations from OthelloGPT. Panel (a) is therefore no longer
identical across the two figures, and the README says which half moves.

It was worth building both. The population effect **survives resampling at 36%
(R1), 74% (R2) and 61% (R3)** of its spliced size — so part of the spliced
figure's straightening is the shared out-of-distribution suffix, most of it for
R1 and a minority of it for R2 and R3. Had panel (a) been built once and reused,
that decomposition would have been invisible and the spliced number would have
been read as the effect.

### What it shows

**A single illegal move measurably straightens the whole population's layer-5
trajectory.** Raw layer 5, spliced tree, largest deviation from the legal class:

| | deviation | at *t* | 95% CI half-width |
|---|---|---|---|
| R1 placement | **−0.74°** | 36 | 0.18 |
| R2 bracketing | **−1.31°** | 22 | 0.20 |
| R3 turn | **−1.05°** | 21 | 0.20 |

4–7 CI widths, so not noise — and also *not visible*, because panel (b) forces
the shared y-axis to span thirty degrees. That is why `make_figures` now prints
the number: a sub-degree gap on a thirty-degree axis is something to read, not
something to measure off the panel.

Three things worth carrying:

- **The ordering R2 > R3 > R1 holds in all nine representation spaces, and on
  both trees** — 18 of 18. The bracketing rule, the one that needs the most board
  computation to check, moves the trajectory most; the occupied-square rule moves
  it least. CosineSpaDE is the extreme (R2 −3.74° at *t* = 21), GatedSAE the
  mildest (R2 −0.57°).
- **The effect is transient.** R2 against legal, averaged in position bands
  (raw L5): **−0.02** over positions 2–9, **−0.76** over 10–20, **−1.20** over
  21–34, **−0.74** over 35–45, **−0.03** over 46–57. It builds as games fork,
  peaks just after the last *t0*, and is gone by the endgame.
- **Its size is the same order as swapping the corpus.** The championship class
  sits +1.06° above legal over positions 10–20 and −0.98° below over 46–57.

### The panel's own sanity check

*t0* is round-robin over [10, 34], so left of position 10 no branch exists yet
and the rule curves and the legal class are near-identical game sets measured by
two pipelines. They agree to **≤ 0.25° in every space**, printed as `pre-fork
|Δ| <= 0.25`. That residual is a fifth of the effect, and it bounds how much of
the gap could be pipeline rather than violation.

## Figure decisions that needed a call

**The blue collision is real and is resolved with line style, not by dropping a
series.** `CLASS_STYLE["rollout"]` and `RULE_STYLE["R1 placement"]` are both
`#0072B2` — the module docstring used to cite that as one of two reasons (a) and
(b) were separate axes. Panel (a) now has six series, which is where
FIGURE_GUIDELINES §1 says line style becomes a required third channel, so the
population rule curves take a long dash `(0, (6, 1.6))` and keep their rule
colour and marker. Every pair on that axes now differs in at least two channels
(rollout is blue *solid* *diamond*, R1 is blue *dashed* *square*). The dash is
also not decorative: it separates whole games from single branched moves inside
games. The alternative — dropping `rollout` from `CLASSES` — would have solved
the colour clash by deleting data, which is worse.

The other reason for two axes stands and is why (a) and (b) were not merged:
(a) is a population and (b) is one game.

Six series is one over §4.1's ≤5 guidance (which is written for layer sweeps, but
the spirit carries). It was checked rather than waved through: converted to
grayscale, the solid/dashed split reads as two groups at a glance and the six
markers separate the series inside each group. That is the case for the dash
doing real work rather than decorating.

**Panel (a) lost its own in-panel legend.** Six series with per-rule n do not fit
in `loc="lower center", ncol=3`. Its entries were already being collected into
the figure-wide legend in the board band (`apparatus` reads handles from all
three panels), so the in-panel copy was a duplicate; per FIGURE_GUIDELINES §5 the
one legend now lives in the band. The band legend runs 11 entries at ncol=2 and
fits.

**The shaded band and its label are new furniture and are load-bearing.** The
rule curves have no single fork position. Without the band a reader reads the
dashed *t0* line — which belongs to game 1235 and to panel (b) — as the rules'
fork too. §4.2's rule for an event with duration is to shade the window and state
it; the band is drawn from `Population.t0_range`, read out of the data rather
than hardcoded.

## Cost and mechanics

2000 games × ~9 branches × K rollouts × 9 spaces, per tree. Streamed a chunk of
games at a time (`ROW_BATCH = 128` sequences of codes, ~245 MB) with only the
`[n_game, n_rule, n_turn]` table of per-game means surviving a chunk, so the
rollout tree's 4 GB activation file is never resident. Cached to
`$DIR/[rollout/]rule_population/<space>.npz`, skip-if-exists; re-rendering reads
a 4 kB npz.

This login node gives one core, where the sweep is ~3 h and would sit near the
8 GB cap, so it runs as `submit_rule_population.sbatch` on `kempner_h100`.
Job 37604348, both trees, 18 space-builds: `COMPLETED 0:0`, elapsed **33:58**,
**MaxRSS 2.0 GB**. No OOM, and the exit code was read rather than assumed. That
sbatch exports
`PYTHONPATH="$SPADE_REPO/src"` — the conda env's editable install points at
`/n/home10/dcao/spade_sae`, so without it a worktree job silently runs the main
clone's code.

One worktree wrinkle worth recording: `checkpoints/**/ae.pt` is gitignored, so a
fresh worktree has the SAE `config.json` files and none of the weights, and
`seed42_entries()` asserts on the first one. Symlinked to
`/n/home10/dcao/spade_sae/checkpoints/`; they stay (gitignored, and needed by
anything in this tree that touches an SAE space).

`load_fork`'s activation-to-θ body is now `_theta`, shared with the sweep. The
fork numbers the README quotes are unchanged to the last printed digit
(`spread 0.906`, `R3 −19.11@−1`), which is the check that the refactor moved code
and not behaviour.

## Not done

- `PROCESS.md` untouched, as instructed.
- `src/spade/figures/straightness_panels.py` and `branch_style.py` untouched;
  `ci_bars`, `MEMBER` and `legend_proxy` are imported or left alone, never edited.
- The 1550-game branch-complete numbers elsewhere in this study
  (`game_selection`, `rule_groups`, the spread ratios) are on a different game set
  and were not touched. Only panel (a) uses the 2000/2000/1997 sets.
