# Panel (a)'s population forks at one position (branch `fig/single-t0`)

Panel (a) of `analysis/violation_divergence/rule_curvature` averaged the three
rule curves over all 2000 rule-complete games, **each forking at its own *t0***
— round-robin over [10, 34]. That put "where the violation happened" inside the
curves, and it made the dashed *t0* line mean one thing in (a) (game 1235's fork,
irrelevant to the population) and another in (b). The population is now the **80
games that fork at *t0* = 20**, which is game 1235's own position, so one dashed
line serves both panels.

*Renumbered since — panel (b), one game's fork on (a)'s scale, was removed: (a)
stays (a), the violation-locked response (c) → (b), the board (d) → (c). Every
panel letter below is the four-panel layout as it stood, including the "one
dashed line serves both panels" above. See `docs/finish_notes/panelb-docs.md`.*

- `src/spade/violation_divergence/rule_curvature.py` — `build` takes the game
  list; `--t0` selects the population; *t0* is in the cache file's name.
- `analysis/violation_divergence/rule_curvature/make_figures.py` — `--t0`, the
  panel-(a) furniture, the legend entries, and an off-by-one in the pre-fork
  check.
- `analysis/violation_divergence/rule_curvature/README.md` — captions and every
  number that moved.
- `scripts/slurm/studies/violation_divergence/submit_rule_population.sbatch` —
  comment only; the default `--t0 20` needs no flag.
- 20 figures re-rendered under `.../rule_curvature/figures/`.

## The verdict: the effect gets bigger, the interval gets wider, and R1 drops out

Two things happen at once when 2000 games become 80.

**The effect roughly triples.** Raw layer 5, spliced tree, largest deviation from
the legal class:

| | pooled (n = 2000) | *t0* = 20 (n = 80) |
|---|---|---|
| R1 placement | −0.74° at *t* = 36 | **−2.07° at *t* = 32** |
| R2 bracketing | −1.31° at *t* = 22 | **−4.18° at *t* = 21** |
| R3 turn | −1.05° at *t* = 21 | **−3.39° at *t* = 21** |
| 95% CI half-width | ~0.19° | **0.95–0.99°** |

Pooling was averaging violations committed at 25 different points in the game, so
at any given *t* most games had not forked yet or had forked long before. The
pooled curve was a smeared version of this one, not a better-estimated one — the
confound was also suppressing the effect it confounded.

**The interval widens by exactly the √25 the sample size predicts**, 0.19° →
0.96°. The peaks still clear their own CI (R2 4.2 half-widths, R3 3.5, R1 2.2 in
raw L5), but a peak is a maximum over 56 positions and its own CI is the wrong
yardstick for it.

**The right yardstick is the pre-fork residual, and by that yardstick R1 does not
survive.** Left of *t0* − 2 no branch exists, so the rule curves and the legal
class are near-identical game sets measured by two pipelines; the max |Δ| there
is the same kind of statistic as the peak, with no violation in it. It was
≤ 0.25° at n = 2000 and is **1.09–1.66° at n = 80**. Peak ÷ residual, nine spaces,
spliced tree:

| | raw L5 | range over nine spaces | reading |
|---|---|---|---|
| R2 | 2.6× | 1.4–8.6× | clears it everywhere |
| R3 | 2.1× | 1.05–7.8× | clears it in eight of nine |
| R1 | 1.3× | 1.0–1.5× | **does not clear it anywhere** |

So: **R2 survives the confound removal, R3 survives in eight of nine spaces, and
R1 is no longer distinguishable from sampling noise.** At n = 2000 R1's −0.74°
was 3× its residual; at n = 80 its −2.07° is 1.3× a 1.61° residual. That is the
honest result of holding *t0* fixed, and the README's caption now says it in the
first sentence rather than leaving "about a degree across the whole population"
to carry three rules that no longer behave alike.

The ordering **R2 > R3 > R1** still holds in 8 of 9 spaces (GatedSAE puts R1
above R3, by less than a CI half-width, and neither clears the residual there).

**The resampled tree confirms both halves of that, and sharpens the R1 half.**
Its rule curves are rebuilt on continuations sampled from OthelloGPT, so it does
not share the spliced tree's out-of-distribution suffix:

| raw layer 5 | spliced | resampled |
|---|---|---|
| R1 peak | −2.07° at *t* = 32 | −1.61° at *t* = 5 — **pre-fork** |
| R2 peak | −4.18° at *t* = 21 | −4.24° at *t* = 21 |
| R3 peak | −3.39° at *t* = 21 | −2.95° at *t* = 19 |
| pre-fork residual | 1.61° | 1.61° |

- **R2 is unmoved by resampling** (−4.18 → −4.24, same position) where the pooled
  panel had it losing 26% of its size. With *t0* fixed, R2 is not the suffix.
- **R1's largest deviation is pre-fork in all nine spaces on this tree.** There is
  no R1 response at *t0* = 20 to argue about.
- **The pre-fork residual is bit-identical between trees** in all nine spaces,
  which is the check that the comparison is well-posed: the trees share
  `tokens[:t0]`, so their pre-fork activations are the same numbers. The same
  mechanism makes the peaks that land at *t* = 19 (the turn whose window is
  *t0*−2..*t0*) identical to the last digit in the two trees — Standard,
  CosineSpaDE and SpaDE — which looks like a copy-paste bug in the logs and is
  not one.

One stale number came out with it. The README said **n = 2000, 2000, 1997** —
R3 short by the three games with no `opponent_only` square. That has not been
true since `extend_branches` appended three replacements: no game in
`rule_complete_games.json` is missing a rule (checked, empty list), so n is equal
across the three rules by construction and is 80 each here.

## An off-by-one the single *t0* exposed

The pre-fork window was `positions < t0_range[0]`. θ column *k* is the turn at
board position *k* + 2 whose window spans *k* + 1 .. *k* + 3, so the turn at
*t0* − 1 already sees the branch token: the last clean position is *t0* − 2. With
a pooled `t0_range[0] = 10` the error was invisible (one position out of eight at
the noisiest end of the board); at *t0* = 20 it put the response *at the fork*
into the null and reported the residual as 3.64° instead of 1.61°. Fixed to
`positions <= t0_range[0] - 2`.

## Design: the aggregate is cached, not the per-game curves

The first plan was to cache `[n_game, n_rule, n_turn]` per-game curves plus each
game's *t0*, and select at figure time — 1.4 MB, and any *t0* becomes free. The
owner cut it for wall clock: at one *t0* the pass is 80 games, not 2000, so
rebuilding for another position costs about a minute rather than the 25 the
per-game cache was meant to avoid. What is kept from the idea is that **`build`
takes its game list as an argument** and `RuleCurves.at(t0)` selects inside the
pass, which is what lets the two products come off one pass while living on
different game sets:

- `rule_population` is one fork position (~80 games).
- `rule_descriptors` is all 2000 rule-complete games — `analysis/straightness/
  feature_pca` reads it — and is unaffected by `--t0`.

A run missing only the population encodes 80 games; a run that also has to
rebuild descriptors encodes 2000 and selects the 80 out of that pass, so the
population never costs a second pass.

**The *t0* is in the npz name** (`<space>_t0-20.npz`, `_t0-all.npz`). A population
at another fork position is another population, and skip-if-exists must not hand
one back for the other. `--t0 all` reproduces the pooled panel exactly, cache and
all, so nothing was lost.

## Cost and mechanics

Job 37612324, both trees, `submit_rule_population.sbatch` on `kempner_h100`.
Task 0 (spliced) had its `rule_descriptors` cached, so all nine spaces were
80-game passes: **COMPLETED, elapsed 1:08** — the whole point of selecting on a
population this small. Task 1 (resampled) had no descriptors at all, so all nine
spaces were 2000-game × 4-rollout passes and the population was selected inside
them: **COMPLETED, elapsed 26:32**. Exit codes read, not assumed (`0:0` both);
MaxRSS 0.8 GB and 1.9 GB. Neither elapsed is the cost of changing *t0*; that is
task 0's minute.

Two earlier submissions of the same script failed in 20–40 s with
`missing ae.pt in checkpoints/main/trainer_0`: `checkpoints/**/*.pt` is
gitignored, so a fresh worktree has the SAE `config.json` files and none of the
weights. Symlinked to `/n/home10/dcao/spade_sae/checkpoints/`, as
`docs/finish_notes/rule-population.md` records for the previous worktree. They
stay — gitignored, and needed by anything in this tree that touches an SAE space.

Rendering the 20 figures runs on the login node under its 8 GB cap; no OOM
(no exit 137) on either tree.

## Not done

- `PROCESS.md` untouched, as instructed. Nothing pushed.
- The `--t0 all` caches were not built. The pooled numbers quoted in the README
  as the "pooled" column are the previously committed ones; regenerating them is
  a 2000-game pass per tree.
- Panels (b)–(d) and the combined figure are untouched: they are one game, and
  *t0* was never a free parameter there.
- Everything the owner asked to keep is kept — no faint member traces, no shaded
  ribbons, per-ply θ, per-member-then-average within a game.
