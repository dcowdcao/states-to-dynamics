# Finish note — the population panel at n = 2000, all forking at *t0* = 20

Branch `data/t0-20-population`. The per-rule population panel is restricted to one
fork position so that *t0* is not confounded with the violation, and the round-robin
over [10, 34] left **80** games there. **1920 further source games, every one forked
at *t0* = 20**, were drawn from the same stream at the same seed and **appended**.
The tree now holds **3923 games / 34546 branch rows**, 3920 rule-complete, and
**n = 2000 at *t0* = 20 for each of R1, R2 and R3**. Every pre-existing row in both
trees is byte-identical.

## What the population is now

`rule_population` prints its own n. Spliced tree, raw layer 5:

```
raw layer 5 (reference space)  R1 141.73deg n=2000 members=2.00
                               R2 141.32deg n=2000 members=2.85
                               R3 141.41deg n=2000 members=1.00
```

All three rules share n = 2000 because every game in the selection is rule-complete
*at t0 = 20*; `members` still varies by rule, R2 averaging under 2.85 of its three
member categories per game on average. Every one of the 9 spaces on both trees prints
`n=2000` for all three rules — the rollout tree's raw layer 5 reads R1 142.05°,
R2 141.45°, R3 141.61°.

Tree totals, both trees: 3923 games, 34546 spliced branch rows (30623 non-reference),
138184 rollout rows (122492 non-reference), window −8..24, 17 spaces each.

Branch counts over all 3923 games: reference / `control_legal` / `control_legal_2` /
`occupied_own` / `occupied_opp` 3923 each, `empty_detached` 3874, `empty_own_only`
3299, `empty_no_bracket` 3838, `opponent_only` 3920.

## Qualification rate at *t0* = 20

A game must admit all three rules **at t0 = 20**, which is a different test from
admitting them at its own round-robin *t0*. It was measured, not assumed:

| | count |
|---|---|
| source games consumed from the stream | 1924 |
| rejected: fewer than 3 legal moves at *t0* = 20 | 3 |
| games that reached the rule test | 1921 |
| rejected: branches do not cover R1, R2 and R3 | **1** |
| accepted and appended | **1920** |

**1920 / 1921 = 99.95%.** Against 3 failures in the round-robin's 2000 games, at
*t0* 17, 28 and 29. R3's only member `opponent_only` is a square that flips for the
opponent and not for the mover, and at move 20 there is essentially always one; the
positions that fail are early and late boards, not the middle. So the fixed-*t0*
population cost almost nothing in draws — 1924 games consumed for 1920 kept.

The cumulative `skips` counter is now `{fewer_than_3_legal: 4, forfeit_at_t: 1,
rule_incomplete: 1}`, up from `{fewer_than_3_legal: 1, forfeit_at_t: 1}`.

## Byte-identity of the existing rows

The binding constraint, unchanged from the last extension: `game_id` is the
accepted-game index, so a regeneration under any changed acceptance rule renumbers
every game and moves `FIGURE_GAME = 1235`, the `branches.pt` ↔ `acts/*.pt` row
alignment, the rollout tree's `row = source_row * K + k`, and every score in
`game_selection/`.

Digests were taken **independently before the run** (a separate read of the two
tensors on the login node) and compared to what the jobs asserted afterwards:

| tensor | quantity | before the run | asserted by the job |
|---|---|---|---|
| `branches/branches.pt` | file sha256 | `45660bd563e4a59b527f77fd024d07b7f4b1ddd97e1dc46d9225361b387477c9` | same, printed at start |
| `branches/branches.pt` | sha256 over all 17554 pre-existing rows, every column | `f75945b3da7874816a7035e65a0aae25037b6230ecfa6981d5787a7cf0fea432` | **same** |
| `rollout/branches/branches.pt` | sha256 over all 70216 pre-existing rows | `641618c93674b57f65a6b5c9c8e4779ac2bedebcdc7db5a72eb134b7aa00eb3d` | **same** |

The row digest is the one that answers the question — the whole-file hash necessarily
changes when rows are appended. It hashes `column[:n_rows].tobytes()` for every column
in order and is asserted three times per tree: on the in-memory concatenation, and on
the tensor re-loaded from disk after `torch.save`.

Two further checks, both of which had to pass before anything was written:

- The replay of all 2003 games reproduced **every saved column** under `torch.equal`
  (8 columns spliced), and its `skips` counter equalled the manifest's.
- `extend_rollout_branches` asserts the rollout tree is the head of `branches.pt`
  before it samples, and `verify` re-derives the whole contract afterwards:
  `verify ok: 34546 branches x 4 rollouts, 4.000 distinct suffixes per branch
  (0 all-identical), 0 references replayed`.

Re-checked once more after every job had finished, reading the tensors fresh:
`f75945b3…` over the first 17554 spliced rows and `641618c9…` over the first 70216
rollout rows, both still equal to the pre-run baselines; `FIGURE_GAME = 1235` still
forks at *t0* = 20; `acts/resid_post_5.pt` has 34546 rows against 34546 branch rows and
`rollout/acts/resid_post_5.pt` 138184 against 138184.

Before touching the real tree, the same code was smoked end-to-end on a 40-game tree
in a scratch `$SPADE_DIVERGENCE_DIR`: one append at *t0* = 20, then a **second** append
on top of it. The second is the one that matters — it replays a tree that is already
a round-robin block plus a fixed-*t0* block — and the first 350 rows' digest was
identical across all three states.

## What had to change in `extend_branches`

`draw(games, rng, t0, skips)` already took *t0*; the caller applied `t0_for(game_id)`.
Three things followed from fixing it at 20 instead:

- **`verify()` does not assert the round-robin.** It reads *t0* per game and only
  checks that a game's rows agree on it. Confirmed by reading, then by the run.
- **`replay()` did.** It called `t0_for(game_id)`, which cannot replay a tree whose
  later games fork at a fixed position. It now reads each game's *t0* out of the saved
  tensor and applies the rule filter from the first appended `game_id` onward — the
  test each game was actually accepted under. That boundary is why `manifest.json`'s
  `appended` became a **list of blocks** in both manifests rather than a single dict
  that each run overwrote; `appended[0]["first_game_id"]` is where filtering starts.
  Without this a third extension would silently accept a different set of games.
- **`--target` counts the population, not the tree.** `--t0` is required and
  `--target` is the number of rule-complete games wanted *at that position*.

**`t0_range` is gone from the manifest.** It was `[T0_MIN, T0_MAX]` — two module
constants, never measured, and now a description that reads as true and is not. The
`t0_histogram` is measured from the reference rows and says what the tree actually
holds: 81 games at 10, 11 and 12, 80 at every other offset, **2000 at 20**. The
`appended` blocks record each block's *t0*.

A one-time migration rewrote both on-disk manifests into the new shape (`appended` to
a list, `t0_range` dropped); the originals are beside them as `manifest.json.pre-t0-20`.

## The pipeline, and what each stage cost

Account `kempner_ba_lab`, partition `kempner_h100`, one H100 per task.

| stage | job | wall clock | peak RSS |
|---|---|---|---|
| `extend_branches --t0 20` + `extract_acts` | 37630389 | **4:23** | 19.6 GB |
| `extend_rollout_branches` + `extract_acts --variant rollout` | 37630526 | **10:03** | 86.5 GB |
| `compute_divergence`, spliced, 17 spaces / 6 tasks | 37630772 | **2:30** (slowest task; 0:59–2:30) | 3.3 GB |
| `compute_divergence --variant rollout`, 17 spaces / 6 tasks | 37631121 | **5:35** (slowest task; 1:29–5:35) | 15.1 GB |
| `rule_curvature --t0 20`, spliced | 37631122_0 | **7:40** | 2.1 GB |
| `rule_curvature --t0 20`, rollout | 37631122_1 | **25:47** | 1.8 GB |

The append itself is the cheap part: replaying 2003 games and drawing 1924 more took
under two minutes of the first job. What costs is `extract_acts`, which has no append
mode and rewrites nine `[n_rows, 59, 512]` fp16 tensors whole.

**The `--mem=96G` in `submit_extend_rollout_branches.sbatch` would not have survived
this.** `extract_acts` holds all ten output tensors in RAM before saving; at 138184
rows that is ~76 GB and the job peaked at 86.5 GB. Raised to 250G. The spliced job was
raised to 192G and used 19.6 GB.

### Disk

`/n/netscratch/ba_lab/Lab/dcao/spade_divergence`: **48 GiB → 93 GiB** (51.1 GB → 99.0 GB apparent). Essentially
all of it is `acts/`:

| | before | after |
|---|---|---|
| `acts/` (34546 rows) | 9.4 GB | 18.6 GB |
| `rollout/acts/` (138184 rows) | 38 GB | 75.6 GB |
| `divergence/` + `rollout/divergence/` (34 npz) | ~1.1 GB | 2.2 GB |
| branch tensors and everything else | ~0.5 GB | ~0.6 GB |

Both trees roughly doubled, as expected — the tree grew from 2003 to 3923 games. The
+45 GiB is 0.09% of the 50 TiB netscratch allocation.

## Caches: what was rebuilt and what is now stale

`compute_divergence` and `rule_curvature` are both **skip-if-exists at whole-file
granularity with no per-row path**, so a rerun over an extended tree recomputes
nothing and silently keeps values computed over the smaller tree. Both output sets
were therefore deleted before their jobs were submitted:

- `$DIR/divergence/*.npz` and `$DIR/rollout/divergence/*.npz` — 17 each, all rebuilt.
  The logs show `wrote <space>.npz` for every space and no `skip` lines.
- `$DIR/rule_population/*.npz` and `$DIR/rollout/rule_population/*.npz` — 9 each,
  all rebuilt at `--t0 20`.

**Still holding 2003-game content and not aware the tree grew:**

- `$DIR[/rollout]/rule_descriptors/` — the per-game 21-D ReStraV descriptors the
  straightness `feature_pca` figure draws the rules from. They were deliberately left:
  `rule_curvature.main` only rebuilds the products it is missing, so leaving them made
  the population pass read 2000 games instead of 3920. The file is a valid subset —
  `game_id` is stable, so its rows still mean what they meant — it simply does not
  include the 1920 new games. Delete it if the descriptors should cover them.
- `$DIR[/rollout]/divergence_by_space/curves.npz` — derived from the divergence npz by
  `analysis/violation_divergence/divergence_by_space/measure.py`, which was not re-run.
- `$DIR/curvature_discriminant/`, `fork_curvature/`, `fork_expanding/`, `fork_r21/`,
  `theta_comparability/` — keyed off `complete_games` or a fixed game, so their content
  stays correct; they just do not include the new games.

Prose quoting a population size is stale the same way. `analysis/violation_divergence/
README.md` was rewritten here; `analysis/violation_divergence/rule_curvature/` was left
alone by instruction, and its README still says n = 80 in several places.

## Two things that bit

**The worktree has no SAE weights.** `checkpoints/*/trainer_*/ae.pt` is gitignored, so
a fresh worktree carries `config.json` and nothing else, and `saes.registry` verifies
`ae.pt` on construction. All six tasks of the first divergence array died in 19
seconds on `AssertionError: missing ae.pt`. Fixed by symlinking the 11 checkpoint files
from `/n/home10/dcao/spade_sae` into the worktree (they are gitignored, so nothing
enters the tree). Anything that touches an SAE from a worktree needs this.

**`submit_violation_divergence_spaces.sbatch` had no `PYTHONPATH` line.** Its siblings
export `${SPADE_REPO}/src` because `pip install -e` pinned `spade` to whichever
checkout ran `make setup`; this one did not, so a worktree run silently imported the
main clone's code. Added.

## Sampling caveat, unchanged

`stream_games` shuffles a 10,000-row reservoir over one shard, so drawing N games reads
the first N + 10,000 rows and no further (`docs/game_corpus.md`). The append moves the
seed-42 window from ~12,000 rows to ~14,400 of the corpus's 792,498 — from 1.6% to
1.8%. The population is 25× larger at *t0* = 20 but it is not a broader sample of the
corpus, and it is not independent of any other study defaulting to seed 42.

## Files

| file | what |
|---|---|
| `src/spade/violation_divergence/extend_branches.py` | `--t0` required, `--target` counts that position; `replay` reads per-game *t0* and the filter boundary; `appended` blocks |
| `src/spade/violation_divergence/branches.py` | `summary` drops `t0_range`; the manifest starts with an empty `appended` list |
| `src/spade/violation_divergence/extend_rollout_branches.py` | appends its block to a list |
| `src/spade/violation_divergence/rollout_branches.py` | writes `appended: []` |
| `src/spade/violation_divergence/rule_curvature.py` | docstrings: the population is one *t0* at n = 2000, not ~80 of 2000 |
| `scripts/slurm/studies/violation_divergence/submit_extend_branches.sbatch` | `--t0 20`, 192G / 8 h |
| `scripts/slurm/studies/violation_divergence/submit_extend_rollout_branches.sbatch` | 250G / 12 h |
| `scripts/slurm/studies/violation_divergence/submit_violation_divergence_spaces.sbatch` | `PYTHONPATH`, 96G / 2 h |
| `scripts/slurm/studies/violation_divergence/submit_rule_population.sbatch` | 6 h, comment corrected |
| `analysis/violation_divergence/README.md` | the section rewritten for the *t0* = 20 population |

`PROCESS.md` was deliberately not touched. Nothing was pushed.
