# violation_divergence, wave 1: the branch set and its activations

What `branches.py` and `extract_acts.py` produced, and what the integrity checks
found. Artifacts live on `$SPADE_DIVERGENCE_DIR`
(`/n/netscratch/ba_lab/Lab/$USER/spade_divergence`), shared across worktrees.

## The branch set

2000 source games, 15531 branch rows, seed 42, dataset
`adamkarvonen/othello_45MB_games`, `datasets` 4.8.4.

| branch | rows | present in |
|---|---|---|
| `reference` | 2000 | 100.0% of games |
| `control_legal` | 2000 | 100.0% |
| `occupied_own` | 2000 | 100.0% |
| `occupied_opp` | 2000 | 100.0% |
| `empty_detached` | 1956 | 97.8% |
| `empty_own_only` | 1636 | 81.8% |
| `empty_no_bracket` | 1941 | 97.1% |
| `opponent_only` | 1998 | 99.9% |

Branches per game: 8 in 1554 games, 7 in 423, 6 in 23. Minimum 6, so no game is
missing more than two illegal categories.

**t0 histogram: 80 games at each of t0 = 10 … 34.** Exactly uniform — the
round-robin never had to skip enough to unbalance it (one skip in 2000 games).

**Skips: `single_legal` 1.** That is the entire list. No `short_game`, no
`pass_token`, no `forfeit_at_t`. 2001 streamed games were consumed to accept
2000. The corpus games are all ≥ 59 tokens and pass-free over the first 59, and
no mover forfeited at a t0 in [10, 34].

**Board change per legal branch.** `reference` and `control_legal` both flip
**1.93** discs on average (max 8). The design doc quotes 2.28 from
`src/spade/rules/flip_labels.py`; that figure is over t ∈ [5, 57], and this study's
t0 ∈ [10, 34] is an earlier and less flippy stretch of the game. Not a
discrepancy, a different window — but the doc's number should not be quoted for
this study's control.

**Legal moves at t0:** mean 10.95, min 2, max 22.

## Two design-doc claims about category availability, checked

> *"Not every category exists at every prefix (`empty_detached` disappears late;
> `opponent_only` is often absent)."*

- `empty_detached` disappearing late is **confirmed** and sharp: absent in 44
  games, all at t0 ≥ 26, rising monotonically to 15/80 games at t0 = 34.
- `opponent_only` is **not** often absent — it is missing in 2 games out of 2000
  (t0 = 15 and t0 = 17). At t0 ∈ [10, 34] it is effectively always available.
- The scarce category is **`empty_own_only`**, absent in 364/2000 games (18.2%),
  spread across the whole t0 range (8/80 at t0 = 10 up to 29/80 at t0 = 34).

So the "report per-category n, never pool without saying so" instruction still
holds, but the category it bites on is `empty_own_only`, not `opponent_only`.

## Activations

`$DIR/acts/`, rows aligned with `branches.pt`, all fp16:

| file | shape | size |
|---|---|---|
| `resid_post_0..7.pt` | [15531, 59, 512] | 938 MB each |
| `token_embed.pt` | [15531, 59, 512] | 938 MB |
| `logits.pt` | [15531, 59, 61] | 112 MB |

**8.0 GB** total for the full set, **203 MB** for the smoke set
(`$DIR/smoke/acts/`, 386 rows from 50 games). Branch tensors and manifests are
2.4 MB and 66 KB.

## Mean model probability of the branch token

Read off `logits[:, t0 - 1]`, softmaxed, gathered at the branch token.

| branch | n | mean p |
|---|---|---|
| `reference` | 2000 | 1.00e-01 |
| `control_legal` | 2000 | 1.00e-01 |
| `occupied_own` | 2000 | 5.50e-06 |
| `occupied_opp` | 2000 | 5.31e-06 |
| `empty_detached` | 1956 | 4.78e-06 |
| `empty_own_only` | 1636 | 4.83e-06 |
| `empty_no_bracket` | 1941 | 1.15e-05 |
| `opponent_only` | 1998 | 1.81e-05 |

The model puts ~10⁻⁵ on every illegal branch and 0.10 on both legal ones — a
four-order-of-magnitude gap, and `reference` and `control_legal` are
indistinguishable, which is the right sanity check on the control: the null
branch is as in-distribution as the real move. `opponent_only` is the most
probable violation and `empty_detached` / `empty_own_only` the least, matching
the taxonomy's computation-depth ordering.

## Integrity checks: what ran, and the one that fired

In `branches.py::verify`, on **all 2000 games** (not a sample — the whole pass
costs 8.9 s), reading only the saved tensors:

- source game replays legally end to end (`first_illegal_pos == -1`)
- every branch is 59 nonzero tokens
- exactly one `reference` row per `game_id`
- all rows of a game agree on `tokens[:t0]` and `tokens[t0+1:]`
- `tokens[t0] == branch_token`, and no two branches of a game play the same token
- `classify(replay(prefix), branch_square) == category`, re-derived from the
  emitted rows rather than from the selection-time taxonomy call
- `n_flipped > 0` on legal branches, `-1` on illegal ones; `n_legal ≥ 2`

Plus `build_pairs.classify_prefix`'s own assert that the taxonomy's legal set
equals `board.get_valid_moves()` at every accepted t0.

In `extract_acts.py`: the layer-5 position-0 sink (**300.7 vs 23.5**, matching
the design doc's ~300 vs ~22), prefix invariance on **all 2000 games** rather
than a sample, the 59-nonzero-token claim, and the p(branch token) assert above.

**The one that fired: prefix invariance, job 35957113.** Layer-5 activations of
game 1977's branches differed before t0 = 12. Cause: rows 15353–15360 of that
game straddled row 15360, the boundary between the last full 256-row batch and
the final 171-row batch. Different batch shape, different cuBLAS reduction
order, so two sequences with an identical prefix stopped agreeing bit for bit.
Only the final partial batch is affected — 49 games straddle a 256 boundary and
48 of them passed, because every other boundary sits between two identically
shaped batches. Fix in `extract_acts.spans()`: batch boundaries are snapped to
game boundaries, giving 62 spans of 122–256 rows. Every game now goes through
one batch shape. This is a real constraint on anyone who re-runs extraction, not
an incidental detail — `spans()` carries the reason in a comment.

Independent read-back of `$DIR/acts/resid_post_5.pt` after the run confirms
`δ̂(r) = ‖z^b − z^ref‖ / ‖z^ref‖` is **exactly 0.0** for every branch at every
position `< t0`, in all 2000 games. At r = 0, layer-5 `δ̂` has min 0.201 and mean
0.789 over branches, so no branch is silently identical to the reference.

## Wall times

| step | where | time |
|---|---|---|
| smoke branches (50 games) | login node, CPU | 24 s |
| smoke extraction (386 rows) | login node, CPU | 1 m 55 s |
| full branches (2000 games) | login node, CPU | 3 m 27 s |
| full extraction (15531 rows) | job **35957610**, 1×H100 | **1 m 49 s**, peak RSS 9.3 GB |

Job 35957113 was the failed first attempt (2 m 49 s, died in the prefix-
invariance check before writing anything, so no partial artifacts).

## For the integrator

1. **`study_dir.py` has only `divergence_dir()`**, per the design doc's interface
   block, so the smoke-path rule
   (`divergence_dir() / "smoke" if args.smoke else divergence_dir()`) is written
   out in each entry point. `branches.py` and `extract_acts.py` each carry one
   copy; `compute_divergence.py` will be the third. If a third copy is one too
   many, the rule belongs in `study_dir.py` — but that changes the published
   interface, so it is the integrator's call, not mine.
2. **`role()` and `ROLE_NAMES` live in `branches.py`** and `extract_acts.py`
   imports them. Every wave-2 agent that reports per-category numbers wants the
   same mapping from `(category, is_control)` to a name; that is where it is.
3. Nothing outside my five files was touched. `submit_violation_divergence.sh`
   needs `SPADE_REPO` exported to a worktree when submitting from one — the
   default is `$HOME/Spade_Research_David_Sumedh_Ba`, following the other scripts
   in `shell/slurm/`.
