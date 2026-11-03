# Finish note — appending rule-complete games to the violation-divergence tree

Branch `data/extend-branches`. The branch set had 2000 source games but only **1997**
that admit all three constraint rules, so a curve drawn per rule read n = 2000, 2000,
1997. Three further games were drawn from the same stream and **appended**; the tree now
holds **2003 games, 2000 of them rule-complete**, on both the spliced and the resampled
trees, with every pre-existing row byte-identical.

## Final per-rule counts

| rule | members | n over all 2003 games | n over the rule-complete 2000 |
|---|---|---|---|
| R1 placement | `occupied_own`, `occupied_opp` | 2003 | **2000** |
| R2 bracketing | `empty_detached`, `empty_own_only`, `empty_no_bracket` | 2003 | **2000** |
| R3 turn | `opponent_only` | 2000 | **2000** |

Branch counts, all 2003 games: reference / `control_legal` / `control_legal_2` /
`occupied_own` / `occupied_opp` 2003 each, `empty_detached` 1954, `empty_own_only`
1633, `empty_no_bracket` 1952, `opponent_only` **2000** (was 1997). 17554 rows over
2003 games spliced; 70216 rows rollout.

Both trees' divergence npz were rebuilt and agree: 2003 games, 15551 spliced branch
rows (17554 − 2003 references), 62204 rollout branch rows (70216 − 8012), window
−8..24, 17 spaces each.

## The gap, and why it was only R3

Every one of the 2000 games offers at least one R1 member (`occupied_own` and
`occupied_opp` are present in all 2000) and at least one R2 member. **R3 is the only rule
that can go missing**, because it has a single member — `opponent_only`, a move that flips
only for the opponent — and three *t0*s offer no such square at all. The three games are
**582** (*t0* = 17), **1268** (*t0* = 28) and **1869** (*t0* = 29); each carries
categories {legal, occupied_own, occupied_opp, empty_detached, empty_no_bracket} and
nothing else.

## How many were drawn, and how many were skipped

**3 drawn, 0 skipped.** The first three games the stream offered after game 1999 all
covered R1, R2 and R3, so `n_rejected_rule_incomplete` is 0 — unsurprising at a 3/2000
base rate (P ≈ 99.6% that three consecutive games all qualify), but it is the number the
manifest records rather than one assumed. The cumulative `skips` counter is unchanged at
`{fewer_than_3_legal: 1, forfeit_at_t: 1}`: the appended draws hit no other filter either.

The appended games are `game_id` 2000, 2001, 2002 and take the *t0* round-robin's next
offsets, **10, 11 and 12**. They were taken in stream order and never searched or scored,
so the appended set is conditioned on rule membership and on no measured outcome. All
three turned out to be *branch*-complete as well (9 rows each, 27 rows total), so
`complete_games` grows from 1550 to 1553 — appended at the end of that list, so every
`complete_games(...)[0]` and `complete[:n]` in the study is unaffected.

## Byte-identity of the existing rows

This was the binding constraint: `game_id` is the accepted-game index, so re-running
`branches.py` under a rule-complete acceptance test renumbers every game and moves
`FIGURE_GAME = 1235`, the `branches.pt` ↔ `acts/*.pt` row alignment, the rollout tree's
`row = source_row * K + k`, and every score in `game_selection/`.

`extend_branches` therefore **replays** the original draw — same seed, same
`stream_games`, same `draw` — for all 2000 existing games before it appends anything, and
asserts the replay equal to the tensor on disk column by column. Four independent checks
passed:

| check | result |
|---|---|
| `branches.pt` sha256 before the run | `d821822d23356b3f8eeff03b9bdea848222c49346a80871ebbd545aa22d12062` |
| replayed columns vs. saved columns, all 8, `torch.equal` | equal |
| replayed `skips` vs. the manifest's `skips` | `{fewer_than_3_legal: 1, forfeit_at_t: 1}` both |
| first 17527 rows, sha256 over all 8 columns' raw bytes, before vs. after the write and after re-loading the written file | `b6ec188bb41006817c4a01191e37c9765a531f665c44ce1d8d8a0786e984d0c7`, unchanged |

The row digest is the one that answers the question, since the whole-file hash necessarily
changes when 27 rows are appended: it hashes `column[:17527].tobytes()` for every column
in order, and it is asserted three times — on the in-memory concatenation, and on the
tensor re-loaded from disk after `torch.save`.

Two cheaper checks were run first, on the 50-game smoke tree, and both are stronger than
they look:

- `branches.py --smoke` **reproduces the committed smoke tensor byte for byte**, a file
  written 2026-07-29. So the refactor that factored the per-game draw out of `main()` is
  behaviour-preserving, and the HF stream is still deterministic at `datasets` 4.8.4.
- A 50-game tree extended by 2 is **byte-identical to a fresh 52-game draw**. The append
  is a true continuation of the stream and the RNG, not a fresh sample that happens to
  land nearby.

## The rollout tree

`rollout_branches.py` regenerates the whole tree. Its uniforms are keyed on
`(game_id, k)` via `Generator(seed * 1_000_003 + game * K + k)`, so **the appended games
draw streams no existing row ever drew from, and no existing row's stream moves** — that
much is guaranteed by the seeding. But `generate` batches by *t0*, and adding rows changes
batch composition, which changes logits in their last bits and so *could* change a sampled
token. Re-running it would therefore leave existing rows only probably fixed.

`extend_rollout_branches` samples the appended source rows **alone** and concatenates, so
no existing row is recomputed at all and the guarantee is exact rather than probable. Its
digest check is the same one as above over all 70108 pre-existing rows. `row = source_row *
K + k` survives because the new source rows are at the end of `branches.pt`.

One consequence worth stating plainly: the appended rollout rows are **not** bit-equal to
what a from-scratch regeneration of the whole tree would have produced for those games,
because they were batched alone rather than alongside every other *t0* = 10/11/12 row.
They are validly sampled from their own `(game_id, k)` streams; there is simply no
"canonical" full-regeneration to compare them against, and preserving the 70108 existing
rows was the requirement that actually mattered.

## What was recomputed and what was skipped

**`extract_acts` has no append mode** and rewrites `$DIR/acts/` and `$DIR/rollout/acts/`
whole, for all rows. That is why the two new sbatch scripts call it unguarded, where
`submit_violation_divergence.sbatch` guards it with `[ -f "$DIR/acts/logits.pt" ]`.
Recomputing the existing rows is harmless — they are recomputed from unchanged tokens —
though the last batch's span shifts, so existing rows are not promised bit-equal at fp16
in the tail. Nothing downstream depends on that; the divergence npz were rebuilt from
these activations anyway.

**`compute_divergence` skips at whole-space granularity, not per row.** The check is

```python
path = out_dir / f"{space}.npz"
if path.exists():
    print(f"skip {space} (exists)")
    continue
```

so a rerun over an extended tree recomputes **nothing** and silently leaves every npz at
2000 games. There is no per-row incremental path — an npz holds `[n_branch, 33]` arrays
for the whole tree. Extending the tree therefore means deleting
`$DIR[/rollout]/divergence/*.npz` and recomputing all 17 spaces per tree, which is what
was done; the array logs show `wrote <space>.npz` for every space and no `skip` lines.

## A concurrent session re-ran the arrays on top of this

Worth knowing, because it looks alarming in a directory listing. Minutes after this
branch's arrays finished writing all 34 npz, both `$DIR/divergence/` and
`$DIR/rollout/divergence/` emptied and began refilling: another session had deleted
them and submitted its own `vd_spliced` / `vd_rollout` arrays (jobs 37607873 /
37607874), while also merging this branch into `/n/home10/dcao/spade_sae`.

No damage, and nothing was done about it. That session reads the same extended
`branches.pt` and the same `acts/`, so it recomputes the same content over the same
2003 games — redundant work, not a conflict. The tensors this branch is responsible
for were re-checked afterwards and both row digests still match
(`b6ec188b…` over the first 17527 spliced rows, `4e96ef82…` over the first 70108
rollout rows). The npz are derived and regenerable by construction; the branch
tensors are the artifact that had to survive, and they did.

The one thing to check: that session's merge landed **before** this note was
committed, so `docs/finish_notes/extend-branches.md` may need a second merge.

## Stale caches this invalidates

Everything else cached off the branch tensors is still at 2000 games and is skip-if-exists
too, so it will *not* notice:

- **`$DIR/rule_population/` and `$DIR/rollout/rule_population/`** — the many-game rule
  curves, written by a sibling's `rule_curvature --tree {branches,rollout}` run that
  completed just before this one. Those are exactly the n = 2000/2000/1997 curves this
  branch exists to fix, and they must be deleted before they can be redrawn at n = 2000.
- `$DIR/curvature_discriminant/`, `$DIR/fork_curvature/`, `$DIR/fork_expanding/`,
  `$DIR/fork_r21/`, `$DIR/theta_comparability/` — all keyed off `complete_games` or a
  fixed game, so their *existing* content stays correct; they simply do not include the
  three new games.

Prose that quotes a population size is now slightly stale in the same way: "1550
branch-complete games" is 1553, and `rule_groups`' docstring medians were measured over
1550. None of those numbers move materially and none were rewritten here — the measurements
behind them were not re-run.

## Where the rule-complete ids live

`$DIR/branches/rule_complete_games.json`, beside the tensor it describes:

```json
{"criterion": "...", "rule_members": {...}, "n_games": 2000, "game_ids": [0, 1, ...]}
```

A per-rule figure should read that file rather than rebuild the membership test.
`game_id` is shared by both trees, so the rollout tree reads the same file. The derivation
lives in `branches.rule_complete_games(meta)` for whoever needs to regenerate it, but the
point of writing it out is that a figure should not have to.

The rule-complete population is not quite flat in *t0*: **81** games at 10, 11 and 12
against **79** at 17, 28 and 29 (the dropped games' offsets) and 80 elsewhere. Assigning
the appended games the dropped games' *t0*s would have restored the balance, but that
would be choosing *t0* on the basis of which games failed, and the round-robin
continuation is the reproducible choice. The imbalance is ±1.3% on three of 25 offsets.

## Files

| file | what |
|---|---|
| `src/spade/violation_divergence/branches.py` | per-game draw factored out of `main()` (`draw`, `t0_for`, `summary`); `rules_present` / `rule_complete_games` added; docstring warns against re-running over an existing tree |
| `src/spade/violation_divergence/extend_branches.py` | replay, verify, append, rewrite manifest, emit the id list |
| `src/spade/violation_divergence/extend_rollout_branches.py` | the same append on the resampled tree |
| `scripts/slurm/studies/violation_divergence/submit_extend_branches.sbatch` | append + re-extract acts |
| `scripts/slurm/studies/violation_divergence/submit_extend_rollout_branches.sbatch` | the same, rollout |
| `analysis/violation_divergence/README.md` | new section: the rule-complete 2000 |

`PROCESS.md` was deliberately not touched.
