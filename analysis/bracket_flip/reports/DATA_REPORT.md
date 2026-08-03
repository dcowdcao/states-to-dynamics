# bracket_flip data: mining yield, integrity, extraction

What `events.py` and `extract_acts.py` produced, from SLURM job **35950961**
(kempner_h100, COMPLETED, elapsed **00:02:46**, MaxRSS 1.0 GB) — mining and both
extractions in one job. Seed 42, corpus `adamkarvonen/othello_45MB_games`
streamed through `rules.build_pairs.stream_games`.

Reproduce: `sbatch --export=ALL,SPADE_REPO=<repo> shell/slurm/submit_bracket_flip.sh`.

## Event yield

2000 games scanned, `t ∈ [5, 57]` → 106,000 positions examined.

| | count | per 1000 games |
|---|---|---|
| positions examined | 106,000 | 53,000 |
| single-flip candidates | 43,301 | **21,650** |
| events emitted | 2,000 | 1,000 |

One event per game: candidates are shuffled within a game and the first with a
usable control move is kept, so rows are independent across games. Every one of
the 2000 games yielded an event.

## t distribution

Median t = 25. Full per-t histogram in `events/manifest.json`.

| t | 5–9 | 10–14 | 15–19 | 20–24 | 25–29 | 30–34 | 35–39 | 40–44 | 45–49 | 50–54 | 55–57 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n | 297 | 259 | 210 | 224 | 204 | 183 | 189 | 160 | 127 | 104 | 43 |

Single-flip moves thin out with t (50–70 per t below t=15, 8 at t=57) because
late boards are dense and most moves flip several discs. No t was empty.

## Control moves

| kind | n | share |
|---|---|---|
| strict — placement and every disc it flips off s's eight rays | 1,596 | 79.8% |
| loose — flip set excludes s, but something lies on an s-ray | 404 | 20.2% |
| absent | 0 | 0% |

A control also has to leave the opponent a regular move, so the parity increment
cancels in `act(P+m) − act(P+m')`. 21 candidates were dropped for having no such
control; the next candidate in the same game supplied one every time.

## Rejections

Per-position filters over the 106,000 positions scanned:

| check | rejected |
|---|---|
| `\|flip_set\| != 1` | 62,445 |
| nominal mover forfeits at t | 198 |
| opponent forfeits at t+1 | 56 |
| played move not regular for the nominal mover | 0 |
| game shorter than 59 tokens | 0 |
| pass token in game | 0 |
| **kept as candidates** | **43,301** |

Plus 21 candidates rejected for having no usable control move (counted per
candidate tried, not per position).

Forfeit-at-t+1 is dropped on purpose: it leaves the same player to move again,
which inverts the design's "s reads *theirs* before and after" prediction. It is
the only filter beyond the design's stated criteria.

## Integrity checks

These are assertions, not filters — the run crashes rather than dropping a row —
and they fired on **0 of 2000** emitted events. Each is re-derived with
`rules.causal.boards.replay` from scratch, independent of the incremental board
walk that mined the event.

| check | rejected |
|---|---|
| `board_before == replay(P)` | 0 |
| `board_after == replay(P + m)` | 0 |
| before and after differ in exactly 2 squares | 0 |
| m's square empty before, mover's colour after | 0 |
| s holds the opponent's colour before | 0 |
| s holds the mover's colour after | 0 |
| `\|flip_set\| == 1` | 0 |
| `t >= 5` | 0 |

Position 0 is never read: `before` sits at position t−1 ≥ 4.

At extraction, `before` (position t−1 of `run(P+m)`) was checked against a
separate forward pass over `P` alone on 8 events: max abs difference
**6.44e-06** across all 8 layers, i.e. float32 kernel noise from the differing
batch shape.

## Alignment check: the committed linear probe on the extracted activations

Decoding `before[:, 5]` and `after[:, 5]` with
`analysis/linear_probe/activations/linear_probe.pt` (layer 5, {empty, theirs,
mine}) reproduces the design doc's table over all 2000 events. This is a check
that rows, positions and square codes line up — **not** the study's measurement,
which is A3's TPR read-out.

| square | prediction | observed |
|---|---|---|
| s | theirs before | 0.9950 |
| s | theirs after | 0.9950 |
| s | class unchanged | 0.9900 |
| m | empty before | 1.0000 |
| m | theirs after | 0.9890 |
| unflipped opponent disc | theirs → mine | 0.9683 |
| unflipped own disc | mine → theirs | 0.9643 |
| empty square | empty in both | 0.9983 |

Unflipped discs swap class in 0.9757 of cases; s swaps in 0.0100. The flip is
the absence of a swap, as designed.

## `t % 2` vs engine replay

The player to move comes from engine replay everywhere. Measured on the 20,000
probe positions:

- `t % 2` names the **wrong** player at **70 / 20,000 positions (0.35%)**, spread
  over **12 / 400 games (3.0%)**.

Corpus context (separate 500-game scan): 36.0% of games contain at least one
forfeit, 0.47 forfeits per game — but forfeits concentrate at the end of the
game (233 events, only 14 at t < 55), so the damage inside the probe's t range is
small. It is not zero, and a single forfeit desynchronizes every later position
in that game.

Cross-check against the committed 10k set: replaying the same 200 ARENA games
through `probe_labels` reproduces `analysis/linear_probe/activations/board_states.pt`
on **9,947 / 10,000** rows. The 53 differing rows are *exactly* the rows where
`t % 2` names the wrong mover (3 of the 200 games). So the label convention here
matches the existing set, and the existing set carries 53 mislabeled rows.

## Outputs

All under `bracket_dir()` = `$SPADE_BRACKET_DIR`, default
`/n/netscratch/ba_lab/Lab/$USER/spade_bracket`. This run:
`/n/netscratch/ba_lab/Lab/dcao/spade_bracket`.

### Events

| path | shape | dtype |
|---|---|---|
| `events/events.csv` | 2000 rows × 12 cols | text, 354 KB |
| `events/before.pt` | (2000, 8, 512) | float32 |
| `events/after.pt` | (2000, 8, 512) | float32 |
| `events/control.pt` | (2000, 8, 512) | float32 |
| `events/manifest.json` | mining stats | |
| `events/acts_manifest.json` | hooks, conditions, shapes | |

Layer axis is `blocks.{0..7}.hook_resid_post`, row i of every tensor is row i of
`events.csv`. `before` = position t−1 of `run(P+m)`; `after` = position t of
`run(P+m)`; `control` = position t of `run(P+m')`.

`events.csv` columns:

| column | meaning |
|---|---|
| `game_id` | index into the event-mining scan (0..1999) |
| `t` | move index of m; prefix is `tokens[:t]` |
| `mover` | +1 black, −1 white, from engine replay |
| `move`, `control_move` | vocab tokens (1..60) |
| `square_s`, `flip_set` | engine square 0..63; `flip_set` is the same square, the set has size 1 |
| `control_kind` | `strict` or `loose` |
| `control_flip_set` | space-joined engine squares flipped by m' |
| `n_legal` | regular moves available to the mover at t |
| `control_squares` | 64-char string, one code per engine square |
| `prefix` | space-joined t vocab tokens |

`control_squares` codes, taken on the board *before* m: `0` empty, `1` the played
square m, `2` the flipped square s, `3` unflipped disc of the mover's colour,
`4` unflipped opponent disc. The within-activation controls are codes 0, 3 and 4.
Read the CSV with plain `pd.read_csv`; `control_squares`, `prefix` and
`control_flip_set` come back as `object` (str) — 64 digits overflows int64, so
pandas will not coerce them.

### Probe-training set

400 games × positions [5, 55) = 20,000 rows, 2× the committed 10k set. Games are
taken from the head of the same stream, before event mining consumes any, so the
two sets share no games.

| path | shape | dtype |
|---|---|---|
| `probe/all_layer_activations.pt` | (8, 20000, 512) | float32 |
| `probe/board_states_relative.pt` | (20000, 8, 8) | int64 |
| `probe/board_states_absolute.pt` | (20000, 8, 8) | int64 |
| `probe/move_indices.pt` | (20000,) | int64 |
| `probe/mover_next.pt` | (20000,) | int8 |
| `probe/game_id.pt` | (20000,) | int64 |
| `probe/tokens.pt` | (400, 59) | int16 |

Layout matches `analysis/linear_probe/activations/` exactly — layer-major
activations, `[N, 8, 8]` int64 labels, `move_indices` in [5, 55) — so a loader
written against that directory works here unchanged. Rows are game-major,
position-minor.

Both label tensors are row-for-row aligned:
`board_states_relative` is `{0: empty, 1: theirs, 2: mine}` with "mine" =
`mover_next`; `board_states_absolute` is `{0: empty, 1: black, 2: white}`.
Verified: the two agree exactly where `mover_next == -1` (white) and are exactly
class-swapped where `mover_next == +1` (black); the empty class is identical in
both. Class counts — relative 590000/372206/317794, absolute 590000/350982/339018.

## Assumptions

1. Events where the opponent forfeits at t+1 are dropped (56 positions). Without
   this the mover plays twice in a row and s reads *mine* rather than *theirs*
   after m, which is the opposite of the headline prediction.
2. One event per game, so events are independent. The raw candidate density is
   reported separately above and is 21.7 per game.
3. The control move must not forfeit the opponent either, so `P+m` and `P+m'`
   have the same next mover and the parity term cancels in their difference.
4. `before` is read from `run(P+m)` at position t−1 rather than a third pass over
   `P`. Attention is causal so the two are the same computation; the 6.44e-06
   check above is the evidence.
5. `mover_next` is the colour that *actually* moves at t+1, which is decidable
   from the board alone: if the nominal mover has no regular move the engine
   hands the turn over. This is what makes it differ from `t % 2`.
