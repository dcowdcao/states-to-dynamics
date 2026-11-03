# The game corpus every study streams

`adamkarvonen/othello_45MB_games`, read through `spade.othello.games.stream_games`
by the rules, straightness, violation-divergence, manifold-capture, legality-probe
and bracket-flip pipelines. Three properties of it are absent from the dataset card
and condition every number mined from it. Short version:

1. It is synthetic random-legal play, not human or engine games.
2. One opening line is missing from it — the corpus is a lexicographically sorted
   list of games cut off at a byte budget, and the cut lands inside the `e6` opening.
3. `stream_games` does not sample the corpus. Every study reads the same leading
   ~1% of it, in the same order, because the split is a single shard and the seed
   only permutes a 10,000-row window.

Numbers below are over the pooled 990,623 games (792,498 `train` + 198,125 `test`)
unless stated; the two splits agree to within 0.2 points on everything measured, so
the split itself is a random one.

## 1. Synthetic random-legal play

- Every game is exactly 60 plies and no game contains the pass token. The corpus is
  conditioned on games that fill the board — a random-legal game that strands both
  players early is not in it. (Unmarked forfeits still occur mid-game, which is why
  `t % 2` does not name the mover; see the bracket-flip design doc.)
- **The played move is uniform over the engine-legal set at essentially every
  prefix.** Grouping games by *k*-ply prefix and testing each prefix's children
  against uniform over `regular_moves`: at *k* = 5, 0 of the 1,185 prefixes with
  n ≥ 400 reject at p < 1e-3. At *k* = 2, 3, 4 exactly one prefix rejects, and it is
  the truncated branch of §2 each time. (One more rejects at *k* = 1 — `c4` splits
  99,012 / 97,308 / 98,208, p = 6e-4 — which at n ≈ 294,500 is a 1% wobble and the
  only deviation anywhere that §2 does not account for.)
- **No opening book.** The `train` split holds 273,772 distinct 8-ply prefixes and
  791,514 distinct 12-ply prefixes over its 792,498 games, and all 990,623 full games
  are distinct. In the bracket-flip event sample, 1,812 of the 1,817 games with ≥ 8
  plies have a distinct 8-ply prefix — 5 collisions against the 9.9 the corpus's own
  8-ply distribution predicts at that sample size. Human play concentrates on a few
  dozen book lines, which would collapse those 1,817 games onto a handful.

So any claim conditioned on this distribution is a claim about random legal play.
`archive/docs/restrav_summary.md` contrasts a synthetic arm against real championship play;
this corpus is the synthetic side of that contrast, for every study here.

## 2. The `e6` opening is missing 64% of its games

The four legal first moves are related by the symmetry group that fixes the starting
position — 180° rotation sends d3↔e6 and c4↔f5, transposition sends d3↔c4 and
f5↔e6 — so random legal play must give each of them 25%. It does not:

| first move | vocab token | games | share |
|---|---|---|---|
| d3 | 20 | 293,987 | 0.2968 |
| c4 | 27 | 294,528 | 0.2973 |
| f5 | 34 | 295,361 | 0.2982 |
| **e6** | **41** | **106,747** | **0.1078** |

χ² = 85,412 on 3 df; `e6` is 2.8× rarer than each of the other three. The
token→square mapping is `stoi` / `stoi_indices` from
`circuits.othello_engine_utils`, with the four centre squares absent from the vocab:
token 20 → square 19 → d3, 27 → 26 → c4, 34 → 37 → f5, 41 → 44 → e6.

The deficit is not spread over the `e6` subtree. It is a single cut in
**lexicographic order of the token sequence**, and following it down gives one
branch where each ply has full children, then one partial child, then nothing —
with `full` = the count random-legal play predicts from a pre-truncation corpus of
N ≈ 4 × 294,625 = 1,178,500 games:

| prefix | full-child expectation | children, in token order |
|---|---|---|
| `41` | 98,208 | 28: **98,007**, 40: *8,740*, 42: 0 |
| `41 40` | 19,642 | 19: *8,740*, 27: 0, 33: 0, 39: 0, 47: 0 |
| `41 40 19` | 3,274 | 20: **3,303**, 22: **3,331**, 28: *2,106*, 34: 0, 42: 0, 50: 0 |
| `41 40 19 28` | 655 | 27: **667**, 29: **696**, 39: **610**, 42: *133*, 48: 0 |
| `41 40 19 28 42` | 109 | 20: **96**, 22: *37*, 27: 0, 34: 0, 43: 0, 50: 0 |

(**bold** = at full strength, *italic* = the cut child.) Walking "the largest child
still present" to the bottom lands on a single game, and it is exactly the
lexicographically largest game in the corpus:
`41 40 19 28 42 22 27 20 23 51 60 ...`. Nothing else in the corpus deviates from
uniform random-legal play, bar the 1% wobble after `c4` noted in §1.

So the corpus is a random-legal sample of ~1.18M games, sorted, and cut off at
990,623 — most likely to the byte budget the dataset is named for (45 MB). The
sorted-then-truncated shape is the finding; which script did it is not recoverable
from the artifact.

**What this means for a study.** Prefix-conditioned work is unaffected: nothing in
the corpus is *illegal*, and off the truncated branch the play is uniform
random-legal. What is unsafe is any statement of the form "over positions the model
sees" that treats the corpus as a symmetric sample of the game — the board's own
symmetry group is broken in it, so a per-square or per-region result can inherit an
asymmetry that has nothing to do with the model. Symmetrize, or condition on the
opening, before reading anything positional off aggregate corpus statistics.

## 3. `stream_games` reads a window, not a sample

```python
dataset.shuffle(seed=seed, buffer_size=10000)
```

On an `IterableDataset` this shuffles the shard order and then draws from a moving
10,000-row reservoir. The `train` split is **one shard**
(`data/train-00000-of-00001.parquet`), so the shard shuffle is a no-op and the seed
only permutes within the window. Drawing N games therefore reads the first
N + 10,000 rows of the corpus and no further — 12,400 of 792,498 rows, 1.6%, for the
bracket flip.

Measured on fresh draws of 2,400 games:

| draw | game overlap with the seed-42 draw | e6 share |
|---|---|---|
| seed 42, buffer 10,000 | 1.000 | 0.113 |
| seed 7, buffer 10,000 | 0.205 | 0.100 |
| seed 1234, buffer 10,000 | 0.203 | 0.094 |
| seed 1234 vs seed 7 | 0.197 | — |
| no shuffle, raw head | 0.209 | 0.110 |
| seed 42, buffer 200,000 | 0.013 | 0.118 |

Changing the seed replaces only ~80% of the games, and 2,400/12,400 = 0.194 is
exactly the overlap expected from draws out of one 12,400-row window. Widening the
buffer to 200,000 drops the overlap to 0.013 ≈ 2,400/202,400, confirming the
mechanism.

**The window is not what produces the opening skew.** Read straight off the parquet,
the `e6` share is 0.1062–0.1085 in every 100,000-row block of the `train` split and
0.1078 over the whole thing, so no choice of window could have produced it — §2 is a
property of the corpus, §3 is a separate defect that happens to make the two arms of
the bracket-flip study reproduce it identically.

Two consequences, both methodological:

- **The studies are not independent samples of each other.** `rules`,
  `straightness`, `manifold_capture`, `legality_probe`, `violation_divergence` and
  `bracket_flip` all default to `--seed 42`, so they consume the same stream from
  position 0 in the same order. Nothing thins it either — every row is 60 tokens with
  no pass, so the `short_game` and `pass_token` skips in `collect_legal` and
  `events.usable` never fire — which makes their game sets nested prefixes of one
  list rather than independent draws. Within `bracket_flip`, the 400 probe games are
  rows 1–400 of that list and the 2,000 event games are rows 401–2,400: disjoint from
  each other, but not from any other study's corpus. Verified against the committed
  artifacts — a fresh seed-42 draw reproduces all 400 rows of `probe/tokens.pt` and
  all 2,000 prefixes in `events.csv`, in order.
- **A different seed is not a replication.** Re-running with `--seed 7` changes
  ~80% of the games but keeps the same 1.6% region of the corpus, so it cannot
  detect anything that varies over the corpus.

To draw from further in, raise `buffer_size` towards the size of the split. Do not
reach for `IterableDataset.skip` — on `datasets` 4.8.4 it does not land where it
says on this dataset: `skip(1000)` starts at row 0, and `skip(5000)` and
`skip(300000)` both start at row 1,000.

`spade.extraction.extract_activations.collect_games` carries its own copy of the same
`shuffle(seed, buffer_size=10000)` idiom, so the SAE pipeline reads the same window
whenever it is pointed at this corpus.

`stream_games` is deliberately left alone: changing the sampler would invalidate
every committed result. The fix, when a study needs a real sample, is at the call
site.
