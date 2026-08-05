# Whose game is the shared continuation legal in?

Every spliced branch replays `tokens[t0+1:]` from the source game. That is a legal
game **on the reference's board and on no other**. This measures how much of it
survives on each branch's own board, from tokens and the engine alone — no GPU, no
model — and, once `$DIR/rollout/` exists, how legal the model's own sampled
continuations are.

```
python analysis/violation_divergence/suffix_legality/measure.py [--smoke]
```

Reads `$DIR[/smoke]/branches/branches.pt` and, if it exists,
`$DIR[/smoke]/rollout/branches/branches.pt`; skips the rollout half with a message
when it does not. Writes `results/violation_divergence/suffix_legality_*.csv`
(`suffix_legality_smoke_*` under `--smoke`).

## The defect, on the 2000-game spliced set

| branch | n | board after t0 | first continuation move legal | moves legal, mean / median | whole continuation legal |
|---|---|---|---|---|---|
| **reference** | 2000 | its own | **1.000** | 36.0 / 36 | **1.000** |
| `control_legal` | 2000 | its own | 0.752 [0.733, 0.771] | 3.94 / 2 | 0.002 |
| `control_legal_2` | 2000 | its own | 0.750 [0.730, 0.768] | 3.92 / 2 | 0.001 |
| the six illegal categories | 11527 | **none exists** | **undefined** | undefined | undefined |

Survival of the *whole* prefix of the continuation, `control_legal`
(`*_survival.csv`): 0.751 at one move, 0.573 at two, 0.353 at four, 0.158 at
eight, 0.016 at twenty-four. The reference is 1.000 at every offset by
construction.

**The asymmetry is the defect.** The reference is the one branch whose future the
study feeds it; a legal alternative loses the thread after a median of two moves,
and an illegal move has no board for the question to be asked about. Any quantity
measured after *t0* therefore compares one branch on its own trajectory against
branches on somebody else's, which is why `control_legal` — nominally the null —
carries the largest handicap in the study (README, "What does not hold").

### The illegal branches have no board, and that is the finding

Suffix legality is not zero for them, it is undefined: there is no state to replay
the continuation from. `*_undefined.csv` records what the engine does when asked to
play the branch move itself, which is the concrete form the undefinedness takes:

| category | n | engine verdict |
|---|---|---|
| `occupied_own`, `occupied_opp` | 4000 | square already holds a disc — `umpire` refuses |
| `empty_detached`, `empty_own_only`, `empty_no_bracket` | 5530 | flips nothing for either colour — `umpire` refuses |
| `opponent_only` | 1997 | the engine's **forfeit fallback replays it as the opponent's move** |

The last row is the one to watch. `opponent_only` is the only illegal category the
engine will accept, and it accepts it by pretending the mover forfeited and letting
the *opponent* play — a different game from the one the model was fed. So "the
engine gives a board" is not the same as "the branch has a board", and the spliced
design cannot be repaired by leaning on the engine. It has to be replaced, which is
what the rollout variant does.

The one statement about the illegal branches that needs no convention:
`p_continuation_replays_branch_square` — the replayed continuation later plays the
very square the branch already occupies in **94.1% / 97.4% / 97.9%** of the three
empty-square categories and **98.6%** of `opponent_only`. Occupancy never shrinks in
Othello, so that is an illegal move under *every* convention for what the branch
move did to the board. Read it with its cause in mind: a 59-move game covers almost
the whole board, so a square left empty at *t0* is nearly certain to be played
later. It says the shared continuation is wrong for these branches too; it does not
say how wrong.

It reads 0 for `occupied_own` and `occupied_opp`, and that is not a reprieve: their
square already holds a disc, so a legal continuation never plays it again. Their
conflict is at *t0* itself.

## Reproduction check

The script recomputes the README's `control_legal` figures and **stops** rather than
printing a table if it misses them, since two implementations disagreeing about
legality means one of them is wrong. It reproduces:

- `p_first_move_legal` **0.7515 [0.7325, 0.7710]** (1000-draw bootstrap over the 2000
  games) against the README's **0.768** — inside the interval.
- `median_moves_legal` **2**, exactly the README's.

The residual is the branch set, not the engine: the README's figure was measured by
the red team on the earlier **one-control** set (15531 rows), and the set on disk is
the **two-control** re-run (17527 rows), a different draw of both the control move
and — after the extra `fewer_than_3_legal` skip shifts the stream — the games. The
two independent controls in the current set agree with each other to 0.2 pp, so the
control draw is not where the difference lives. Settling it exactly needs the
one-control set regenerated, which needs the HF corpus; it is not reachable offline.

## Columns

Both `suffix_legality_{spliced,rollout}.csv` carry one row per branch role.

| column | meaning |
|---|---|
| `n` | branch rows: one per game in the spliced set, K per game in the rollout set |
| `board` | `legal` (the branch move is legal, the engine gives its board), `none` (no board exists — legality columns are empty), `placed convention` (rollout only, see below) |
| `p_first_move_legal` | the continuation's first move is legal on that board, with a 1000-draw bootstrap interval over rows |
| `mean_moves_legal`, `median_moves_legal` | continuation moves playable in order before the first illegal one |
| `p_all_moves_legal` | the whole continuation replays legally |
| `p_continuation_replays_branch_square` | the continuation plays the branch's own square — convention-free, defined for every branch |
| `p_reuse_before_end`, `mean_moves_before_reuse` | first move landing on a square already holding a disc, counting only discs the *prefix* placed and the continuation itself placed. Convention-free lower bound on illegality, and the honest number for a branch with no board |

`*_survival.csv` is `p_legal_through_offset` = P(the first `offset_after_t0` moves are
all legal), per role. `n_at_risk` shrinks with offset because a game's continuation
is `58 − t0` long.

`*_undefined.csv` is the engine verdict counts above.

## The rollout half

Runs only when `$DIR/rollout/branches/branches.pt` exists. Same columns, over the
model's own sampled continuation rather than the source game's.

For the branches whose move is legal, the numbers need no convention: the board is
the engine's, and `p_first_move_legal` is exactly OthelloGPT's own legal-move rate at
temperature 1.0 — the reference row is the baseline every other row should be read
against.

For the illegal branches there is still no board, so `moves_legal` is computed on a
**stated convention** (`board` reads `placed convention`): the disc lands on the
square in the mover's colour, the brackets it closes flip, the turn passes. That is
the reading the study's probe read-out supports, not a fact, and the rollout numbers
for illegal branches are only as good as it is. `p_reuse_before_end` in the same row
is the convention-free reading of the same question, and is the number to quote if
the convention is in doubt. Nothing on the spliced set uses the convention at all.

An incoherent rollout out of an impossible position is the new design's own caveat.
This is where it gets measured instead of assumed.
