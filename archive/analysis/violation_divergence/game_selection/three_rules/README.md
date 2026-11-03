# One game, three rules, nine spaces

The parent sub-study picks the game whose six illegal **categories** separate best in
raw layer 5. The figure being built asks a different question: one game, drawn in all
nine spaces, showing the three constraint **rules** — R1 placement (`occupied_own`,
`occupied_opp`), R2 bracketing (`empty_detached`, `empty_own_only`,
`empty_no_bracket`), R3 turn (`opponent_only`) — with a board panel beside it. That
adds two constraints a single-space search cannot see: the game has to work in *every*
cell of the grid, not just the best one, and its violating squares have to be legible
on an 8×8 board at 3.9 in.

```
sbatch --export=ALL,SPADE_REPO=$PWD \
    scripts/slurm/studies/violation_divergence/submit_three_rules_selection.sbatch
```

Both scripts hold one game and one space at a time and peak at 617 MB, but a login node
caps a user at 8 GB and has no GPU, and the all-games-in-every-space version this
replaced was OOM-killed there before printing a line. 3 min on one H100.

## 1. Feasibility is not the constraint

**All 1550 branch-complete games admit all three rules at their own *t0*.** Branch
completeness *is* rule completeness: a complete game carries all six illegal categories
by definition, and the six partition the three rules. The design of one game, one *t0*,
three branch groups is safe — the question was worth asking, and the answer is that it
never binds.

Relaxing to one member per rule, **1997 of the 2000 source games** qualify. Neither
mechanism feared here bites:

| category | rule | missing in |
|---|---|---|
| `occupied_own` | R1 | **0** / 2000 |
| `occupied_opp` | R1 | **0** / 2000 |
| `empty_detached` | R2 | 49 / 2000 |
| `empty_own_only` | R2 | **370** / 2000 |
| `empty_no_bracket` | R2 | 51 / 2000 |
| `opponent_only` | R3 | 3 / 2000 |

R1 needs an occupied square and never lacks one: *t0* ≥ 10, so both colours are always
on the board. R3's single member is missing in 3 games. The scarce category is R2's
`empty_own_only` — empty, touching only the mover's discs — and R2 has two other
members, so scarcity there costs a *member*, not a *rule*.

**What does bind is the R^21 window, not the rules.** An R^21 curvature panel needs a
full-width *W*=8 descriptor window that still ends before the branch, so *t0* ≥ 17:
**1066 of 1550** complete games qualify. Two named candidates fail it outright and can
carry no `fork_curvature_space` grid and no `fork_and_board --plane r21`.

| game | *t0* | *t0* pct | R^21 window | nominated by |
|---|---|---|---|---|
| 1558 | 18 | 37.7 | yes | SpaDE argmax (`spread_ratio` 0.1st pct) |
| 191 | 26 | 71.3 | yes | CosineSpaDE argmax |
| 659 | 19 | 42.2 | yes | high in both |
| 1442 | 27 | 75.4 | yes | SpaDE runner-up |
| **706** | **16** | 29.0 | **no** | SpaDE robust alternative |
| **1202** | **12** | 11.2 | **no** | SpaDE robust alternative |
| 1235 | 20 | 46.4 | yes | incumbent `FIGURE_GAME` |
| 662 | 22 | 54.6 | yes | parent sub-study's pick |

## 2. Cross-space agreement

**Consistency is cheap. That is a negative result and it is the useful finding here.**
*sep* — mean pairwise distance between the three rule means over their RMS norm, on
full-width displacements — orders games almost identically in eight of the nine spaces.
Spearman ρ over the 300-game reference sample, between every pair of those eight:
**0.915 to 0.991** (raw L5 vs SpaDE 0.991, vs PAnneal 0.988; the floor is TopK vs
SpaDE). The SpaDE agent's ρ = 0.989 against raw layer 5 is not a SpaDE property; it is
what every dictionary but one does.

**CosineSpaDE is the one dissenter**: ρ 0.774–0.876 against the other eight, worst
against TopK. It also has the highest *sep* median (0.669 against raw layer 5's 0.616)
and the lowest within/between `ratio` median (0.938 against 0.960) — it both separates
the rule means further and holds their members together more tightly than any other
space. Its argmax is a different game, and that game does not travel: **191, the
CosineSpaDE argmax, sits at pct_min 61.7** — in its worst space it beats 62% of a
random complete game.

Ranking on the worst space rather than the peak therefore changes little at the top: the
pct_min and pct_max top-12 lists share five games, and the leaders have sep_cv 0.02–0.05
over all nine spaces. It is not free everywhere — game 2 (pct_max 100.0, pct_min 87.0),
game 509 (100.0 / 86.3) and game 659 (83.3 / **24.7**) are games a peak search buys and
one space then drops.

Named candidates, on their worst space:

| game | *t0* | pct_min | pct_mean | pct_max | sep_min | sep_cv | raw-L5 rank / 401 |
|---|---|---|---|---|---|---|---|
| 662 | 22 | **97.3** | 99.3 | 100.0 | 0.918 | 0.026 | 10 |
| 1442 | 27 | 96.7 | 98.3 | 99.7 | 0.857 | 0.030 | 12 |
| **1235** | 20 | 96.7 | **99.6** | 100.0 | **0.920** | 0.026 | **4** |
| 1558 | 18 | 95.0 | 97.7 | 99.0 | 0.837 | 0.038 | 45 |
| 706 | 16 | 87.0 | 92.0 | 96.0 | 0.758 | 0.038 | 121 |
| 191 | 26 | 61.7 | 70.1 | 84.0 | 0.649 | 0.064 | 194 |
| 1202 | 12 | 54.3 | 81.9 | 93.7 | 0.627 | 0.066 | 155 |
| 659 | 19 | **24.7** | 45.0 | 83.3 | 0.518 | **0.136** | 255 |

Two candidates do not survive contact with *sep*. **659 is the least consistent game on
the list** — worst-space 24.7, sep_cv 0.136, double any other candidate's — and its peak
space is CosineSpaDE, the dissenter. **191 is CosineSpaDE's argmax and no one else's.**
Both were nominated by searches scoring something other than rule separation, so this is
a disagreement between criteria and not an error in those searches. 1558 is the reverse
case: its rule members group tightest of the eight — worst-space `ratio` 0.650, against
per-space medians of 0.938–0.983 — while its *sep* is 45th of 401 in raw layer 5.
Coherent rules whose means are not especially far apart.

Selecting the pool in raw layer 5 missed nothing: the best pct_min inside the reference
sample alone is game 1532 at 98.0, below the pool's best, game 1628 at 100.0.

**The R^21 field is a different ranking altogether.** Rescoring the 17-game shortlist on
Δφ — φ applied per member and the rule mean taken afterwards, because φ is nonlinear and
does not commute with the mean — gives Spearman ρ = **0.074** on worst-space *sep* and
**−0.020** on mean *sep* against the displacement field. The best worst-space game is
1532 in R^21 and 499 in displacements. For the R^21 panels, a game chosen on
displacements is close to a random draw.

## 3. Board legibility

Chebyshev distances in cells, on the panel `branch_panels.board_panel` draws: 3.9 in ÷ 8
cells = **35.1 pt per cell**.

**Nothing collides, in any game.** Two branches never play the same square
(`branches.verify` asserts it), so `min_rule_gap` ≥ 1 always, which is 35 pt between
markers 8 pt across. The tightest configuration the panel can produce is the diagonal
one — a branch's would-be disc in its down-right corner against the next square's marker
in its up-left corner, 0.4·√2 = 0.57 cells = **19.9 pt apart, against 8.2 pt of combined
radius**. It occurs in 444/1550 games and still leaves an 11.7 pt gap. Crammed corners
are not the failure mode either: the six illegal squares fall inside one 4×4 quadrant in
**1 of 1550** games.

What varies is how much work the reader does. Over all 1550 complete games:

| term | min | median | max | what a low value costs |
|---|---|---|---|---|
| `min_rule_gap` | 1 | 1 | 4 | two rules' squares adjacent, colour the only cue |
| `spread` (RMS from centroid) | 1.41 | 3.17 | 4.48 | six squares in one region of the board |
| `min_context` | 1 | 1 | 3 | a branch square with one neighbouring disc |
| `detached_gap` | 2 | 2 | 3 | `empty_detached` drifting off alone (down is better) |

Different-rule squares share an edge or corner in **1014/1550** games, so the incumbent's
one touching pair is the norm and not a defect. `legibility` is the mean percentile of
the four terms above, the last reversed; it is **argued, not measured** — no reader was
tested.

| game | *t0* | legibility | `min_rule_gap` | touching | glyph pairs | `spread` | quadrants |
|---|---|---|---|---|---|---|---|
| **191** | 26 | **73.9** | 2 | 0 | 0 | 3.32 | 4 |
| 1558 | 18 | 60.2 | 2 | 0 | 0 | 3.27 | 3 |
| 1235 | 20 | 54.3 | 1 | 1 | 0 | **3.57** | 4 |
| 1442 | 27 | 50.5 | 1 | 1 | 0 | 3.37 | 3 |
| 706 | 16 | 48.4 | 2 | 0 | 0 | 2.69 | 4 |
| 662 | 22 | 38.6 | 1 | 2 | 1 | 2.83 | 3 |
| 659 | 19 | 34.8 | 1 | 1 | 0 | 2.48 | 3 |
| 1202 | 12 | 28.1 | 1 | **4** | 1 | 2.95 | 4 |

All eight are drawable. The two at the bottom are the ones a reader has to work at: 1202
has four touching different-rule pairs and its `empty_detached` square 3 cells from any
disc, and 662 has two touching pairs, a glyph pair, and a spread of 2.83 against the
population median 3.17.

## Where the three criteria disagree

Sharply, and in a way no single search would have surfaced.

- **The best game on the board is nearly the worst across spaces, and vice versa.** 191
  leads legibility at 73.9 and sits at pct_min 61.7; 662 leads the candidates at
  pct_min 97.3 and sits at legibility 38.6. Ranking on either alone picks a game the
  other rejects.
- **659 fails both** (pct_min 24.7, legibility 34.8) while passing feasibility. Being
  high in both SpaDE searches did not survive a criterion neither of them used.
- **706 and 1202 fail feasibility**, which neither other criterion can see: *t0* = 16
  and 12 leave no branch-free *W*=8 window, so no R^21 panel exists for them however
  they score.
- **The two fields disagree with each other.** ρ(R^21, displacement) = 0.074 on
  worst-space *sep*. Whichever field the game is picked in, the other is not served, and
  `fork_and_board` draws both.
- **The incumbent is the argmax of nothing, and that is the case for it.** Game 1235 is
  level 2nd on consistency (pct_min 96.7, tied with 1442; 1st on pct_mean at 99.6), 3rd
  on legibility (54.3), has the R^21 window, and its *t0* = 20 is at the 46th
  percentile. Three candidates sit in the upper half of both scored criteria — 1558,
  1235, 1442 — and on the mean of the two percentiles they are **1558 at 77.6, 1235 at
  75.5, 1442 at 73.6**, a 2-point spread inside an argued composite. **No change to
  `FIGURE_GAME` is warranted**: nothing here separates the incumbent from its nearest
  alternative by more than the criterion's own slack.
- **Better games exist outside the nominated set.** Ranked on consistency and legibility
  jointly, **game 1068** (*t0* = 28, pct_min 97.7, pct_max 100.0, legibility 73.7,
  `min_rule_gap` 3, no touching pair, all four quadrants, R^21 window) beats every named
  candidate's consistency (662's 97.3 is the best of them) while matching the best
  legibility on the list (191's 73.9); 1541 and 499 also clear both. Observations only —
  a criterion fixed after the candidates were known is not a pre-registration, and
  swapping `FIGURE_GAME` on it would be selecting on the outcome.

## Outputs

| file | what |
|---|---|
| `results/three_rules_board.csv` | one row per complete game: *t0*, its percentile, the R^21 window, the eight board terms, the legibility composite |
| `results/three_rules_ranking.csv` | the 401 pool games: `sep` in all nine spaces, the consistency percentiles, the board terms |
| `results/three_rules_spaces.txt` | which `sep_j` column is which space |
| `results/three_rules_summary.txt` | all three sections, with the tables this README quotes |
| `results/three_rules_r21_check.txt` | the shortlist rescored in the R^21 curvature field |
