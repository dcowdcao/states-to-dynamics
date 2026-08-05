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
python analysis/violation_divergence/game_selection/three_rules/score_three_rules.py
python analysis/violation_divergence/game_selection/three_rules/check_r21_rules.py
```

## 1. Feasibility is not the constraint

<!--FEASIBILITY-->

## 2. Cross-space agreement

<!--CONSISTENCY-->

## 3. Board legibility

<!--LEGIBILITY-->

## Where the three criteria disagree

<!--DISAGREE-->

## Outputs

| file | what |
|---|---|
| `results/three_rules_board.csv` | one row per complete game: *t0*, its percentile, the eight board terms, the legibility composite |
| `results/three_rules_ranking.csv` | the pool games: `sep` in all nine spaces, the consistency percentiles, the board terms |
| `results/three_rules_spaces.txt` | which `sep_j` column is which space |
| `results/three_rules_summary.txt` | all three sections, with the tables this README quotes |
| `results/three_rules_r21_check.txt` | the shortlist rescored in the R^21 curvature field |
