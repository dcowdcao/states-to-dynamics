# Numbers the appendix prints with no figure behind them

One script per group of numbers, each writing one committed file in `results/`. All CPU.

| Paper | Number | Script | Writes | Reads |
|---|---|---|---|---|
| A.2 | 792,498 / 198,125 games; 60 plies, no pass; e6 10.78%, χ² 106,901 (pooled); one rejecting prefix at k = 2, 3, 4 and none with n ≥ 400 at k = 5 | `corpus_stats.py` | `corpus_stats.json` | HF parquet at `caafaff9` |
| A.2 | no game in the corpus is illegal | `corpus_legality.py` | `corpus_legality.json` | HF parquet |
| A.2 | N draws read the first N + 10,000 rows; another seed keeps ≈ 20% | `stream_window.py` | `stream_window.json` | HF stream (network) |
| A.4 | covariance-matched floor ≈ 0.09 | `covariance_null.py` | `covariance_null.csv` | move-legality fit-set acts |
| A.9 | the rule panel's scan, ranked | `rule_panel_scan.py` | `rule_panel_scan.csv` | HF stream (network) |
| B.2 | past t0 + 8 within 0.5%, crossing by t0 + 13 | `rollout_ratio.py` | `rollout_ratio.csv` | `rollout/divergence/resid_post_5.npz` |
| B.2 | ~234×, 536×, within 1.4× | `span_stats.py` | `span_stats.csv` | the committed by-space CSVs |

    python analysis/appendix_numbers/<script>.py

What each number is, exactly, is the script's docstring. Three readings the paper's sentence
does not carry:

- **A.4's floor is ≈ 0.09**, the population E|cos| of covariance-shaped random directions.
  The paper's "0.084 to 0.110" is a range over `robustness_cells.csv`'s cells (which give
  0.082-0.110), from a battery that needs the gitignored rules corpus.
- **B.2's cut is on the original 2000 games.** On the extended tree the crossing is at r = 9;
  both populations are in the CSV.
- **The rule panel's 606/16 is not what a stated rule picks.** The criteria admit ~1,900
  positions, and no ranking over one or two of the CSV's features selects 606/16;
  `rule_panel_scan.csv` gives its rank under the tie-break the script states.
