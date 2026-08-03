"""Merge the baseline, sweep and revived-config result CSVs into one table.

Joins on wandb_name. Exists because the chosen-best SpaDE configuration was retrained
after its original checkpoint was lost, and its row has to sit alongside the older rows in
the combined figure.

The retrained model is NOT metric-identical to the lost original — same config, same seed,
different numbers, because the trainer moved underneath it. Label that row as the config
under the current trainer, never as a reproduction.
"""

# SOURCES = [...]   # results/<experiment>/all_results.csv, resolved via paths.results_dir
# def main(): ...
