"""Aggregate the CosineSpaDE gamma sweep across seeds and sweep segments.

The grid was run in two segments with independent trainer numbering, so rows are matched
by configuration name and never by trainer index. Emits the per-seed table and the
mean/std table that the gamma-response figure draws its bands from.
"""

# def main(): ...   # -> results/multiseed/{cosine_sweep_all_seeds,aggregate_cosine_sweep}.csv
