"""CosineSpaDE metrics against gamma, across the full sweep range.

L0 is a sigmoid in log gamma; a grid that sits in the flat left tail reads as "gamma does
nothing", which is exactly the conclusion this figure exists to correct. Marks the loss-
parity gamma (~16, where the simplex-bounded regularizer meets the reconstruction term)
as the reference point the range should have been chosen from in the first place.

Reads results/multiseed/aggregate_cosine_sweep.csv (5 seeds); bands are across seeds.
"""

# def main(): ...   # -> figures/cosine_spade_sweep/
