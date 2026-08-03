"""Sweep drivers: run one SAE type over a grid of sparsity settings.

Sweeping is a grid problem, not a per-type problem, so it lives beside core.py rather than
inside train_spade.py and train_cosine_spade.py — otherwise each entrypoint grows a copy.

Two hazards this module owns. Trainer indices restart at 0 in every sweep directory, so a
rerun with a different grid silently overwrites checkpoints and desynchronizes them from
their results; a sweep therefore writes into a named segment directory and refuses to
start on top of an existing one. And a grid that spans more than one segment must be
reassembled by config name, never by index — see spade.multiseed.aggregate_cosine_sweep.

Pick the range from the loss balance (spade.training.sweep_loss_balance), not by
extrapolating an L0 trend.
"""

# def run_grid(trainer_type, grid, segment, seed): ...
