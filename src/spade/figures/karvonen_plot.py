"""Shared Karvonen Figure-4 plotting library (pure matplotlib, no repo imports).

The 2×2 grid (loss recovered / L0 / coverage / reconstruction F1 vs sparsity) with per-
SAE-type styling and ±1-std error bars. Both figure entrypoints call `render_grid`.
"""

# SAE_CONFIG = {...}          # per-type color/marker/label
# TRAINER_CLASS_MAP = {...}   # trainer_class string -> display type
# ORDERED_TYPES = [...]

# def get_display_info(trainer_class): ...
# def plot_karvonen_subplot(ax, df, metric): ...   # incl. _error_bars from <metric>_std columns
# def render_grid(df, out_path, annotate=None): ...
