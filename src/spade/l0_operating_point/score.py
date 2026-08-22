"""Collapse seeds to one point per config, then score each point across the four metrics.

The headline score is the equal-weight mean of the four min-max normalized Karvonen
metrics. `norm` runs over the whole pooled table, per metric, so the eight types stay on
one scale.

Two properties of that number the caller has to keep in view, both reported by `spread`:

- **Coverage and recon-F1 rank-disagree** (Spearman -0.6 to -0.9 across his sweep), so the
  mean of the two is flatter than either and an argmax over it can be weakly determined.
- **The min-max range is set by the collapsed SAEs.** His sweep contains dead Standard
  configs at L0 0 with frac_recovered as low as -0.81, so after normalization every healthy
  SAE's fidelity terms sit within a hair of 1 and contribute almost no variance. The score
  is in practice a mean of coverage and recon-F1 with the fidelity terms acting as a
  penalty on degeneracy.

`rank_score` is the sensitivity twin: mean of per-metric ranks, which cannot be compressed
by an outlier's range. Where the two rules disagree about a type, that type's pick is not
resolved by this table and gets said so.
"""

import pandas as pd

from spade.l0_operating_point.aggregate import SCORED

MINMAX = "score"
RANK = "score_rank"


def per_config(df: pd.DataFrame) -> pd.DataFrame:
    """One row per config: seed means, plus the L0 and score spread across seeds.

    Our configs carry 4-5 seeds and his carry one, so scoring the raw rows would weight a
    five-seed config five times against a one-seed config.
    """
    keys = ["family", "trainer_class", "config"]
    numeric = [*SCORED, "l0", "coverage_unmasked", "recon_f1_unmasked",
               "coverage_rescaled", "num_alive_features"]
    g = df.groupby(keys, dropna=False)
    out = g[numeric].mean()
    out["l0_std"] = g["l0"].std()
    for metric in SCORED:
        out[f"{metric}_std"] = g[metric].std()
    out["n_seeds"] = g["seed"].nunique().clip(lower=1)
    out["dict_size"] = g["dict_size"].first()
    out["eval_n_activations"] = g["eval_n_activations"].max()
    return out.reset_index()


def add_scores(df: pd.DataFrame, metrics: list[str] = SCORED) -> pd.DataFrame:
    df = df.copy()
    span = df[metrics].max() - df[metrics].min()
    normed = (df[metrics] - df[metrics].min()) / span.replace(0, pd.NA)
    df[MINMAX] = normed.mean(axis=1)
    df[RANK] = df[metrics].rank(pct=True).mean(axis=1)
    for metric in metrics:
        df[f"norm_{metric}"] = normed[metric]
    return df


def spread(df: pd.DataFrame, metrics: list[str] = SCORED) -> pd.DataFrame:
    """How much of the score's variance each normalized term actually carries."""
    normed = df[[f"norm_{m}" for m in metrics]]
    return pd.DataFrame({
        "std_of_norm": normed.std().values,
        "range_source_min": df[metrics].min().values,
        "range_source_max": df[metrics].max().values,
    }, index=metrics).sort_values("std_of_norm", ascending=False)
