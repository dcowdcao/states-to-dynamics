"""Every L0-vs-board-metric point, his and ours, on one schema.

Karvonen's published aggregate plus any number of `f1_results.csv` files written by
`karvonen_repro.evaluate`, normalized to one tidy frame the scoring step consumes.
Our rows are identified from the `config.json` beside each checkpoint, so a config is
named by the hyperparameters that produced it and not by a trainer index that gets
reused between sweeps. Rows that cannot be identified raise; they are never dropped.

Run it to see what landed, what each class covers on the L0 axis, and which points sit
below the random-weight model:

    python -m spade.l0_operating_point.load_points

Taken from `karvonen/pareto-pipeline`, which holds the same file under `karvonen_pareto/`
for its figure. Merging that branch means reconciling the two copies.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn.functional as F

from spade.common.paths import repo_root
from spade.saes.spade import softplus_inverse

SAVED_DATA = repo_root() / "external/Karvonen_Research/autoencoders/saved_data"
KARVONEN_CSV = SAVED_DATA / "06-10-othello-trained-results_with_07-10_topk.csv"
RANDOM_CSV = SAVED_DATA / "othello-6-10-random-f1_results.csv"

# Each value is a karvonen_repro.evaluate output directory: `f1_results.csv` once the
# whole set is aggregated, `per_checkpoint/**/f1_results.csv` while it is still running.
# A new sweep is a new entry here and needs no other change.
REPRO_ROOT = Path(
    os.environ.get("SPADE_KARVONEN_REPRO_DIR", f"/n/netscratch/ba_lab/Lab/{os.environ['USER']}")
)
# His five types come from his own sweep, so `karvonen_repro_ours_baselines` is
# deliberately absent: identify() has no DIALS entry for Standard/Gated/PAnneal/
# GatedAnneal/TopK and would raise on those rows rather than mix two training pipelines
# under one type name.
OUR_STREAMS = {
    "canonical": REPRO_ROOT / "karvonen_repro_spade_canonical",
    "gamma_sweep": REPRO_ROOT / "karvonen_repro_spade_sweep",
    "l0_coverage": REPRO_ROOT / "karvonen_repro_l0_coverage",
    "batch_topk_k80": REPRO_ROOT / "karvonen_repro_batchtopk",
    # The three 20-config grids, one per architecture, each internally consistent:
    # scripts/slurm/l0_grids/. These are what the operating point is chosen from; the
    # streams above are earlier runs kept because they add points, not because the
    # picks depend on them.
    "spade_l0_grid": REPRO_ROOT / "karvonen_repro_spade_l0_grid",
    "cosine_spade_l0_grid": REPRO_ROOT / "karvonen_repro_cosine_spade_l0_grid",
    "batch_topk_l0_grid": REPRO_ROOT / "karvonen_repro_batch_topk_l0_grid",
}
# karvonen_repro_cosine_beta is deliberately absent: the beta dial lives on
# karvonen/cosine-beta, so this branch would score those checkpoints at beta=1 and cannot
# retrain them. Pass it to --ours to look at it anyway.

PREFIX = "games_batch_to_state_stack_mine_yours_blank_mask_BLRRC"  # the paper's headline metric
METRICS = {
    "recon_f1": f"{PREFIX}_best_f1_score_per_class",
    "coverage": f"{PREFIX}_best_average_f1",
    "frac_recovered": "frac_recovered",
    "variance_explained": "frac_variance_explained",
}
COLUMNS = ["source", "trainer_class", "config", "seed", "l0", *METRICS, "dict_size", "path"]

# config.json trainer keys that name the run rather than the model. Everything else is
# part of the config's identity, so a knob added to a future sweep (a CosineSpaDE beta)
# splits configs by default instead of silently merging them.
IGNORED_CONFIG_KEYS = frozenset({"device", "seed", "wandb_name"})
# The subset that goes into the printed config name. Add a key here when a sweep adds a
# dial; forgetting to trips the label/identity check in `identify`.
DIALS = {
    "SpaDETrainer": ("lambda_init", "gamma_reg"),
    "CosineSpaDETrainer": ("beta", "gamma_reg"),
    "BatchTopKTrainer": ("k",),
}
SHORT = {"lambda_init": "lam", "gamma_reg": "gam", "beta": "beta", "k": "k"}
DIAL_DEFAULTS = {"beta": 1.0}  # CosineSpaDE ran at beta=1 before the dial was added

# One config, two training runs at the same seed. Value is a substring of the path to keep.
RESOLVED_COLLISIONS = {
    # CosineSpaDE gamma=1 was trained twice at seed 42 under different standardization
    # (checkpoints/cosine_spade at L0 175.96, the seed sweep at L0 175.24). Keeping the
    # sweep member makes gamma=1 one internally consistent five-seed config; the
    # canonical checkpoint is the same config, not a second point.
    "CosineSpaDE-beta1-gam1": "autoencoders_cosine_spade_sweep",
}


class PointsError(Exception):
    pass


def effective_lambda(lambda_init: float) -> float:
    """`softplus(softplus_inverse(x))` in fp32, the round trip SpaDEAutoEncoder does.

    `exp(x) - 1` quantizes there, so 8e-8, 9e-8 and 1e-7 all give lambda_pre
    -15.94238472 and train to byte-identical weights (md5-verified on seeds 43-46).

    Call the trainer's own function rather than reimplementing it: torch and numpy round
    `exp(x) - 1` differently at this magnitude, and a numpy emulation put 8e-8 and 1e-7
    in different configs, plotting one checkpoint as two points.
    """
    return float(F.softplus(softplus_inverse(float(lambda_init))).item())


NORMALIZE = {"lambda_init": effective_lambda}


def identify(path: Path) -> dict:
    """Trainer class, config name, config identity, seed and dict_size from config.json."""
    config_file = path / "config.json"
    if not config_file.exists():
        raise PointsError(f"no config.json beside {path}; cannot identify the row")
    trainer = json.loads(config_file.read_text())["trainer"]

    cls = trainer["trainer_class"]
    if cls not in DIALS:
        raise PointsError(f"{config_file}: no DIALS entry for {cls}")
    hypers = {k: NORMALIZE.get(k, lambda v: v)(v) for k, v in trainer.items()}
    # Checkpoints trained before a dial existed carry its default, not the key.
    for dial, default in DIAL_DEFAULTS.items():
        hypers.setdefault(dial, default)
    name = f"{cls.removesuffix('Trainer')}-" + "-".join(
        f"{SHORT[k]}{hypers[k]:.3g}" for k in DIALS[cls]
    )
    identity = tuple(sorted((k, str(v)) for k, v in hypers.items() if k not in IGNORED_CONFIG_KEYS))
    return {
        "trainer_class": cls,
        "config": name,
        "identity": identity,
        "seed": trainer["seed"],
        "dict_size": trainer["dict_size"],
    }


def read_f1_results(path: Path, source: str, ours: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "recon_f1_bm" in df.columns:  # karvonen_othello_sweep/sweep_full.csv naming
        df = df.rename(columns={"recon_f1_bm": METRICS["recon_f1"], "coverage_bm": METRICS["coverage"]})

    out = pd.DataFrame({"source": source, "l0": df["l0"], "path": df["autoencoder_path"]})
    for short, column in METRICS.items():
        out[short] = df[column]

    if not ours:
        out["trainer_class"] = df["trainer_class"]
        out["config"] = df["autoencoder_path"]
        out["identity"] = df["autoencoder_path"]
        out["seed"] = pd.NA
        out["dict_size"] = pd.NA
        return out

    ids = pd.DataFrame([identify(Path(p)) for p in df["autoencoder_path"]], index=out.index)
    mismatched = ids["trainer_class"] != df["trainer_class"]
    if mismatched.any():
        raise PointsError(
            f"{path}: trainer_class disagrees with config.json for "
            f"{df.loc[mismatched, 'autoencoder_path'].tolist()}"
        )
    return pd.concat([out, ids], axis=1)


def stream_csvs(path: Path) -> list[Path]:
    """A karvonen_repro output directory's results: the aggregate plus whatever
    per-checkpoint runs have landed while it is still evaluating. A path to one
    f1_results.csv is taken as given."""
    if path.is_file():
        return [path]
    return ([path / "f1_results.csv"] if (path / "f1_results.csv").exists() else []) \
        + sorted(path.glob("per_checkpoint/**/f1_results.csv"))


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (config, seed). Reports every collapse; raises on ones it cannot judge.

    Two rows measuring the same numbers are one measurement, however the checkpoint was
    named: the same mirror re-read, a checkpoint mirrored into two evaluation runs, or a
    pair like trainer_8/trainer_9 whose lambda_init differs only below fp32 resolution.
    Rows that share a config and seed but measure differently were trained twice and only
    RESOLVED_COLLISIONS can say which one is the point.
    """
    by_name = df.groupby("config")["identity"].nunique()
    if (by_name > 1).any():
        bad = by_name[by_name > 1].index.tolist()
        raise PointsError(
            f"configs {bad} differ in a hyperparameter that is not in DIALS; add it there"
        )

    keep = []
    for (config, seed), rows in df.groupby(["config", "seed"], dropna=False):
        distinct = rows.drop_duplicates(["l0", *METRICS])
        if len(distinct) < len(rows):
            print(f"  {config} seed {seed}: {len(rows)} rows, {len(distinct)} distinct "
                  f"measurement(s) [{', '.join(sorted(rows['source'].unique()))}]")
        if len(distinct) == 1:
            keep.append(distinct)
            continue
        chosen = distinct[distinct["path"].str.contains(
            RESOLVED_COLLISIONS.get(config, "\0"), regex=False)]
        if len(chosen) != 1:
            raise PointsError(
                f"{config} seed {seed} was trained {len(distinct)} times:\n  "
                + "\n  ".join(f"L0 {r.l0:.2f}  {r.path}" for r in distinct.itertuples())
                + f"\nadd '{config}' to RESOLVED_COLLISIONS with the path to keep"
            )
        print(f"  {config} seed {seed}: kept the run under "
              f"{RESOLVED_COLLISIONS[config]} of {len(distinct)}")
        keep.append(chosen)
    return pd.concat(keep).sort_index()


def one_per_config(df: pd.DataFrame) -> pd.DataFrame:
    """His figure is one point per config, single seed. Ours are trained over seeds 42-46."""
    lowest = df.groupby("config")["seed"].transform("min")
    keep = df["seed"].isna() | (df["seed"] == lowest)
    if not keep.all():
        print(f"  {(~keep).sum()} rows dropped: kept the lowest seed of each config")
    return df[keep]


def random_weight_floor(prefix: str = PREFIX) -> dict[str, float]:
    """Best board metric his 40 SAEs on the random-weight OthelloGPT reached.

    Metric-family specific: on the unmasked `..._mine_yours_BLRRC` columns the floor is
    recon 0.4267 / coverage 0.4316, but the figures plot the blank-masked family, whose
    floor is far lower because a random model cannot place the blanks either.
    """
    random_df = pd.read_csv(RANDOM_CSV)
    return {"recon_f1": random_df[f"{prefix}_best_f1_score_per_class"].max(),
        "coverage": random_df[f"{prefix}_best_average_f1"].max()}


def check_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of ours that are impossible, or no better than the random-weight model."""
    ours = df[df["dict_size"].notna()]
    floor = random_weight_floor()
    flags = {
        "L0 <= 0": ours["l0"] <= 0,
        "L0 >= dict_size": ours["l0"] >= ours["dict_size"],
        f"recon_f1 <= {floor['recon_f1']:.4f} (random weights)": ours["recon_f1"] <= floor["recon_f1"],
        f"coverage <= {floor['coverage']:.4f} (random weights)": ours["coverage"] <= floor["coverage"],
    }
    return pd.concat([ours[mask].assign(flag=name) for name, mask in flags.items() if mask.any()]) \
        if any(m.any() for m in flags.values()) else pd.DataFrame(columns=[*COLUMNS, "flag"])


def streams(patterns: list[str] | None) -> dict[str, Path]:
    """A `--ours` argument as given, or every stream we know about."""
    if patterns is None:
        return OUR_STREAMS
    return {Path(p).name: Path(p)
        for pattern in patterns for p in sorted(glob.glob(pattern, recursive=True))}


def load(our_streams: dict[str, Path] = OUR_STREAMS, karvonen_csv: Path = KARVONEN_CSV) -> pd.DataFrame:
    frames = [read_f1_results(karvonen_csv, "karvonen", ours=False)]
    for name, directory in our_streams.items():
        csvs = stream_csvs(directory)
        if not csvs:
            print(f"  {name}: MISSING ({directory})")
            continue
        rows = pd.concat([read_f1_results(p, name, ours=True) for p in csvs], ignore_index=True)
        print(f"  {name}: {len(rows)} rows from {len(csvs)} csv(s)"
              f"{' (per-checkpoint, still running)' if len(csvs) > 1 else ''}")
        frames.append(rows)
    df = pd.concat(frames, ignore_index=True)
    df["seed"] = df["seed"].astype("Int64")
    return dedupe(df)


# Karvonen's L0 range, split into the decades a Pareto figure is read in.
L0_BINS = [0, 10, 25, 50, 100, 200, 400, 571, np.inf]


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    binned = pd.cut(df["l0"], L0_BINS, right=False)
    table = pd.crosstab(df["trainer_class"], binned)
    table.insert(0, "L0 max", df.groupby("trainer_class")["l0"].max().round(1))
    table.insert(0, "L0 min", df.groupby("trainer_class")["l0"].min().round(1))
    table.insert(0, "n", df.groupby("trainer_class").size())
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--karvonen-csv", type=Path, default=KARVONEN_CSV)
    ap.add_argument("--ours", nargs="*", help="result dirs or f1_results.csv paths, globs allowed")
    args = ap.parse_args()

    df = one_per_config(load(streams(args.ours), args.karvonen_csv))

    print(f"\n{len(df)} configs\n")
    print(coverage_table(df).to_string())

    floor = random_weight_floor()
    print(f"\nrandom-weight floor on the plotted metrics: recon_f1 {floor['recon_f1']:.4f}, "
          f"coverage {floor['coverage']:.4f}")
    flagged = check_physics(df)
    print(f"{len(flagged)} of our points sit below it or outside 0 < L0 < dict_size")
    if len(flagged):
        print(flagged[["config", "seed", "l0", "recon_f1", "coverage", "flag"]].to_string(index=False))


if __name__ == "__main__":
    main()
