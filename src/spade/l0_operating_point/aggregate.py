"""Every L0 point on one table: his five types from his sweep, our three from ours.

    python -m spade.l0_operating_point.aggregate

His side is `results/shared/l0_roster/karvonen_sweep_full.csv` rather than the published aggregate
`load_points` defaults to. It is the same 320 checkpoints run through his own pipeline,
so it carries the coverage his released per-checkpoint CSVs omit, both expansion factors,
and the 40 TopK configs he never published — one of which (trainer_53, k=140) is the best
coverage point in the whole sweep.

Types are never mixed across training pipelines. His Standard and p-anneal were trained at
`dictionary_learning@31b2828`; this repo vendors `43421f5`, where commit 8b95ec9 rewrote
those two trainers' recon loss from ||e||_2 to ||e||_2^2 and left gdm/gated_anneal alone.
His Standard sits at L0 131.7 where ours sits at 1094.9. So `StandardTrainer` on this table
means his, and our five retrained baselines are not on it.

Headline metrics are blank-masked, matching his figures. That family is mechanically capped
at 128/192 = 0.6667 because `one_hot_mask_idx` is unset for Othello: rankings are unaffected,
absolute values are not comparable to the unmasked floor, and `*_rescaled` is the x1.5 read.
"""

import argparse
from pathlib import Path

import pandas as pd

from spade.common.paths import repo_root
from spade.l0_operating_point import load_points

SWEEP_FULL = repo_root() / "results/shared/l0_roster/karvonen_sweep_full.csv"
OUT = repo_root() / "results/shared/l0_roster/pooled.csv"

# Blank-masked is the headline; the unmasked twin travels beside it, never scored.
MASKED = "games_batch_to_state_stack_mine_yours_blank_mask_BLRRC"
UNMASKED = "games_batch_to_state_stack_mine_yours_BLRRC"
BM_CAP = 128 / 192  # one_hot_mask_idx unset for Othello

SCORED = ["coverage", "recon_f1", "frac_recovered", "variance_explained"]
COLUMNS = ["family", "trainer_class", "config", "seed", "l0", *SCORED,
           "coverage_unmasked", "recon_f1_unmasked", "coverage_rescaled",
           "num_alive_features", "dict_size", "source", "eval_n_activations", "path"]


def _ours(csv: Path, source: str) -> pd.DataFrame:
    """One karvonen_repro f1_results.csv, with both metric families and the sample size.

    `path` is relative to `load_points.REPRO_ROOT` (the data root by default), so the
    committed table names no machine's directories. The released streams name their
    checkpoints that way too; a local run names them absolutely."""
    df = pd.read_csv(csv)
    paths = [load_points.REPRO_ROOT / p for p in df["autoencoder_path"]]
    ids = pd.DataFrame([load_points.identify(p) for p in paths])
    return pd.DataFrame({
        "family": "ours",
        "trainer_class": ids["trainer_class"],
        "config": ids["config"],
        "identity": ids["identity"],
        "seed": ids["seed"],
        "l0": df["l0"],
        "coverage": df[f"{MASKED}_best_average_f1"],
        "recon_f1": df[f"{MASKED}_best_f1_score_per_class"],
        "frac_recovered": df["frac_recovered"],
        "variance_explained": df["frac_variance_explained"],
        "coverage_unmasked": df[f"{UNMASKED}_best_average_f1"],
        "recon_f1_unmasked": df[f"{UNMASKED}_best_f1_score_per_class"],
        "num_alive_features": df["num_alive_features"],
        "dict_size": ids["dict_size"],
        "source": source,
        "eval_n_activations": df["eval_n_activations"],
        "path": [p.relative_to(load_points.REPRO_ROOT).as_posix() for p in paths],
    })


def _his(csv: Path = SWEEP_FULL) -> pd.DataFrame:
    df = pd.read_csv(csv)
    config = df["group_dir"].str.replace("othello-trained_model-layer_5-", "", regex=False) \
        + "/" + df["trainer_dir"]
    return pd.DataFrame({
        "family": "karvonen",
        "trainer_class": df["trainer_class"],
        "config": config,
        "identity": config,
        "seed": df["seed"].astype("Int64"),
        "l0": df["l0"],
        "coverage": df["coverage_bm"],
        "recon_f1": df["recon_f1_bm"],
        "frac_recovered": df["frac_recovered"],
        "variance_explained": df["frac_variance_explained"],
        "coverage_unmasked": df["coverage"],
        "recon_f1_unmasked": df["recon_f1"],
        "num_alive_features": df["num_alive_features"],
        "dict_size": df["dict_size"],
        "source": "karvonen_othello_sweep",
        # His published SAE-quality columns are one batch of 25 (PROTOCOL.md section 2),
        # and sweep_full reproduces that protocol.
        "eval_n_activations": 25,
        "path": df["group_dir"] + "/" + df["trainer_dir"],
    })


def pooled(our_streams: dict[str, Path] | None = None, sweep_full: Path = SWEEP_FULL) -> pd.DataFrame:
    frames = []
    for name, directory in (our_streams or load_points.OUR_STREAMS).items():
        csvs = load_points.stream_csvs(directory)
        if not csvs:
            print(f"  {name}: MISSING ({directory})")
            continue
        rows = pd.concat([_ours(c, name) for c in csvs], ignore_index=True)
        print(f"  {name}: {len(rows)} rows from {len(csvs)} csv(s)")
        frames.append(rows)

    ours = load_points.dedupe(pd.concat(frames, ignore_index=True))
    his = _his(sweep_full)
    print(f"  karvonen: {len(his)} rows from {sweep_full.name}")

    df = pd.concat([ours, his], ignore_index=True).drop(columns="identity")
    df["coverage_rescaled"] = df["coverage"] / BM_CAP
    return df[COLUMNS].sort_values(["family", "trainer_class", "l0"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-full", type=Path, default=SWEEP_FULL)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    df = pooled(sweep_full=args.sweep_full)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n{len(df)} points -> {args.out}")
    print(df.groupby(["family", "trainer_class"]).agg(
        n=("l0", "size"), l0_min=("l0", "min"), l0_max=("l0", "max"),
        seeds=("seed", "nunique")).round(1).to_string())


if __name__ == "__main__":
    main()
