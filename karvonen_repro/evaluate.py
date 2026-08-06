"""Run Karvonen's published board-state evaluation on arbitrary SAE checkpoints.

    python -m karvonen_repro.evaluate --output-dir OUT CKPT_DIR [CKPT_DIR ...]

Every number in the output CSV is produced by `circuits/`. This file decides only
which checkpoints run, where the artifacts land, and what has already finished.

Two modes, both first class, differing only in how many activation batches the
SAE-quality columns average over. PROTOCOL.md section 2: his published `l0`,
`l1_loss`, `l2_loss`, `frac_variance_explained`, `cossim`, `l2_ratio`, `frac_alive`
are single-batch estimates over 25 activation vectors, because `get_evals` calls
`evaluate` without `n_batches`. `--mode repro` keeps that, and is the only mode whose
output may be compared column-by-column against his CSV. `--mode measurement`
averages enough batches for the numbers to mean something on their own. The board
state stage is identical in both — it always sees 1000 games x 59 positions.

Loader contract: `load_sae(checkpoint_dir) -> (ae, meta)` returns a module that
consumes RAW model activations. His SAEs were trained on raw residual stream, ours on
standardized; folding that transform into `ae` is the loader's job and this harness
never applies one, so a loader that forgets it produces silently wrong F1.

One output dir per (n_inputs, mode, checkpoint set). n_inputs is baked into the
artifact filenames his code writes, and a second n_inputs in the same dir makes his
`get_all_results_file_names` see two candidates and refuse. One process per output
dir too: his per-group results.csv is a fixed path that two runs would race on.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch

from karvonen_repro import add_karvonen_to_path

add_karvonen_to_path()

import circuits.full_pipeline as full_pipeline  # noqa: E402
import circuits.pipeline_config as pipeline_config  # noqa: E402
import circuits.utils as utils  # noqa: E402

from karvonen_repro import patches  # noqa: E402
from spade.sae_loader.load_sae import PARAMS_FILE, load_sae  # noqa: E402

# His pipeline writes results.pkl / reconstruction.pkl / evals.pkl next to ae.pt, and
# the released checkpoints already ship his. Mirroring by symlink would write through
# and destroy them, so the work dir carries the loader's inputs and nothing else.
ARTIFACT_SUFFIXES = (".pkl", ".csv")

MODE_BATCHES = {"repro": 1, "measurement": 100}


def link(dst: Path, src: Path) -> None:
    if not dst.exists():
        dst.symlink_to(src.resolve())


def mirror(checkpoint_dir: Path, work_root: Path) -> Path:
    """`<sweep parent>/<sweep>/<trainer>/`, with the sweep's standardization params.

    `autoencoder_path` is what identifies a row downstream and compare_to_published.py
    reads the group off its second-to-last component, so `<sweep>/<trainer>/` has to
    survive. The level above it is what makes the mirror injective: every seed of the
    multi-seed study calls its sweep dir `autoencoders`, so on two levels seed 43 and
    seed 44 mirror onto the same path and the second is reported as already done.
    """
    sweep = checkpoint_dir.parent
    work_dir = work_root / sweep.parent.name / sweep.name / checkpoint_dir.name
    work_dir.mkdir(parents=True, exist_ok=True)
    for src in checkpoint_dir.iterdir():
        if src.is_file() and src.suffix not in ARTIFACT_SUFFIXES:
            link(work_dir / src.name, src)
    # trainSAE writes these once per sweep, and load_sae reads them from there.
    if (sweep / PARAMS_FILE).is_file():
        link(work_dir.parent / PARAMS_FILE, sweep / PARAMS_FILE)
    return work_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", nargs="+", type=Path, help="dirs holding ae.pt + config.json")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--mode", choices=MODE_BATCHES, default="repro")
    p.add_argument("--eval-batches", type=int, help="override the mode's batch count")
    p.add_argument("--n-inputs", type=int, default=1000, help="his published setting")
    p.add_argument("--batch-size", type=int, default=25, help="his published setting")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def stamp_protocol(csv: Path, mode: str, n_batches: int, batch_size: int) -> None:
    """The SAE-quality columns are meaningless without the sample size behind them,
    and repro mode's is 25. Record it so no row can be misread later."""
    df = pd.read_csv(csv)
    df["eval_protocol_mode"] = mode
    df["eval_n_batches"] = n_batches
    df["eval_n_activations"] = n_batches * batch_size
    df.to_csv(csv, index=False)


def main():
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoints = [c.resolve() for c in args.checkpoints]
    out = args.output_dir.resolve()
    work_root = out / "checkpoints"
    per_checkpoint = out / "per_checkpoint"
    work_root.mkdir(parents=True, exist_ok=True)
    per_checkpoint.mkdir(parents=True, exist_ok=True)

    config = pipeline_config.Config()
    config.N_GPUS = 1
    config.batch_size = args.batch_size
    config.eval_sae_n_inputs = args.n_inputs
    config.eval_results_n_inputs = args.n_inputs
    config.board_reconstruction_n_inputs = args.n_inputs

    # His dataset cache is two bare relative filenames, so cwd is what decides where
    # the shared othello_{train,test}_dataset.pkl land and get reused across runs.
    os.chdir(out)

    eval_n_batches = args.eval_batches if args.eval_batches else MODE_BATCHES[args.mode]
    work_dirs = [mirror(c, work_root) for c in checkpoints]

    for work_dir in work_dirs:
        relative = work_dir.relative_to(work_root)
        results_csv = per_checkpoint / relative / "results.csv"
        done = results_csv.with_name("f1_results.csv")
        if done.exists():
            print(f"done already, skipping: {relative}")
            continue
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        patches.bind(load_sae, [f"{work_dir}/"], device, eval_n_batches)
        full_pipeline.analyze_sae_groups([f"{work_root}/"], str(results_csv), config)
        stamp_protocol(done, args.mode, eval_n_batches, args.batch_size)

    finished = sorted(per_checkpoint.glob("*/*/*/f1_results.csv"))
    assert finished, "no checkpoint produced results"
    utils.concatenate_csv_files([str(p) for p in finished], str(out / "f1_results.csv"))


if __name__ == "__main__":
    main()
