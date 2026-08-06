"""Run Karvonen's published board-state evaluation on arbitrary SAE checkpoints.

    python -m karvonen_repro.evaluate --output-dir OUT CKPT_DIR [CKPT_DIR ...]

Every number in the output CSV is produced by `circuits/`. This file decides only
which checkpoints run, where the artifacts land, and what has already finished.

Loader contract: `load_sae(checkpoint_dir) -> (ae, meta)` returns a module that
consumes RAW model activations. His SAEs were trained on raw residual stream, ours on
standardized; folding that transform into `ae` is the loader's job and this harness
never applies one, so a loader that forgets it produces silently wrong F1.

One output dir per (n_inputs, checkpoint set). n_inputs is baked into the artifact
filenames his code writes, and a second n_inputs in the same dir makes his
`get_all_results_file_names` see two candidates and refuse.
"""

import argparse
import os
from pathlib import Path

import torch

from karvonen_repro import add_karvonen_to_path

add_karvonen_to_path()

import circuits.full_pipeline as full_pipeline  # noqa: E402
import circuits.pipeline_config as pipeline_config  # noqa: E402
import circuits.utils as utils  # noqa: E402

from karvonen_repro import patches  # noqa: E402
from karvonen_repro.provisional_loader import load_sae  # noqa: E402

# His pipeline writes results.pkl / reconstruction.pkl / evals.pkl next to ae.pt, and
# the released checkpoints already ship his. Mirroring by symlink would write through
# and destroy them, so the work dir carries the loader's inputs and nothing else.
ARTIFACT_SUFFIXES = (".pkl", ".csv")


def mirror(checkpoint_dir: Path, work_root: Path) -> Path:
    name = f"{checkpoint_dir.parent.name}__{checkpoint_dir.name}"
    work_dir = work_root / name
    work_dir.mkdir(parents=True, exist_ok=True)
    for src in checkpoint_dir.iterdir():
        if not src.is_file() or src.suffix in ARTIFACT_SUFFIXES:
            continue
        link = work_dir / src.name
        if not link.exists():
            link.symlink_to(src.resolve())
    return work_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", nargs="+", type=Path, help="dirs holding ae.pt + config.json")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-inputs", type=int, default=1000, help="his published setting")
    p.add_argument("--batch-size", type=int, default=25, help="his published setting")
    p.add_argument("--device", default="auto")
    return p.parse_args()


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

    work_dirs = [mirror(c, work_root) for c in checkpoints]
    assert len({d.name for d in work_dirs}) == len(work_dirs), "checkpoint names collide"

    for work_dir in work_dirs:
        results_csv = per_checkpoint / work_dir.name / "results.csv"
        if results_csv.with_name("f1_results.csv").exists():
            print(f"done already, skipping: {work_dir.name}")
            continue
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        patches.bind(load_sae, [f"{work_dir}/"], device)
        full_pipeline.analyze_sae_groups([f"{work_root}/"], str(results_csv), config)

    finished = sorted(per_checkpoint.glob("*/f1_results.csv"))
    assert finished, "no checkpoint produced results"
    utils.concatenate_csv_files([str(p) for p in finished], str(out / "f1_results.csv"))


if __name__ == "__main__":
    main()
