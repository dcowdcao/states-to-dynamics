"""The 9 spaces of the legality-probe study, and both activation sets encoded through each.

The roster is `spade.spaces.representative_spaces()` restricted to the seed-42
checkpoints: raw plus 8 SAEs. That function also carries the two random controls and,
once the multiseed netscratch tree exists, the six seed-43 sweep points — 17 entries in
all — so the filter is on `entry.seed`, not on a name set. `{"raw", *BASELINE_CONFIGS}`
(`bracket_flip/sae_arm.py:45`) is the wrong filter twice over: it readmits the degenerate
SpaDETrainer-othello that `spaces.EXCLUDED_SAE` drops, and it never names the default
SpaDE, `SpaDE-lam1e-03-gam1e-04`. `random_proj` / `random_topk` ride along as NULLS: not
part of the 9, but encoded in the same pass because every figure needs them.

Cached to `study_dir("legality")/latents/`, resumable per space:

    probe_<space>.pt   [10000, d] fp16   the layer-5 probe fit set
    row_<space>.json                     d, measured L0, prefix invariance

The probe fit set is cached because refitting a probe is the expensive thing. The branch
window is not cached: `stream_branch` yields it one `spans()` batch at a time, so a
read-out reduces [rows, 33, d] to its own scalars without a [17527, 33, 8192] tensor —
9.5 GB per space — ever existing. `space_quality.csv` joins the Karvonen eval metrics
(L0, variance explained, reconstruction F1) that F4 regresses rule preservation against.

Usage: python -m spade.legality_probe.spaces [--spaces raw,TopKTrainer-othello] [--smoke]
"""

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import torch

from spade.common.config import LAYER
from spade.common.device import select_device
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.multiseed.aggregate_seeds import config_key
from spade.saes.registry import SAEEntry
from spade.spaces import batched_apply, load_space, representative_spaces
from spade.violation_divergence.divergence import WINDOW
from spade.violation_divergence.extract_acts import spans

NULLS = ("random_proj", "random_topk")
N_BEFORE = sum(r < 0 for r in WINDOW)
# extract_acts.spans keeps a game's branches inside one chunk, and 248 x 33 <= the
# ENCODE_BATCH of 8192 keeps that chunk inside one GEMM, so two branches of a game are
# rows of the same matmul. Split them across GEMMs of different shape and cuBLAS reduces
# their shared prefix differently, which would break prefix invariance in the code only.
ROW_BATCH = 248
SMOKE_ROWS, SMOKE_PROBE = 200, 2000

# Same resolution as tpr_probe.ACTIVATIONS: the directory is gitignored, so a worktree
# needs the override to point at a checkout that has it.
ACTIVATIONS = Path(os.environ.get(
    "SPADE_LINEAR_PROBE_ACTIVATIONS",
    repo_root() / "analysis" / "linear_probe" / "activations"))


def roster() -> list[tuple[str, SAEEntry | None]]:
    """(space_name, entry_or_None) for raw + the 8 seed-42 SAEs. Never the nulls."""
    out = [(n, e) for n, e in representative_spaces()
        if n == "raw" or (e is not None and e.seed == 42)]
    names = [n for n, _ in out]
    assert len(out) == 9, names
    assert "SpaDETrainer-othello" not in names, names
    return out


def quality() -> pd.DataFrame:
    """L0 / variance explained / reconstruction F1 per space, indexed by config name."""
    evals = pd.read_csv(repo_root() / "results" / "acceptance" / "all_results.csv")
    evals["space"] = evals["wandb_name"].map(config_key)
    return evals.set_index("space")[["l0", "variance_explained", "reconstruction_f1"]]


def probe_acts() -> torch.Tensor:
    """[10000, 512] float32: the linear-probe fit set at layer 5."""
    return torch.load(ACTIVATIONS / "all_layer_activations.pt",
        map_location="cpu", weights_only=True)[LAYER]


def branch_window() -> tuple[torch.Tensor, torch.Tensor]:
    """[n, 33, 512] fp16 at board positions t0 + WINDOW, and each branch's game_id."""
    d = study_dir("divergence")
    meta = torch.load(d / "branches" / "branches.pt", map_location="cpu", weights_only=True)
    acts = torch.load(d / "acts" / f"resid_post_{LAYER}.pt", map_location="cpu", weights_only=True)
    pos = meta["t0"].unsqueeze(1) + torch.tensor(WINDOW)
    assert pos.min() >= 0 and pos.max() < acts.shape[1], "window leaves the board"
    return acts[torch.arange(len(pos)).unsqueeze(1), pos], meta["game_id"]


def prefix_bitwise(code: torch.Tensor, game_id: torch.Tensor) -> bool:
    """Are the r < 0 columns bitwise identical across every branch of a game?

    Causal attention makes the pre-t0 activations of a game's branches bit-identical;
    whether an encoder preserves that is what makes r < 0 a paired in-distribution
    control rather than an approximate one. `spans` never splits a game, so a span holds
    every branch of every game it touches and the question is answerable span by span.
    """
    _, inverse = game_id.unique(return_inverse=True)
    first = torch.full((int(inverse.max()) + 1,), len(game_id))
    first.scatter_reduce_(0, inverse, torch.arange(len(game_id)), "amin")
    before = code[:, :N_BEFORE]
    return bool(before.eq(before[first[inverse]]).all())


def stream_branch(name: str, entry: SAEEntry | None, branch: torch.Tensor,
                  game_id: torch.Tensor, device: str) -> Iterator[tuple[slice, torch.Tensor]]:
    """The branch window in this space, one `spans` batch at a time: [rows, 33, d] fp16.

    Never holds more than one span, so a caller that reduces each span as it arrives
    never materializes the whole [n, 33, d] code.
    """
    fn = load_space(name, entry, device)
    for start, end in spans(game_id, ROW_BATCH):
        rows = branch[start:end]
        yield slice(start, end), batched_apply(
            fn, rows.flatten(0, 1).float(), device).half().unflatten(0, rows.shape[:2])


def measure(name: str, entry: SAEEntry | None, probe: torch.Tensor, branch: torch.Tensor,
            game_id: torch.Tensor, out: Path, device: str) -> dict:
    probe_code = batched_apply(load_space(name, entry, device), probe, device).half()
    torch.save(probe_code, out / f"probe_{name}.pt")

    nonzero, positions, bitwise = 0, 0, True
    for rows, code in stream_branch(name, entry, branch, game_id, device):
        nonzero += int((code != 0).sum())
        positions += code.shape[0] * code.shape[1]
        bitwise &= prefix_bitwise(code, game_id[rows])
    return {"space": name, "d_space": probe_code.shape[-1],
            "l0_window": round(nonzero / positions, 3),
            "prefix_bitwise": bitwise, "n_window_positions": positions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default="", help="comma-separated filter")
    parser.add_argument("--smoke", action="store_true",
        help=f"{SMOKE_ROWS} branches, {SMOKE_PROBE} probe positions, into latents_smoke/")
    args = parser.parse_args()

    device = select_device()
    out = study_dir("legality") / ("latents_smoke" if args.smoke else "latents")
    out.mkdir(parents=True, exist_ok=True)

    probe = probe_acts()
    branch, game_id = branch_window()
    if args.smoke:
        probe, branch, game_id = probe[:SMOKE_PROBE], branch[:SMOKE_ROWS], game_id[:SMOKE_ROWS]
    for start, end in spans(game_id, ROW_BATCH):
        assert prefix_bitwise(branch[start:end], game_id[start:end]), \
            "raw r < 0 activations are not bitwise shared"

    chosen = roster() + [(n, None) for n in NULLS]
    if args.spaces:
        keep = set(args.spaces.split(","))
        chosen = [s for s in chosen if s[0] in keep]

    rows = []
    for name, entry in chosen:
        row_path = out / f"row_{name}.json"
        if row_path.exists():
            rows.append(json.loads(row_path.read_text()))
            print(f"skip {name} (cached)", flush=True)
            continue
        row = measure(name, entry, probe, branch, game_id, out, device)
        row_path.write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(f"{name:28} d {row['d_space']:5}  L0 {row['l0_window']:9.2f}  "
              f"prefix_bitwise {row['prefix_bitwise']}", flush=True)

    table = pd.DataFrame(rows).join(quality(), on="space")
    print(table.to_string(index=False), flush=True)
    if args.smoke:
        return
    csv = repo_root() / "analysis" / "legality_probe" / "results" / "space_quality.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv, index=False)
    print(f"wrote {csv}", flush=True)


if __name__ == "__main__":
    main()
