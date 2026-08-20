"""Does A3's layer-5 read-out survive resampling the probe's free choices?

Refits the TPR over seed, weight decay and role dimension, and recomputes the
five headline numbers of analysis/bracket_flip/reports/RESULTS_A3.md at every
setting.

Many probes is the point here, so this is the one place in the study that fits its
own TPRs rather than loading `spade.probes.load`'s. What it borrows is the
*protocol*: `spade.probes.train.fit`, split by game, early stop on validation
accuracy. Probes cache under $SPADE_BRACKET_DIR/robustness/, never a canonical path.

Everything reported is gauge-invariant -- accuracies, decoded classes,
proportions. Filler coordinates swing across seeds at constant accuracy
(analysis/bracket_flip/reports/PROBE_REPORT.md 4), so resampling them would
measure nothing.

Usage: sbatch scripts/slurm/studies/bracket_flip/submit_bracket_robustness.sbatch
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from spade.bracket_flip.probe_space import (
    BATCH,
    EMPTY,
    MAIN_LAYER,
    MINE,
    N_TRAIN,
    RESULTS,
    ROLE_CODE,
    SEED,
    THEIRS,
    event_acts,
    events,
    fit_split,
    occupancy_ceiling,
    owner,
    probe_acts,
    probe_labels,
    role_codes,
    swapped,
)
from spade.common.device import select_device
from spade.common.study import study_dir
from spade.probes.tpr import ROLE_DIM, WEIGHT_DECAY, TPR, solve_residual_delta_for_binding_delta
from spade.probes.train import CLASSES, fit

SEEDS = (42, 7, 2026, 4242, 31337, 90210)  # 42 is the canonical protocol's
WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2, 0.5, 2.0)  # 1e-3 the canonical argmax, 2.0 the old fit's
ROLE_DIMS = (16, 32, 52, 64)  # 16 upstream's default, 52 the design's, 64 one role per square
BLOCKS = tuple(range(8))

EDIT_SCALE = 8.0  # PROBE_REPORT.md 5: flips every event at r=52, so ranks are compared at parity
N_EDITS = 200

CACHE = study_dir("bracket") / "robustness"


@dataclass(frozen=True)
class Fit:
    layer: int
    basis: str
    seed: int = SEED
    weight_decay: float = WEIGHT_DECAY
    role_dim: int = ROLE_DIM

    @property
    def key(self) -> str:
        return (f"tpr_{self.basis}_layer{self.layer}_seed{self.seed}"
                f"_wd{self.weight_decay:g}_r{self.role_dim}")


def tpr(setting: Fit, device: str) -> TPR:
    """One TPR at one setting of the three swept knobs, under the shared protocol."""
    path = CACHE / f"{setting.key}.pt"
    acts = probe_acts(setting.layer)
    arm = TPR(acts.shape[-1], setting.seed, setting.weight_decay, setting.role_dim)
    if path.exists():
        arm.load_state_dict(torch.load(path, weights_only=True))
        return arm.to(device).requires_grad_(False)

    started = time.time()
    tr, va = fit_split()
    y = probe_labels(setting.basis).reshape(-1, 64)
    with torch.enable_grad():
        fit(arm, acts[tr], y[tr], acts[va], y[va], device, BATCH)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(arm.state_dict(), path)
    print(f"fit {setting.key} in {time.time() - started:.0f}s", flush=True)
    return arm.to(device).requires_grad_(False)


def predictions(probe: TPR, layer: int, device: str) -> Tensor:
    """[n_eval, 64] decoded class on the held-out rows."""
    return probe(probe_acts(layer)[N_TRAIN:].to(device)).argmax(-1)


def accuracy(probe: TPR, layer: int, basis: str, device: str) -> float:
    labels = probe_labels(basis)[N_TRAIN:].reshape(-1, 64).to(device)
    return (predictions(probe, layer, device) == labels).float().mean().item()


def occupancy_split(probe: TPR, basis: str, device: str) -> dict:
    """Layer-5 accuracy against the occupancy-only ceiling: call empty/occupied
    perfectly, then always answer the majority colour. A3 claims acc_abs sits exactly
    there, so the same decomposition on the relative probe is the contrast that reads it."""
    labels = probe_labels(basis)[N_TRAIN:].reshape(-1, 64).to(device)
    pred = predictions(probe, MAIN_LAYER, device)
    occupied = labels != EMPTY
    share = [(labels == c).float().mean().item() for c in range(len(CLASSES))]
    ceiling = occupancy_ceiling(labels)
    acc = (pred == labels).float().mean().item()
    colour = (pred[occupied] == labels[occupied]).float().mean().item()
    majority = max(share[1:]) / (1 - share[EMPTY])  # the ceiling's colour half, on occupied
    tag = {"relative": "rel", "absolute": "abs"}[basis]
    return {
        f"acc_{tag}_L5": acc,
        f"ceiling_{tag}": ceiling,
        f"acc_minus_ceiling_{tag}_pts": 100 * (acc - ceiling),
        f"occupancy_{tag}": ((pred != EMPTY) == occupied).float().mean().item(),
        f"colour_on_occupied_{tag}": colour,
        f"majority_colour_{tag}": majority,
        f"colour_over_majority_{tag}_pts": 100 * (colour - majority),
    }


def readout(probe: TPR, layer: int, ev: pd.DataFrame,
            acts: dict[str, Tensor], device: str) -> dict:
    """M1's proportions at one layer: point estimates only, since the sweep's own
    spread replaces A3's bootstrap as the uncertainty."""
    codes, square_s = role_codes(ev), ev.square_s.to_numpy()
    pred = {c: probe(acts[c][:, layer].to(device)).argmax(-1).cpu().numpy()
            for c in ("before", "after")}
    s = {c: p[np.arange(len(square_s)), square_s] for c, p in pred.items()}
    unflipped = np.isin(codes, [ROLE_CODE["unflipped_own"], ROLE_CODE["unflipped_opp"]])
    swaps = swapped(pred["before"], pred["after"])
    return {
        "p_s_unchanged_rel": float((s["before"] == s["after"]).mean()),
        "p_s_absolute_correct": float(((owner(s["before"], "before") == -1)
                                       & (owner(s["after"], "after") == 1)).mean()),
        "p_unflipped_swap_rel": float((swaps & unflipped).sum() / unflipped.sum()),
    }


def headline(seed: int, weight_decay: float, role_dim: int, ev: pd.DataFrame,
             acts: dict[str, Tensor], device: str) -> dict:
    """All five of RESULTS_A3.md's layer-5 numbers at one setting, bar peak_layer."""
    probe = tpr(Fit(MAIN_LAYER, "relative", seed, weight_decay, role_dim), device)
    absolute = tpr(Fit(MAIN_LAYER, "absolute", seed, weight_decay, role_dim), device)
    return {"seed": seed, "weight_decay": weight_decay, "role_dim": role_dim,
            **occupancy_split(probe, "relative", device),
            **readout(probe, MAIN_LAYER, ev, acts, device),
            **occupancy_split(absolute, "absolute", device)}


# ---------------------------------------------------------------- rank collateral

def role_overlap(probe: TPR) -> dict:
    """Off-diagonal |cos| of the role Gram: 64 roles in d_r < 64 dimensions cannot be
    orthogonal, and PROBE_REPORT.md 5 shows this alone predicts an edit's collateral."""
    roles = probe.roles
    cos = (roles @ roles.T) / torch.outer(roles.norm(dim=-1), roles.norm(dim=-1))
    off = cos[~torch.eye(64, dtype=torch.bool, device=cos.device)].abs()
    return {"role_cos_mean": off.mean().item(), "role_cos_max": off.max().item()}


def collateral(probe: TPR, layer: int, device: str) -> dict:
    """Squares other than s whose decoded class changes under a theirs->mine edit at s.

    One square per activation, so the trials stay independent; fixed scale 8, which
    flips every event at r=52, so ranks are compared where the edit always lands.
    """
    acts = probe_acts(layer)[N_TRAIN:N_TRAIN + N_EDITS].to(device)
    roles = probe.roles
    delta_filler = probe.fillers[MINE] - probe.fillers[THEIRS]
    rng = np.random.default_rng(SEED)

    changed, flipped = [], []
    for h in acts:
        pred = probe(h).argmax(-1)
        candidates = torch.nonzero(pred == THEIRS).flatten()
        square = int(candidates[rng.integers(len(candidates))])
        step = solve_residual_delta_for_binding_delta(
            probe.module.binding_map, torch.outer(roles[square], delta_filler))
        after = probe(h + EDIT_SCALE * step).argmax(-1)
        differs = after != pred
        changed.append(int(differs.sum()) - int(differs[square]))
        flipped.append(bool(after[square] == MINE))
    return {"n_edits": len(changed), "edit_flip_rate": float(np.mean(flipped)),
            "collateral_mean": float(np.mean(changed)),
            "collateral_zero_rate": float(np.mean(np.array(changed) == 0)),
            "collateral_max": int(np.max(changed))}


# ---------------------------------------------------------------- driver

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    torch.set_grad_enabled(False)
    print(f"device {device}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    ev, acts = events(), event_acts()

    rows = []
    for seed in SEEDS:
        for layer in BLOCKS:
            probe = tpr(Fit(layer, "relative", seed=seed), device)
            rows.append({"seed": seed, "layer": layer,
                         "acc_rel": accuracy(probe, layer, "relative", device),
                         **readout(probe, layer, ev, acts, device)})
    depth = pd.DataFrame(rows)
    depth.to_csv(RESULTS / "robustness_depth.csv", index=False)

    seeds = pd.DataFrame([headline(seed, WEIGHT_DECAY, ROLE_DIM, ev, acts, device)
                          for seed in SEEDS])
    for metric in ("p_s_absolute_correct", "p_s_unchanged_rel",
                   "p_unflipped_swap_rel", "acc_rel"):
        peak = depth.loc[depth.groupby("seed")[metric].idxmax()].set_index("seed").layer
        seeds[f"peak_layer_{metric}"] = seeds.seed.map(peak)
    seeds.to_csv(RESULTS / "robustness_seeds.csv", index=False)
    print(seeds.to_string(index=False), flush=True)

    decay = pd.DataFrame([headline(SEED, wd, ROLE_DIM, ev, acts, device)
                          for wd in WEIGHT_DECAYS])
    decay.to_csv(RESULTS / "robustness_weight_decay.csv", index=False)
    print(decay.to_string(index=False), flush=True)

    rows = []
    for role_dim in ROLE_DIMS:
        probe = tpr(Fit(MAIN_LAYER, "relative", role_dim=role_dim), device)
        rows.append({**headline(SEED, WEIGHT_DECAY, role_dim, ev, acts, device),
                     **role_overlap(probe), **collateral(probe, MAIN_LAYER, device)})
    rank = pd.DataFrame(rows)
    rank.to_csv(RESULTS / "robustness_rank.csv", index=False)
    print(rank.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
