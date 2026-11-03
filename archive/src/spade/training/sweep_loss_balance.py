"""Why the CosineSpaDE gamma sweep is flat: measure the two loss terms.

Loss is `MSE + gamma * reg`, with `reg = mean_b sum_j (1 - cos(x, d_j)) f_j`.
Sparsemax puts f on the simplex (`sum_j f_j = 1`), so reg is a convex combination
of per-atom misalignments, bounded in [0, 2] no matter how many atoms are active
— it is not an L0/L1 penalty and only sparsifies second-order, by concentrating
mass on the best-aligned atoms. MSE meanwhile sums over 512 dims. Printing both
terms per trained gamma gives the crossover gamma where the regularizer starts to
dominate, which is where L0 can actually move.

Also used to read L0 off the short-training probe runs that vet a candidate grid
before a full multi-seed sweep is submitted.

Usage:
    python -m spade.training.sweep_loss_balance [--sweep_dir DIR]
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from spade.common.study import study_dir
from spade.evaluation.karvonen_adapter import load_karvonen_sae, load_standardization_params

MULTISEED = study_dir("multiseed")
DEFAULT_SWEEP_DIR = MULTISEED / "seed_42/autoencoders_cosine_spade_sweep"
ACTS = MULTISEED / "data/activations/test_activations.pt"
N_ROWS = 16384


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", type=Path, default=DEFAULT_SWEEP_DIR)
    args = parser.parse_args()
    SWEEP_DIR = args.sweep_dir

    params = load_standardization_params(str(SWEEP_DIR))
    mean = torch.tensor(params["mean"])
    scale = params["scale"]

    acts = torch.load(ACTS, mmap=True, weights_only=True)[:N_ROWS].float()
    x = scale * (acts - mean)
    x_n = F.normalize(x, dim=-1)

    print(f"{'gamma':>8} {'L0':>8} {'MSE':>10} {'reg':>8} {'gamma*reg':>10} {'reg share':>10}")
    rows = []
    for d in sorted(SWEEP_DIR.glob("trainer_*"), key=lambda p: int(p.name.split("_")[1])):
        ae, cfg = load_karvonen_sae(str(d))
        gamma = cfg["gamma_reg"]
        with torch.no_grad():
            f = ae.encode(x)
            mse = (x - ae.decode(f)).pow(2).sum(-1).mean().item()
            mis = 1.0 - x_n @ F.normalize(ae.Ad, dim=0)
            reg = (mis * f).sum(-1).mean().item()
            l0 = (f > 0).sum(-1).float().mean().item()
        share = gamma * reg / (mse + gamma * reg)
        rows.append((gamma, l0, mse, reg))
        print(f"{gamma:>8g} {l0:>8.1f} {mse:>10.3f} {reg:>8.4f} {gamma * reg:>10.3f} {share:>9.1%}")

    gamma, _, mse, reg = rows[-1]
    print(f"\nAt the sweep's top gamma={gamma:g}, the regularizer is still "
          f"{mse / (gamma * reg):.1f}x smaller than MSE.")
    print(f"Parity (gamma*reg == MSE) needs gamma ~ {mse / reg:.0f}; "
          f"10x dominance needs gamma ~ {10 * mse / reg:.0f}.")


if __name__ == "__main__":
    main()
