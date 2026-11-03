"""Does the cross-space ranking survive the move into the R^21 curvature field?

`score_three_rules.py` ranks games on how far apart their three rule means sit in
the displacement field D(t) = z_b(t) - z_ref(t). `fork_and_board --plane r21` and
all four `fork_curvature_space` grids draw a different field: the ReStraV
descriptor difference dphi(t) = phi(z_b(t)) - phi(z_ref(t)). A game chosen in one
and drawn in the other is chosen on a proxy, so the shortlist is rescored here.

**phi is nonlinear, so the rule mean is taken per member and never before phi.**
`windowed` descriptors each branch's own trajectory -- rows are independent -- and
the mean over a rule's members is taken afterwards, inside R^21. Averaging the
member trajectories first would measure the curvature of a path no branch takes,
and would look entirely plausible. `test_curvature_plane.py` pins the difference.

Two consequences of the field, both structural rather than statistical:

- A full-width W=8 window that still ends before the branch needs t0 >= 17, so a
  shortlist entry below that cannot be scored here at all and is reported, not
  dropped silently. That is the binding feasibility constraint on any figure with
  an R^21 panel.
- dphi's 21 columns mix degrees with activation-scale lengths, so they are
  column-scaled before any distance is taken, exactly as `fork_and_board`'s
  `spread_field` does.

R3 has one member: its within-rule spread is 0 by construction here too.

Usage: python analysis/violation_divergence/game_selection/three_rules/check_r21_rules.py [--n 12]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from score_three_rules import CANDIDATES, OUT_DIR, score_field

from spade.common.device import select_device
from spade.common.study import study_dir
from spade.violation_divergence.branches import load_game, reference_row
from spade.violation_divergence.compute_divergence import sae_encoder
from spade.violation_divergence.curvature_plane import W, windowed
from spade.violation_divergence.fork_plane import positions, spaces

RANKING = OUT_DIR / "three_rules_ranking.csv"


def r21_field(F: np.ndarray, ref: int, post: np.ndarray) -> np.ndarray:
    """Column-scaled dphi over the drawn positions. phi is applied to each branch
    on its own inside `windowed`; the caller means over members afterwards."""
    phi = windowed(F, W)
    dphi = phi - phi[ref]
    sd = np.nanstd(dphi.reshape(-1, dphi.shape[-1]), axis=0)
    field = (dphi / np.where(sd > 0, sd, 1))[:, post]
    assert np.isfinite(field).all(), "an undefined descriptor window reached the score"
    return field


def shortlist(n: int) -> list[dict]:
    rows = list(csv.DictReader(RANKING.open()))
    top = rows[:n] + [r for r in rows if int(r["game"]) in CANDIDATES]
    return list({int(r["game"]): r for r in top}.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12,
                        help="how many of the ranking's top games to rescore")
    args = parser.parse_args()

    device = select_device()
    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)

    rows = shortlist(args.n)
    games = [load_game(meta, int(r["game"])) for r in rows]
    skipped = [g.game for g in games if positions(g.t0, W - 1)[W - 1] >= g.t0]
    games = [g for g in games if g.game not in skipped]
    print(f"{len(games)} of {len(rows)} shortlisted games carry a branch-free W={W} "
          f"window; no R^21 panel is possible for {skipped}")

    windows = [positions(g.t0, W - 1) for g in games]
    Z = [acts[g.idx][:, p, :].float() for g, p in zip(games, windows)]
    sep = np.zeros((len(games), len(spaces())))
    for j, (label, entry) in enumerate(spaces()):
        enc = None if entry is None else sae_encoder(entry, device)
        for i, (g, p, z) in enumerate(zip(games, windows, Z)):
            if enc is None:
                F = z.numpy()
            else:
                with torch.no_grad():
                    f = enc(z.reshape(-1, z.shape[-1]).to(device))
                F = f.reshape(z.shape[:2] + (-1,)).cpu().numpy()
            ref = reference_row(g.rows)
            sep[i, j] = score_field(r21_field(F, ref, p >= g.t0), g.rows)[0]
        print(f"  {label:<32} sep median {np.median(sep[:, j]):.4f}")

    disp = np.array([[float(r[f"sep_{j}"]) for j in range(sep.shape[1])]
                     for r in rows if int(r["game"]) not in skipped])
    lines = [f"R^21 curvature field, {len(games)} shortlisted games x {sep.shape[1]} spaces.",
             "phi per member, rule means taken afterwards; R3 (n=1) has no within-rule "
             "spread.", "",
             f"{'game':>6} {'t0':>4} {'r21_min':>9} {'r21_mean':>9} {'disp_min':>9} "
             f"{'disp_mean':>10}"]
    for i, g in enumerate(games):
        lines.append(f"{g.game:>6} {g.t0:>4} {sep[i].min():>9.4f} {sep[i].mean():>9.4f} "
                     f"{disp[i].min():>9.4f} {disp[i].mean():>10.4f}")
    rho_min = spearmanr(sep.min(1), disp.min(1)).statistic
    rho_mean = spearmanr(sep.mean(1), disp.mean(1)).statistic
    lines += ["",
              f"Spearman rho over the shortlist, R^21 vs displacement: worst-space sep "
              f"{rho_min:.3f}, mean sep {rho_mean:.3f}.",
              f"Best worst-space game in R^21: {games[int(sep.min(1).argmax())].game}; "
              f"in the displacement field: {games[int(disp.min(1).argmax())].game}."]
    text = "\n".join(lines) + "\n"
    (OUT_DIR / "three_rules_r21_check.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
