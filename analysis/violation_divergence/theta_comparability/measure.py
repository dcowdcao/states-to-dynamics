"""Is the straightness study's "mean turning angle" the fork study's, on one axis?

Both pipelines call `spade.straightness.geometry.temporal_geometry` on layer-5
resid_post and both report degrees, but they hand it different sequences. The
straightness study passes every ply (positions 1..58). The fork study passes
`fork_plane.positions()`, every STRIDE=2 plies at t0's parity, and then takes the
`mean θ` component of the ReStraV descriptor over a W=8 window of those.

An angle over a 2-ply step is a different geometric quantity from one over a
1-ply step, and here the difference is not a nuisance: OthelloGPT's layer-5 board
code is mover-relative, so the frame inverts every ply and the per-ply path is a
near-reversal sawtooth. Sampling every second ply stays inside one mover's frame
and removes it. This module measures both rules on the same activation rows and
reports the level, the per-class contrast, its AUROC, and the two rules'
correlation over games.

Usage: python analysis/violation_divergence/theta_comparability/measure.py
"""

import numpy as np
import torch

from spade.common.study import study_dir
from spade.straightness.build_games import CLASSES
from spade.straightness.classify import boot_ci
from spade.straightness.geometry import (
    RESTRAV_FEATURE_NAMES, temporal_geometry, train_mask)
from spade.violation_divergence import curvature_plane as dsc
from spade.violation_divergence.branches import FIGURE_GAME
from spade.violation_divergence.fork_plane import RAW_SPACE, STRIDE

FIRST_POS, LAST_POS = 1, 58  # position 0 is the attention sink, dropped by both
MEAN_THETA = RESTRAV_FEATURE_NAMES.index("mean θ")
MEAN_D = RESTRAV_FEATURE_NAMES.index("mean d")
RULES = ("per-ply (straightness)", f"stride-{STRIDE} (fork)")
CACHE = study_dir("divergence") / "theta_comparability" / "per_game.npz"


def positions(stride: int, parity: int) -> np.ndarray:
    """Board positions a rule samples: every `stride` plies from FIRST_POS+parity."""
    p = np.arange(FIRST_POS, LAST_POS + 1)
    return p[(p - FIRST_POS - parity) % stride == 0]


def geometry(Z: torch.Tensor, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean step length, per-position turning angles) over the sampled path.

    `Z` is indexed by board position, so `pos - FIRST_POS` selects into it. This
    is `temporal_geometry` on exactly the sequence each pipeline hands it. Angle
    column k is vertexed at board position `pos[k + 1]`, which is where both
    pipelines' profile panels plot it.
    """
    d, theta = temporal_geometry(Z[:, pos - FIRST_POS, :])
    return d.mean(1).numpy(), theta.numpy()


def per_game() -> dict[str, np.ndarray]:
    """Per-game mean d and mean θ under each rule, plus the per-class θ profile
    over board position, for all 5 classes of games.pt.

    Cached: the pass reads a 600 MB fp16 trajectory tensor one class at a time.
    """
    if CACHE.exists():
        return dict(np.load(CACHE))
    base = study_dir("straightness")
    cls = torch.load(base / "games" / "games.pt", weights_only=True)["cls"]
    traj = torch.load(base / "traj" / "resid_post_5.pt", weights_only=True, mmap=True)
    out = {"cls": cls.numpy()}
    for stride, parity, rule in [(1, 0, RULES[0]), (STRIDE, 0, RULES[1]),
                                 (STRIDE, 1, f"{RULES[1]} odd parity")]:
        pos = positions(stride, parity)
        d, mean_theta, mu, sem = [], [], [], []
        for c in range(len(CLASSES)):
            Z = traj[(cls == c).nonzero().flatten()].float()[:, FIRST_POS:LAST_POS + 1, :]
            dc, theta = geometry(Z, pos)
            n = (~np.isnan(theta)).sum(0)
            d.append(dc)
            mean_theta.append(np.nanmean(theta, axis=1))
            mu.append(np.nanmean(theta, axis=0))
            sem.append(np.nanstd(theta, axis=0) / np.sqrt(np.maximum(n, 1)))
        out[f"d {rule}"] = np.concatenate(d)
        out[f"θ {rule}"] = np.concatenate(mean_theta)
        out[f"vertex {rule}"] = pos[1:-1]
        out[f"profile {rule}"] = np.stack(mu)
        out[f"profile sem {rule}"] = np.stack(sem)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def delta_ci(g: dict, rule: str, ood: str, n_boot: int = 1000) -> tuple[float, float, float]:
    """class mean θ − legal mean θ, with a percentile bootstrap over games."""
    rng = np.random.default_rng(0)
    theta, cls = g[f"θ {rule}"], g["cls"]
    a, b = theta[cls == CLASSES.index(ood)], theta[cls == 0]
    reps = [a[rng.integers(len(a), size=len(a))].mean()
            - b[rng.integers(len(b), size=len(b))].mean() for _ in range(n_boot)]
    return (a.mean() - b.mean(), *np.percentile(reps, [2.5, 97.5]))


def auroc(g: dict, rule: str, ood: str) -> tuple[float, float, float]:
    """Straightness's own convention and split: score = mean θ of the OOD class,
    on the odd-index test half, so >0.5 means the OOD class is curvier."""
    from sklearn.metrics import roc_auc_score
    theta, cls = g[f"θ {rule}"], g["cls"]
    test = ~train_mask(len(cls)).numpy()
    idx = np.flatnonzero(test & ((cls == 0) | (cls == CLASSES.index(ood))))
    y = (cls[idx] != 0).astype(float)
    return (roc_auc_score(y, theta[idx]), *boot_ci(y, theta[idx]))


def levels(g: dict) -> None:
    print("1. the level, per class: per-game mean turning angle and step length,\n"
          f"   layer 5 raw, n = {len(g['cls']) // len(CLASSES)} games per class "
          "(mean ± SD over games)")
    rules = [*RULES, f"{RULES[1]} odd parity"]
    for quantity, unit in [("θ", " (deg)"), ("d", "")]:
        print(f"\n{'class':16}" + "".join(f"{r + unit:>32}" for r in rules))
        for c, name in enumerate(CLASSES):
            b = g["cls"] == c
            print(f"{name:16}" + "".join(
                f"{g[f'{quantity} {r}'][b].mean():>23.2f} ± {g[f'{quantity} {r}'][b].std():<6.2f}"
                for r in rules if f"{quantity} {r}" in g))


def contrasts(g: dict) -> None:
    print("\n2. the contrast the straightness figures state, under each rule.\n"
          "   Δ = class mean − legal mean (deg); AUROC on the odd-index test half,\n"
          "   >0.5 = curvier than legal (straightness's convention), 95% percentile\n"
          "   bootstrap over games, 1000 resamples.")
    print(f"{'class':16}" + "".join(f"{r:>46}" for r in RULES))
    for name in CLASSES[1:]:
        row = f"{name:16}"
        for rule in RULES:
            delta, dlo, dhi = delta_ci(g, rule, name)
            a, lo, hi = auroc(g, rule, name)
            row += (f"  Δ {delta:>+6.2f} [{dlo:+.2f},{dhi:+.2f}]"
                    f"  AUROC {a:.3f} [{lo:.3f},{hi:.3f}]")
        print(row)

    a, b = g[f"θ {RULES[0]}"], g[f"θ {RULES[1]}"]
    print(f"\n   over all {len(a)} games: corr(per-ply, stride-{STRIDE}) = "
          f"{np.corrcoef(a, b)[0, 1]:+.3f}; per-game difference "
          f"{np.mean(b - a):+.2f} ± {np.std(b - a):.2f} deg, "
          f"ratio {np.mean(b / a):.4f} ± {np.std(b / a):.4f}")


def shared_games() -> None:
    """The two studies' game populations overlap almost exactly, so the gap can be
    attributed to the sampling rule and to nothing else — not the game set, not
    the extraction pass, not the dtype."""
    s = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)
    b = torch.load(study_dir("divergence") / "branches" / "branches.pt", weights_only=True)
    key = {bytes(s["tokens"][i].numpy().astype("uint8")): int(i)
           for i in (s["cls"] == 0).nonzero().flatten()}
    pairs = [(key[k], int(j))
             for j in ((b["category"] == 0) & (b["is_control"] == 0)).nonzero().flatten()
             for k in [bytes(b["tokens"][j].numpy().astype("uint8"))] if k in key]
    si, di = torch.tensor([p[0] for p in pairs]), torch.tensor([p[1] for p in pairs])

    def load(path, idx):
        return torch.load(path, weights_only=True, mmap=True)[idx].float()[
            :, FIRST_POS:LAST_POS + 1, :]

    Zs = load(study_dir("straightness") / "traj" / "resid_post_5.pt", si)
    Zd = load(study_dir("divergence") / "acts" / "resid_post_5.pt", di)
    diff = (Zs - Zd).abs()
    print(f"\n3. the two studies share their games: {len(pairs)} of "
          f"{int((s['cls'] == 0).sum())} straightness legal games are also a fork\n"
          "   reference branch, token for token. Their layer-5 activations agree in "
          f"{int((diff.amax(dim=(1, 2)) == 0).sum())}/{len(pairs)}\n"
          f"   rows exactly (max |Δ| {diff.max():.1e} against mean ‖z‖ "
          f"{Zs.norm(dim=-1).mean():.1f}; fp16 batch-composition noise,\n"
          "   which extract_acts.spans documents). So the gap below is the sampling "
          "rule alone.")
    print(f"\n{'read from':26}" + "".join(f"{r + ' mean θ (deg)':>32}" for r in RULES))
    for name, Z in [("straightness traj tensor", Zs), ("divergence acts tensor", Zd)]:
        row = f"{name:26}"
        for stride in (1, STRIDE):
            _, theta = geometry(Z, positions(stride, 0))
            row += f"{np.nanmean(theta):>32.3f}"
        print(row)


def same_rows(gid: int, w: int) -> None:
    """Both rules on one game's nine branches — the same activation rows, end to
    end through each pipeline's own loader."""
    g = dsc.figure_game(gid, w)
    Z = g[RAW_SPACE]
    ref = int(np.flatnonzero((g["category"] == 0) & (g["is_control"] == 0))[0])
    k0 = int(np.flatnonzero(g["pos"] == g["t0"])[0])
    phi = dsc.windowed(Z, w)
    whole = dsc.descriptor(torch.from_numpy(Z[:, k0:]))

    base = study_dir("divergence")
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(base / "acts" / "resid_post_5.pt", weights_only=True, mmap=True)
    idx = (meta["game_id"] == gid).nonzero().flatten()
    d1, th1 = temporal_geometry(acts[idx].float()[:, FIRST_POS:LAST_POS + 1, :])

    print(f"\n4. the same nine branches of game {gid} (t0 = {g['t0']}), layer 5 raw,\n"
          "   through each pipeline's own code path")
    print(f"{'quantity':52}{'reference branch':>18}{'all 9 branches':>18}")
    print(f"{'straightness: per-ply mean θ (deg)':52}"
          f"{np.nanmean(th1[ref].numpy()):>18.2f}{np.nanmean(th1.numpy()):>18.2f}")
    print(f"{'fork, whole variant: R^21 mean θ (deg)':52}"
          f"{whole[ref, MEAN_THETA]:>18.2f}{whole[:, MEAN_THETA].mean():>18.2f}")
    print(f"{f'fork, windowed variant (w={w}): R^21 mean θ (deg)':52}"
          f"{np.nanmean(phi[ref, :, MEAN_THETA]):>18.2f}"
          f"{np.nanmean(phi[:, :, MEAN_THETA]):>18.2f}")
    print(f"{'straightness: per-ply mean d':52}"
          f"{d1[ref].mean():>18.2f}{d1.mean():>18.2f}")
    print(f"{'fork, whole variant: R^21 mean d':52}"
          f"{whole[ref, MEAN_D]:>18.2f}{whole[:, MEAN_D].mean():>18.2f}")
    print(f"\n   spread of the fork's whole-variant mean θ across the nine branches: "
          f"{whole[:, MEAN_THETA].ptp():.2f} deg")

    print(f"\n5. what one plotted point covers, at the fork's w={w} window")
    defined = np.flatnonzero(np.isfinite(phi[ref, :, MEAN_THETA]))
    for k in defined[:3]:
        span = g["pos"][k - w + 1:k + 1]
        print(f"   plotted at t = {g['pos'][k]:2d} (r = {g['pos'][k] - g['t0']:+3d}): "
              f"{w} sampled positions {span[0]}..{span[-1]} = {STRIDE * (w - 1)} plies, "
              f"{w - 2} angles vertexed at {span[1]}..{span[-2]}")
    print(f"   straightness plots its θ at the ply the turn is vertexed on "
          f"(profile_panel: index k -> position k+2), one angle per point.")


def main():
    g = per_game()
    levels(g)
    contrasts(g)
    shared_games()
    same_rows(FIGURE_GAME, dsc.W)


if __name__ == "__main__":
    main()
