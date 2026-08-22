"""Step length, activation norm and curvature angle through the identical pipeline.

`theta_ablation.py` asked whether curvature can replace step length and answered on the
argmax hit rate alone, which is a forced choice and throws away every position that is not
the peak. This asks the same question of three quantities at once and reads each one both
ways -- the hit rate it always had, and the per-position AUROC the thresholded read-out
added -- because those disagree about curvature.

  d      step length, ||z_{t+1} - z_t||, the study's statistic
  norm   the norm of the representation itself, latent norm in an SAE space
  theta  the turning angle, curvature proper

The three keys have different widths and different move offsets -- the attention sink is
dropped, and an angle needs two consecutive displacements -- so each gets the band that puts
it on moves 14..46. One chance level, one window, one set of arms; only the scored quantity
changes.

Both signs are run and the better is reported. That is not a free parameter: a label-free
detector has to fix the direction in advance, so a quantity whose sign moves across spaces is
unusable however well it scores, and `sign` is a column for exactly that reason.

Writes results/quantity_ablation.csv and figures/quantity_ablation.png.

Usage: PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/quantity_ablation.py [--camera-ready]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from spade.common.study import study_dir
from spade.figures.paper_style import (
    INK, MUTED, NULL, OKABE_ITO, REFERENCE_SPACE, SPACE_STYLE, apply_style, save_figure)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from changepoint import (  # noqa: E402  (sibling module, not an installed package)
    auroc, by_game, ci, out_of_fold, profile, rates, resamples, scan, take, threshold)
from measure_regenerated import M, N_LEGAL, NULL_POOL, PAIRED, SPACES, VIOLATION  # noqa: E402

HERE = Path(__file__).resolve().parent
MOVES = np.arange(14, 47)          # the 33 plies all three keys can scan
# (candidate step indices, move offset) per key: `d[a]` spans positions a+1..a+2 and `theta[a]`
# pivots at a+2, but `norm[a]` IS position a+1, so its offset is one smaller.
BANDS = {"d": (np.arange(12, 45), 2), "norm": (np.arange(13, 46), 1),
         "theta": (np.arange(12, 45), 2)}
LABEL = {"d": "step length $d$", "norm": "representation norm $\\|z\\|$",
         "theta": "curvature angle $\\theta$"}
# Hue means the scored quantity here, not the architecture family: the quantity IS the
# comparison. Hatch carries it too, so the three stay separable in grayscale.
STYLE = {"d": (OKABE_ITO[0], None), "norm": (OKABE_ITO[1], "///"),
         "theta": (OKABE_ITO[2], "...")}
BAR = 0.26


def corpus():
    games = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)
    block = np.arange(N_LEGAL) + VIOLATION * N_LEGAL
    src, t0 = games["src_game"][block].numpy(), games["t0"][block].numpy()
    keep = (src % 2 == 1) & (t0 >= MOVES[0]) & (t0 <= MOVES[-1])
    return block[keep], src[keep], t0[keep]


def arrays(space: str, key: str) -> np.ndarray:
    stem = "resid_post_5" if space == REFERENCE_SPACE else f"sae_{space}"
    return np.load(study_dir("straightness") / "geometry" /
                   f"{stem}_raw.npz")[key].astype(np.float64)


def score_one(x, cand, offset, sign, viol, src, t0):
    """(hit per game, pooled position score, its label, its cluster id)."""
    mu, sd = profile(x[np.arange(0, N_LEGAL, 2)])
    g = np.nan_to_num(sign * (x - mu) / sd, nan=0.0)   # theta is NaN at a zero-length step
    s_ill, s_pool = scan(g[viol], cand, M, PAIRED), scan(g[NULL_POOL], cand, M, PAIRED)
    hit = cand[s_ill.argmax(1)] + offset == t0
    return (hit,
            np.concatenate([s_ill.ravel(), s_pool.ravel()]),
            np.concatenate([(cand[None, :] + offset == t0[:, None]).ravel(),
                            np.zeros(s_pool.size, dtype=bool)]),
            np.concatenate([np.repeat(src, len(cand)), np.repeat(NULL_POOL, len(cand))]))


def row(space: str, key: str, viol, src, t0, seed: int) -> dict:
    cand, offset = BANDS[key]
    x = arrays(space, key)
    both = {s: score_one(x, cand, offset, s, viol, src, t0) for s in (-1.0, 1.0)}
    sign = max(both, key=lambda s: auroc(both[s][1][both[s][2]], both[s][1][~both[s][2]]))
    hit, score, label, game = both[sign]

    order, counts = by_game(game)
    draws = resamples(len(counts), seed)
    game_draws = resamples(len(hit), seed)
    _, predicted = out_of_fold(score, label, game, seed=seed)
    at = lambda f: (lambda d: f(take(order, counts, d)))
    out = {"space": space, "quantity": key, "sign": "-" if sign < 0 else "+",
           "n_games": len(hit), "n_positions": len(score), "n_positive": int(label.sum()),
           "chance_hit": 1.0 / len(cand),
           "auroc_position": auroc(score[label], score[~label]),
           "hit": float(hit.mean()), "threshold": threshold(score, label)}
    out |= dict(zip(("auroc_position_lo", "auroc_position_hi"),
                    ci(at(lambda i: auroc(score[i][label[i]], score[i][~label[i]])), draws)))
    out |= dict(zip(("hit_lo", "hit_hi"), ci(lambda d: float(hit[d].mean()), game_draws)))
    out |= rates(predicted, label)
    out |= dict(zip(("balanced_accuracy_lo", "balanced_accuracy_hi"),
                    ci(at(lambda i: rates(predicted[i], label[i])["balanced_accuracy"]), draws)))
    return out


def panel(ax, table, value: str, ylabel: str, chance: float, index: int, top: float,
          chance_label: str, spaces: list[str]) -> None:
    ax.axhline(chance, color=NULL["color"], ls=NULL["ls"], lw=1.2, zorder=2)
    for k, key in enumerate(BANDS):
        block = table[table.quantity == key].set_index("space").loc[spaces]
        colour, hatch = STYLE[key]
        x = np.arange(len(spaces)) + (k - 1) * BAR
        ax.bar(x, block[value], width=BAR, color=colour, hatch=hatch, edgecolor="white",
               linewidth=0.5, zorder=3, label=LABEL[key] if index == 0 else None)
        ax.errorbar(x, block[value],
                    yerr=[block[value] - block[f"{value}_lo"], block[f"{value}_hi"] - block[value]],
                    color=MUTED, elinewidth=0.8, capsize=1.6, ls="none", zorder=4)

    label = lambda s: "raw L5 (ref.)" if s == REFERENCE_SPACE else SPACE_STYLE[s]["short"]
    ax.set_xticks(range(len(spaces)), [label(s) for s in spaces], fontsize=6.8, rotation=35,
                  ha="right", rotation_mode="anchor")
    ax.set_ylim(0, top)
    # A slot past the last group, so the chance label never lands on a bar.
    ax.set_xlim(-0.7, len(spaces) + 0.45)
    ax.set_ylabel(ylabel)
    ax.text(len(spaces) - 0.55, chance, chance_label, color=NULL["color"], fontsize=6.8,
            ha="left", va="center", bbox={"fc": "white", "ec": "none", "pad": 1.2})
    ax.text(0.0, 1.02, f"({chr(97 + index)})", va="bottom", transform=ax.transAxes,
            color=INK, fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)


def draw(table: pd.DataFrame, camera_ready: bool) -> Path:
    order = (table[table.quantity == "d"].sort_values("auroc_position", ascending=False)
             .space.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    panel(axes[0], table, "auroc_position",
          "per-position AUROC, illegal vs legal move\n($\\uparrow$ better)",
          0.5, 0, 1.0, "chance", order)
    panel(axes[1], table, "hit",
          "localization: argmax lands on $t_0$\n($\\uparrow$ better)",
          float(table.chance_hit.iloc[0]), 1, 0.55, "chance", order)
    axes[0].legend(loc="upper right", fontsize=6.8, frameon=False, handlelength=1.4,
                   labelspacing=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.text(0.005, 0.985, "hue and hatch = the scored quantity; spaces are ordered by $d$'s "
             "per-position AUROC, raw layer 5 wherever it lands", fontsize=7, color=MUTED,
             va="top")
    return save_figure(fig, HERE / "figures" / "quantity_ablation", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    viol, src, t0 = corpus()
    print(f"{len(t0)} violations + {len(NULL_POOL)} legal games, {len(MOVES)} plies "
          f"(moves {MOVES[0]}..{MOVES[-1]}), chance hit {1 / len(MOVES):.4f}", flush=True)
    table = pd.DataFrame([row(s, k, viol, src, t0, args.seed)
                          for s in SPACES for k in BANDS
                          if not print(f"  {s} / {k}", flush=True)])

    (HERE / "results").mkdir(exist_ok=True)
    csv = HERE / "results" / "quantity_ablation.csv"
    table.round(5).to_csv(csv, index=False)
    apply_style(args.camera_ready)
    print(f"\nwrote {csv}\nwrote {draw(table, args.camera_ready)}")
    wide = table.pivot(index="space", columns="quantity",
                       values=["auroc_position", "hit", "sign"]).loc[SPACES]
    print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
