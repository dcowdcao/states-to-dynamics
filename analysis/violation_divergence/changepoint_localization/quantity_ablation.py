"""What the change-point score should be computed on: step length, norm, or curvature.

`theta_ablation.py` asked whether curvature can replace step length and answered on the argmax
hit rate alone, which is a forced choice and discards every position that is not the peak.
This asks the same of three quantities and reads each one both ways, because they disagree
about curvature.

  d      step length, ||z_{t+1} - z_t||, the study's statistic
  norm   the norm of the representation itself, latent norm in an SAE space
  theta  the turning angle, curvature proper

Both suffix regimes. `--variant regenerated` scores the straightness violation corpus, whose
suffix is a fresh legal continuation, against the held-out odd-index legal games.
`--variant spliced` scores the divergence branch tree, whose suffix is the source game's
verbatim -- so ground truth stops at t0 and the position set is cut there, for every arm
alike, exactly as the thresholded read-out cuts it.

The three keys have different widths and different move offsets, because the attention sink is
dropped and an angle needs two consecutive displacements; each gets the band that puts it on
the same plies, so one chance level and one window serve all three and only the quantity
changes.

`norm` on the spliced tree comes from `curvature_discriminant/<space>_norm.npz`
(`branch_norms.py`). If that has not been run, the spliced figure carries d and theta and says
so, rather than silently dropping a variant.

`--l0-matched` swaps the seed-42 roster for the eight `l0_operating_point` picks, one
checkpoint per architecture at the shared L0. The canonical roster's L0s span 32.5 to 1,808,
so its space ORDERING confounds the architecture with its sparsity -- which is what the
by-space figures in this directory were redrawn for. It does not touch the ablation's own
claim: d vs norm vs theta is a within-space comparison, three bars in one group off one
corpus, and matching L0 across architectures cannot move it.

Both signs are run and the better is reported. That is not a free parameter: a label-free
detector has to fix the direction in advance, so a quantity whose sign moves across spaces is
unusable however well it scores, and `sign` is a column for exactly that reason.

Writes results/quantity_ablation_<variant>.csv and figures/quantity_ablation_<variant>.png.

Usage: PYTHONPATH=src python analysis/violation_divergence/changepoint_localization/quantity_ablation.py
           [--variant regenerated] [--camera-ready]
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
    INK, MUTED, NULL, OKABE_ITO, REFERENCE_SPACE, apply_style, save_figure)
from spade.l0_operating_point import roster
from spade.move_legality import localize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from changepoint import (  # noqa: E402  (sibling module, not an installed package)
    CAND_OFFSET, auroc, by_game, ci, out_of_fold, profile, rates, resamples, scan, take,
    threshold)
from measure import POOL_ARMS, npz_stem  # noqa: E402
from measure_regenerated import M, N_LEGAL, NULL_POOL, PAIRED, SPACES, VIOLATION  # noqa: E402

HERE = Path(__file__).resolve().parent
MOVES = np.arange(14, 47)          # the 33 plies all three keys can scan
# (candidate step indices, move offset) per key. `d[a]` spans positions a+1..a+2 and `theta[a]`
# pivots at a+2, but `norm[a]` IS position a+1, so its offset is one smaller.
BANDS = {"d": (np.arange(12, 45), CAND_OFFSET), "norm": (np.arange(13, 46), CAND_OFFSET - 1),
         "theta": (np.arange(12, 45), CAND_OFFSET)}
LABEL = {"d": "step length $d$", "norm": "representation norm $\\|z\\|$",
         "theta": "curvature angle $\\theta$"}
# Hue means the scored quantity here, not the architecture family: the quantity IS the
# comparison. Hatch carries it too, so the three stay separable in grayscale.
STYLE = {"d": (OKABE_ITO[0], None), "norm": (OKABE_ITO[1], "///"),
         "theta": (OKABE_ITO[2], "...")}
BAR = 0.26


def legal_geometry(space: str, key: str) -> np.ndarray:
    """The straightness geometry: the profile for both regimes, and the regenerated series.

    A pick's twin is written by `analysis/straightness/l0_matched_profile.py` and carries the
    same five-class row layout as the canonical file, so the corpus indices are shared.
    """
    stem = ("resid_post_5" if space == REFERENCE_SPACE else
            space if space in roster.SPACES else f"sae_{space}")
    return np.load(study_dir("straightness") / "geometry" /
                   f"{stem}_raw.npz")[key].astype(np.float64)


def regenerated_corpus(seed: int) -> dict:
    games = torch.load(study_dir("straightness") / "games" / "games.pt", weights_only=True)
    block = np.arange(N_LEGAL) + VIOLATION * N_LEGAL
    src, t0 = games["src_game"][block].numpy(), games["t0"][block].numpy()
    keep = (src % 2 == 1) & (t0 >= MOVES[0]) & (t0 <= MOVES[-1])
    return {"illegal": [block[keep]], "legal": [NULL_POOL], "t0": t0[keep],
            "game": [src[keep], NULL_POOL], "fit": np.arange(0, N_LEGAL, 2), "cut": False}


def spliced_corpus(seed: int) -> dict:
    """Four disjoint arms of the branch tree, and the profile fit that excludes their games."""
    meta = dict(np.load(study_dir("divergence") / "curvature_discriminant" / "meta.npz"))
    torch_meta = {k: torch.from_numpy(v) for k, v in meta.items() if k != "held_out_legal"}
    arms = {k: v.numpy() for k, v in localize.arms(torch_meta, seed).items()}
    held = set(meta["held_out_legal"].tolist())
    game = meta["game_id"][arms["reference"]]
    return {"illegal": [arms["illegal"]], "legal": [arms[a] for a in POOL_ARMS[1:]],
            "t0": meta["t0"][arms["reference"]], "game": [game] * len(POOL_ARMS),
            "fit": [i for i in range(N_LEGAL) if i not in held], "cut": True}


def series(space: str, key: str, variant: str):
    """The scored array for one (space, quantity), or None where it was never computed."""
    if variant == "regenerated":
        return legal_geometry(space, key)
    directory = study_dir("divergence") / "curvature_discriminant"
    if key == "norm":
        path = directory / f"{npz_stem(space)}_norm.npz"
        return np.load(path)["norm"].astype(np.float64) if path.exists() else None
    return np.load(directory / f"{npz_stem(space)}.npz")[key].astype(np.float64)


def score_one(x, fit, cand, offset, sign, corpus):
    """(hit per game, pooled position score, label, cluster id) for one sign."""
    mu, sd = profile(fit[:, :x.shape[1]] if fit.ndim > 1 else fit)
    g = np.nan_to_num(sign * (x - mu) / sd, nan=0.0)   # theta is NaN at a zero-length step
    move, t0 = cand + offset, corpus["t0"]
    rows = corpus["illegal"] + corpus["legal"]
    blocks = [scan(g[r], cand, M, PAIRED) for r in rows]
    # Ground truth stops at t0 on the spliced tree, and the cut lands on every arm alike so the
    # two classes keep identical move-index marginals. A legal block of a different length --
    # the regenerated pool -- carries no t0, so it is scored over the whole band.
    def mask(block, i):
        if not corpus["cut"] or len(block) != len(t0):
            return np.ones(block.shape, dtype=bool)
        return move[None, :] <= t0[:, None]

    keeps = [mask(b, i) for i, b in enumerate(blocks)]
    labels = [(move[None, :] == t0[:, None]) if i == 0 else np.zeros(b.shape, dtype=bool)
              for i, b in enumerate(blocks)]
    hit = cand[blocks[0].argmax(1)] + offset == t0
    return (hit,
            np.concatenate([b[k] for b, k in zip(blocks, keeps)]),
            np.concatenate([l[k] for l, k in zip(labels, keeps)]),
            np.concatenate([np.broadcast_to(g_[:, None], b.shape)[k]
                            for g_, b, k in zip(corpus["game"], blocks, keeps)]))


def row(space: str, key: str, variant: str, corpus: dict, seed: int) -> dict | None:
    x = series(space, key, variant)
    if x is None:
        return None
    cand, offset = BANDS[key]
    fit = legal_geometry(space, key)[corpus["fit"]]
    both = {s: score_one(x, fit, cand, offset, s, corpus) for s in (-1.0, 1.0)}
    pick = lambda v: auroc(v[1][v[2]], v[1][~v[2]])
    sign = max(both, key=lambda s: pick(both[s]))
    hit, score, label, game = both[sign]

    order, counts = by_game(game)
    draws, game_draws = resamples(len(counts), seed), resamples(len(hit), seed)
    _, predicted = out_of_fold(score, label, game, seed=seed)
    at = lambda f: (lambda d: f(take(order, counts, d)))
    out = {"space": space, "quantity": key, "variant": variant,
           "sign": "-" if sign < 0 else "+", "n_games": len(hit), "n_positions": len(score),
           "n_positive": int(label.sum()), "chance_hit": 1.0 / len(cand),
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
          spaces: list[str], keys: list[str]) -> None:
    ax.axhline(chance, color=NULL["color"], ls=NULL["ls"], lw=1.2, zorder=2)
    offsets = np.linspace(-BAR, BAR, len(keys)) if len(keys) > 1 else [0.0]
    for k, key in enumerate(keys):
        block = table[table.quantity == key].set_index("space").loc[spaces]
        colour, hatch = STYLE[key]
        x = np.arange(len(spaces)) + offsets[k]
        ax.bar(x, block[value], width=BAR, color=colour, hatch=hatch, edgecolor="white",
               linewidth=0.5, zorder=3, label=LABEL[key] if index == 0 else None)
        ax.errorbar(x, block[value],
                    yerr=[block[value] - block[f"{value}_lo"], block[f"{value}_hi"] - block[value]],
                    color=MUTED, elinewidth=0.8, capsize=1.6, ls="none", zorder=4)

    styles = roster.styles()
    name = lambda s: "raw L5 (ref.)" if s == REFERENCE_SPACE else styles[s]["short"]
    ax.set_xticks(range(len(spaces)), [name(s) for s in spaces], fontsize=6.8, rotation=35,
                  ha="right", rotation_mode="anchor")
    ax.set_ylim(0, top)
    ax.set_xlim(-0.7, len(spaces) + 0.45)      # a slot past the last group for the chance label
    ax.set_ylabel(ylabel)
    ax.text(len(spaces) - 0.55, chance, "chance", color=NULL["color"], fontsize=6.8,
            ha="left", va="center", bbox={"fc": "white", "ec": "none", "pad": 1.2})
    ax.text(0.0, 1.02, f"({chr(97 + index)})", va="bottom", transform=ax.transAxes,
            color=INK, fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)


def draw(table: pd.DataFrame, variant: str, camera_ready: bool, suffix: str = "") -> Path:
    keys = [k for k in BANDS if (table.quantity == k).any()]
    order = (table[table.quantity == "d"].sort_values("auroc_position", ascending=False)
             .space.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    against = ("legal move" if variant == "regenerated" else
               "legal move at $t \\leq t_0$")
    panel(axes[0], table, "auroc_position",
          f"per-position AUROC, illegal vs {against}\n($\\uparrow$ better)",
          0.5, 0, 1.0, order, keys)
    panel(axes[1], table, "hit", "localization: argmax lands on $t_0$\n($\\uparrow$ better)",
          float(table.chance_hit.iloc[0]), 1, 0.55, order, keys)
    axes[0].legend(loc="upper right", fontsize=6.8, frameon=False, handlelength=1.4,
                   labelspacing=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.text(0.005, 0.985, "hue and hatch = the scored quantity; spaces are ordered by $d$'s "
             f"per-position AUROC, raw layer 5 wherever it lands ({variant} suffix)",
             fontsize=7, color=MUTED, va="top")
    return save_figure(fig, HERE / "figures" / f"quantity_ablation_{variant}{suffix}",
                       camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("regenerated", "spliced"), default="regenerated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l0-matched", dest="l0_matched", action="store_true",
                        help="the eight l0_operating_point picks instead of the seed-42 roster")
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    spaces = [REFERENCE_SPACE] + roster.SPACES if args.l0_matched else SPACES
    suffix = "_l0_matched" if args.l0_matched else ""
    build = regenerated_corpus if args.variant == "regenerated" else spliced_corpus
    corpus = build(args.seed)
    corpus["fit"] = np.asarray(corpus["fit"])
    print(f"{args.variant}: {len(corpus['t0'])} illegal rows, "
          f"{sum(len(r) for r in corpus['legal'])} legal rows, {len(MOVES)} plies "
          f"(moves {MOVES[0]}..{MOVES[-1]}), chance hit {1 / len(MOVES):.4f}", flush=True)

    rows = [r for s in spaces for k in BANDS
            if (r := row(s, k, args.variant, corpus, args.seed)) is not None
            or print(f"  {s} / {k}: not computed on this tree", flush=True)]
    table = pd.DataFrame([r for r in rows if r])
    missing = [k for k in BANDS if not (table.quantity == k).any()]
    if missing:
        print(f"\nNOT SCORED on the {args.variant} tree: {missing} "
              f"-- run curvature_discriminant/branch_norms.py to add them")

    (HERE / "results").mkdir(exist_ok=True)
    csv = HERE / "results" / f"quantity_ablation_{args.variant}{suffix}.csv"
    table.round(5).to_csv(csv, index=False)
    apply_style(args.camera_ready)
    print(f"\nwrote {csv}\nwrote {draw(table, args.variant, args.camera_ready, suffix)}")
    print(table.pivot(index="space", columns="quantity",
                     values=["auroc_position", "hit", "sign"]).loc[spaces].round(3).to_string())


if __name__ == "__main__":
    main()
