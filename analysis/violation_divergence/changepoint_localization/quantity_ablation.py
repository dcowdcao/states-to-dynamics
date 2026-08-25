"""What the change-point score should be computed on: step length, norm, or curvature.

`theta_ablation.py` asked whether curvature can replace step length and answered on the argmax
hit rate alone, which is a forced choice and discards every position that is not the peak.
This asks the same of three quantities and reads each one both ways, because they disagree
about curvature.

  d      step length, ||z_{t+1} - z_t||, the study's statistic
  norm   the norm of the representation itself, latent norm in an SAE space
  theta  the turning angle, curvature proper

Both arms are spliced: the suffix after t0 is the source game's own, never regenerated.

**Default** -- the divergence branch tree, per-position AUROC over every candidate in a band,
ground truth cut at t0 for every arm alike. The three keys have different widths and different
move offsets, because the attention sink is dropped and an angle needs two consecutive
displacements; each gets the band that puts it on the same plies, so one chance level and one
window serve all three and only the quantity changes. `norm` here comes from
`curvature_discriminant/<space>_norm.npz` (`branch_norms.py`); without it the figure carries d
and theta and says so, rather than silently dropping a quantity.

**`--l0-matched`** -- the eight `l0_operating_point` picks, on
`illegal_move_detector`'s corpus and statistic: its 500 t0-stratified games from
`results/eval_games.csv`, the pooled illegal branch against the SAME source game's legal
control, ranked on the score at that game's own t0. That is deliberate rather than incidental:
**`d` here is the same number as panel (a) of `illegal_move_detector/figures/
sae_detector_overview.png`**, asserted against `auroc_by_space_l0_matched.csv` on every run, so
the two figures read on one axis and this one adds the two quantities that figure does not
carry. Each quantity keeps its own band -- the statistic reads one position per game, its own
t0, and every band covers the t0 range, so unlike the per-position AUROC nothing here depends
on the bands agreeing.

Both signs are run for the challengers and the better is reported. That is not a free
parameter: a label-free detector has to fix the direction in advance, so a quantity whose sign
moves across spaces is unusable however well it scores, and `sign` is a column for exactly
that reason. `d` on the detector corpus is held at the direction this study fixed for it --
shorter is illegal -- because that is what makes its bar the detector's own number; freeing it
would move exactly one space, CosineSpaDE, from 0.498 to 0.502, both chance.

Writes results/quantity_ablation_<variant>.csv and figures/quantity_ablation_<variant>.png.

Usage: PYTHONPATH=src python
           analysis/violation_divergence/changepoint_localization/quantity_ablation.py
           [--l0-matched] [--seed 42] [--camera-ready]
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
    MUTED, NULL, OKABE_ITO, REFERENCE_SPACE, apply_style, letters_below, save_figure)
from spade.l0_operating_point import roster
from spade.move_legality import localize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from changepoint import (  # noqa: E402  (sibling module, not an installed package)
    CAND_OFFSET, auroc, by_game, candidates, ci, out_of_fold, profile, rates, resamples, scan,
    take, threshold)
from measure import POOL_ARMS, npz_stem  # noqa: E402
from measure_regenerated import M, N_LEGAL, PAIRED, SPACES  # noqa: E402

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
    """The straightness legal games, which is what "normal at this move index" means.

    A pick's file is the twin `analysis/straightness/l0_matched_profile.py` writes, with the
    same five-class row layout as the canonical one, so the row indices are shared.
    """
    stem = ("resid_post_5" if space == REFERENCE_SPACE else
            space if space in roster.SPACES else f"sae_{space}")
    return np.load(study_dir("straightness") / "geometry" /
                   f"{stem}_raw.npz")[key].astype(np.float64)


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


def series(space: str, key: str):
    """The scored array for one (space, quantity), or None where it was never computed."""
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
    # Ground truth stops at t0, and the cut lands on every arm alike so the two classes keep
    # identical move-index marginals.
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


def row(space: str, key: str, corpus: dict, seed: int) -> dict | None:
    x = series(space, key)
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
    out = {"space": space, "quantity": key,
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


def detector_table(seed: int) -> pd.DataFrame:
    """The three quantities under `illegal_move_detector`'s corpus and at-t0 statistic.

    Imported rather than reimplemented: `d` has to come out equal to that study's bar, not
    close to it, so the corpus (`eval_rows`), the arms, the standardization and the AUROC all
    come from the code that wrote `auroc_by_space_l0_matched.csv`, and `main` asserts the
    result against that file.
    """
    # Local: the detector study is a sibling script directory, not an installed package, and
    # both directories carry a `make_figures` that must not shadow each other on import.
    sys.path.insert(0, str(HERE.parent / "illegal_move_detector"))
    import detector
    from figure_auroc_l0_matched import BOOT_SEED, eval_rows

    base = detector.load()
    sel = detector.arms(base, seed)
    keep = eval_rows(base, sel)
    draws = detector.resamples(len(keep), BOOT_SEED)
    rows = []
    for space in [REFERENCE_SPACE] + roster.SPACES:
        for key in BANDS:
            x = detector_series(detector, space, key)
            # `d` takes that study's own profile file, not the twin beside it: a pick's two
            # legal-block files differ by encode noise, and `d` has to reproduce its bar.
            mu, sd = (detector.legal_profile(base["held_out_legal"], space) if key == "d" else
                      profile(legal_geometry(space, key)[detector_legal(base)]))
            cand = candidates(x.shape[1], detector.M)
            offset = CAND_OFFSET - (1 if key == "norm" else 0)
            # `d` is not on trial for direction: this study fixed it long ago, and holding it
            # there is what makes this bar BE the detector's. Each challenger still gets its
            # best sign, which is the only asymmetry here and the one that favours them.
            best = None
            for sign in ((-1.0,) if key == "d" else (-1.0, 1.0)):
                g = np.nan_to_num(sign * (x - mu) / sd, nan=0.0)
                s = scan(g, cand, detector.M, PAIRED)
                at = lambda arm: s[sel[arm][keep], np.searchsorted(
                    cand + offset, base["t0"][sel[arm][keep]])]
                illegal, control, control2 = at("illegal"), at("control"), at("control2")
                score = detector.auroc(illegal, control)
                if best is None or score > best[0]:
                    best = (score, sign, illegal, control, control2)
            auroc, sign, illegal, control, control2 = best
            lo, hi = detector.ci(lambda d: detector.auroc(illegal[d], control[d]), draws)
            rows.append({"space": space, "quantity": key,
                         "l0": roster.L0[space] if space in roster.SPACES else "",
                         "sign": "-" if sign < 0 else "+", "n_games": len(keep),
                         "auroc_position": auroc, "auroc_position_lo": lo,
                         "auroc_position_hi": hi,
                         "auroc_null": detector.auroc(control2, control)})
    return pd.DataFrame(rows)


def detector_series(detector, space: str, key: str) -> np.ndarray:
    """One quantity over the detector's branch rows, from `extract_geometry.py`."""
    if key == "d":
        return detector.load(space)["d"]
    stem = "theta_norm" + ("" if space == REFERENCE_SPACE else f"_{space}")
    return np.load(detector.DIR / f"{stem}.npz")[key].astype(np.float64)


def detector_legal(base: dict) -> np.ndarray:
    """The straightness legal games that no scored trajectory is, for the profile fit."""
    held = base["held_out_legal"]
    return np.setdiff1d(np.arange(N_LEGAL), held[held >= 0])


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
    ax.set_xlim(-0.7, len(spaces) + 1.0)       # a slot past the last group for the chance label
    ax.set_ylabel(ylabel)
    ax.text(len(spaces) - 0.45, chance, "chance", color=NULL["color"], fontsize=6.8,
            ha="left", va="center", bbox={"fc": "white", "ec": "none", "pad": 1.2})
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)


def draw(table: pd.DataFrame, name: str, camera_ready: bool) -> Path:
    """One panel of AUROC. What the bars mean, what orders the spaces, and the localization
    half of the spliced table all live in QUANTITY_ABLATION.md, which is this figure's
    caption; `hit` stays in the CSV."""
    keys = [k for k in BANDS if (table.quantity == k).any()]
    order = (table[table.quantity == "d"].sort_values("auroc_position", ascending=False)
             .space.tolist())
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    panel(ax, table, "auroc_position", "AUROC", 0.5, 0, 1.0, order, keys)
    ax.legend(loc="upper right", fontsize=6.8, frameon=False, handlelength=1.4,
              labelspacing=0.3)
    fig.tight_layout()
    letters_below(fig, [ax])
    return save_figure(fig, HERE / "figures" / name, camera_ready)


def assert_matches_detector(table: pd.DataFrame) -> None:
    """`d` must BE `illegal_move_detector`'s bar, not merely agree with it."""
    theirs = pd.read_csv(HERE.parent / "illegal_move_detector" / "results" /
                         "auroc_by_space_l0_matched.csv").set_index("space").auroc
    ours = table[table.quantity == "d"].set_index("space").auroc_position
    gap = (ours - theirs.loc[ours.index]).abs().max()
    assert gap < 5e-6, (f"d differs from auroc_by_space_l0_matched.csv by {gap:.2e}; "
                        "the two figures are no longer one statistic on one corpus")
    print(f"d matches sae_detector_overview (a) to {gap:.1e}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l0-matched", dest="l0_matched", action="store_true",
                        help="the eight l0_operating_point picks, on illegal_move_detector's "
                             "500 games and its at-t0 statistic")
    parser.add_argument("--camera-ready", dest="camera_ready", action="store_true")
    args = parser.parse_args()

    if args.l0_matched:
        spaces = [REFERENCE_SPACE] + roster.SPACES
        table = detector_table(args.seed)
        assert_matches_detector(table)
        name, values = "quantity_ablation_l0_matched", ["auroc_position", "sign"]
    else:
        spaces = SPACES
        corpus = spliced_corpus(args.seed)
        corpus["fit"] = np.asarray(corpus["fit"])
        print(f"spliced: {len(corpus['t0'])} illegal rows, "
              f"{sum(len(r) for r in corpus['legal'])} legal rows, {len(MOVES)} plies "
              f"(moves {MOVES[0]}..{MOVES[-1]}), chance hit {1 / len(MOVES):.4f}", flush=True)
        rows = [r for s in spaces for k in BANDS
                if (r := row(s, k, corpus, args.seed)) is not None
                or print(f"  {s} / {k}: not computed on this tree", flush=True)]
        table = pd.DataFrame([r for r in rows if r])
        missing = [k for k in BANDS if not (table.quantity == k).any()]
        if missing:
            print(f"\nNOT SCORED on the spliced tree: {missing} "
                  f"-- run curvature_discriminant/branch_norms.py to add them")
        name, values = "quantity_ablation_spliced", ["auroc_position", "hit", "sign"]

    (HERE / "results").mkdir(exist_ok=True)
    csv = HERE / "results" / f"{name}.csv"
    table.round(5).to_csv(csv, index=False)
    apply_style(args.camera_ready)
    print(f"\nwrote {csv}\nwrote {draw(table, name, args.camera_ready)}")
    print(table.pivot(index="space", columns="quantity", values=values)
               .loc[spaces].round(3).to_string())


if __name__ == "__main__":
    main()
