"""At the spliced move, how well does each probe head separate illegal from legal?

One bar chart, over an evaluation set of EVAL_GAMES source games split evenly across the
five conditions -- legal, illegal, R1, R2, R3 -- so each condition is a disjoint group of
games and no game appears twice. Every bar is that group's score at its own t0 against the
legal group's, and t0 varies over the whole [10, 34] range within every group. The 2-class
head is read as P(illegal), the 4-class head as 1 - P(legal) and split by which rule the
branch actually broke, so a rule the probe cannot catch shows up as its own short bar
rather than being averaged away.

**The even split costs the pairing, and that is a real trade.** Scoring an illegal branch
against *the same game's* legal control at *the same* position held game identity and move
index fixed, so the contrast was the move alone; disjoint groups reintroduce both as
between-group variance, and 200 per group widens every interval against 3,923 pairs. The
paired figure is still computed and still in the CSV (`pairing` column) -- it is the number
README.md quotes -- so nothing is lost by drawing the split one.

AUROC is threshold-free, so a head's hit rate and the false alarms it pays for it are one
number rather than two bars. The legal-vs-legal null -- a game's legal control against the
second legal control it carries -- is still measured, since it is what makes the 0.5 line
empirical rather than assumed, but it is neither a bar nor a caption: it goes to the CSV
and to README.md, and the figure carries no text under its axes.

Usage: python analysis/move_legality/detection_auroc.py [--space raw] [--camera_ready]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from spade.common.paths import repo_root, results_dir
from spade.figures.paper_style import (
    CLASS_STYLE, MUTED, NULL, PROBE_CLASS_STYLE, apply_style, save_figure)
from spade.move_legality import localize

FIGURES = repo_root() / "analysis" / "move_legality" / "figures"
RESULTS = repo_root() / "analysis" / "move_legality" / "results"
PAPER_RESULTS = results_dir("fig02_probe_geometry_and_detection")   # the raw-space table, Fig 2c
RULES = localize.RULES
SEED = 42
GAP = 0.8            # blank slot between the two heads
EVAL_GAMES = 1000    # split evenly over the five conditions, so 200 games each
# The paired bars the combined figure draws are over the SAME games `illegal_move_detector`'s
# population figures score -- it writes them down, this reads them -- so the panels beside each
# other are one evaluation set and not two. The full-corpus paired rows stay in the CSV: they
# are what README.md quotes.
EVAL_GAMES_CSV = results_dir("shared") / "eval_games.csv"
N_BOOT = 1000
CONDITIONS = ["control", "illegal", *RULES]

ARM_STYLE = {
    "illegal": {**CLASS_STYLE["violation"], "label": "illegal / legal"},
    "control": NULL,   # the legal-vs-legal null: measured, printed in the note, not a bar
    **{r: {**PROBE_CLASS_STYLE[r], "label": r} for r in RULES},
}


def evaluation_split(meta: dict, selection: dict, seed: int = SEED) -> dict[str, np.ndarray]:
    """EVAL_GAMES games dealt evenly and disjointly to the five conditions, stratified by t0.

    Dealt *within* each t0 rather than over the pool: an equal quota per condition at every
    move index, so the five groups carry an identical t0 distribution -- same n, same counts
    at every value, and therefore the same median and mean -- rather than merely the same
    distribution in expectation. Move index is the strongest nuisance variable this tree has
    (score and probe output both move with it), so matching it exactly is what lets a
    difference between bars be read as the rule rather than as when in the game it happened.

    Only games carrying every condition are eligible, so a missing R3 cannot shrink a group.
    """
    game, t0 = meta["game_id"].numpy(), meta["t0"].numpy()
    common = set.intersection(*(set(game[selection[c]].tolist()) for c in CONDITIONS))
    reference = selection["reference"].numpy()
    eligible = reference[np.isin(game[reference], list(common))]
    values = np.unique(t0[eligible])
    quota, rng = EVAL_GAMES // (len(CONDITIONS) * len(values)), np.random.default_rng(seed)

    chosen = {c: [] for c in CONDITIONS}
    for value in values:
        at = game[eligible[t0[eligible] == value]]
        assert len(at) >= quota * len(CONDITIONS), f"t0={value}: {len(at)} games, need {quota * len(CONDITIONS)}"
        dealt = rng.permutation(np.sort(at))[:quota * len(CONDITIONS)]
        for i, condition in enumerate(CONDITIONS):
            chosen[condition].append(dealt[i * quota:(i + 1) * quota])

    out = {}
    for condition, blocks in chosen.items():
        want = set(np.concatenate(blocks).tolist())
        rows = selection[condition]
        out[condition] = rows[np.isin(game[rows], list(want))].numpy()
        assert len(out[condition]) == quota * len(values), condition
    return out


def unpaired(score: torch.Tensor, meta: dict, rows: np.ndarray, legal: np.ndarray,
             seed: int = SEED) -> dict:
    """AUROC at t0 of one condition's games against the legal group's, and a bootstrap CI.

    Unpaired, because the groups are disjoint by construction: the two arms are resampled
    independently rather than together, which is what a between-group contrast requires.
    """
    from spade.move_legality.probe import auroc

    at_t0 = score.gather(1, meta["t0"].unsqueeze(1)).squeeze(1)
    positive, negative = at_t0[rows], at_t0[legal]
    labels = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))])
    generator = torch.Generator().manual_seed(seed)
    draw = lambda n: torch.randint(n, (N_BOOT, n), generator=generator)
    boot = torch.tensor([auroc(torch.cat([positive[p], negative[q]]), labels)
                         for p, q in zip(draw(len(positive)), draw(len(negative)))])
    return {"auroc": auroc(torch.cat([positive, negative]), labels),
            "lo": float(boot.quantile(0.025)), "hi": float(boot.quantile(0.975)),
            "n_games": len(rows), "n_legal": len(negative)}


def shared_games(meta: dict, selection: dict) -> np.ndarray:
    """The games `illegal_move_detector`'s population figures score, read from its results.

    Not re-derived here. Those two panels and this one are read side by side, so the bars have
    to be over one evaluation set rather than two sets built by the same recipe -- that study's
    eligibility carries a held-out rule this one has no reason to reproduce, and a recipe
    copied into two files drifts. Their t0 is asserted equal, which is what would catch a
    stale file.
    """
    games = pd.read_csv(EVAL_GAMES_CSV)
    game, t0 = meta["game_id"].numpy(), meta["t0"].numpy()
    common = set.intersection(*(set(game[selection[c]].tolist()) for c in CONDITIONS))
    missing = set(games.game_id) - common
    assert not missing, f"{len(missing)} of the detector's games carry no full branch set here"
    reference = selection["reference"].numpy()
    at = dict(zip(game[reference].tolist(), t0[reference].tolist()))
    assert all(at[g] == t for g, t in zip(games.game_id, games.t0)), "t0 disagrees; stale file"
    return games.game_id.to_numpy()


def bar(head: str, score: torch.Tensor, arm: str, meta: dict, split: dict) -> dict:
    """One condition's AUROC against the legal group of the same evaluation split."""
    return {"head": head, "arm": arm, "style": ARM_STYLE[arm], "pairing": "disjoint groups",
            **unpaired(score, meta, split[arm], split["control"])}


def paired_bar(head: str, score: torch.Tensor, arm: str, meta: dict, selection: dict,
               games: np.ndarray | None = None) -> dict:
    """The arm the way the study has always scored it: one pair per game, same t0.

    `games` restricts it to a t0-stratified subset; without it, every game the tree carries,
    which is the number README.md quotes.
    """
    rows = selection[arm]
    if games is not None:
        rows = rows[np.isin(meta["game_id"].numpy()[rows], games)]
    null = selection["control2" if arm == "control" else "control"]
    label = "paired on game" if games is None else f"paired on game, n = {len(games)}"
    return {"head": head, "arm": arm, "style": ARM_STYLE[arm], "pairing": label,
            **localize.detection(score, meta, rows,
                                 localize.matched_control(meta, rows, null), seed=SEED)}


def bars(space: str, meta: dict, selection: dict, split: dict) -> list[dict]:
    """Each head's illegal arms on the even split, and the same arms paired, for the CSV."""
    heads = [("2-class", localize.read(space, "p_illegal"), ["illegal"]),
             ("4-class", 1.0 - localize.read(space, "proba4")[..., 0], RULES)]
    drawn = [bar(head, score, arm, meta, split)
             for head, score, illegal_arms in heads for arm in illegal_arms]
    subset = shared_games(meta, selection)
    reference = [paired_bar(head, score, arm, meta, selection, games)
                 for head, score, illegal_arms in heads
                 for arm in [*illegal_arms, "control"]
                 for games in (None, subset)]
    return drawn, reference


def draw(space: str, rows: list[dict], camera_ready: bool) -> Path:
    """The illegal arms as bars, and nothing else.

    Drawn, the null was two bars at the chance line saying the same thing the line says; the
    caption that replaced it said what README.md says. Both are still measured -- the null is
    what licenses the chance line being read as empirical -- and both live in the CSV.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.array([i + (GAP if r["head"] == "4-class" else 0.0) for i, r in enumerate(rows)])

    split = next(i for i, r in enumerate(rows) if r["head"] == "4-class")
    ax.axhline(0.5, color=NULL["color"], ls=NULL["ls"], lw=1.2, zorder=1)
    ax.text(x[split - 1: split + 1].mean(), 0.51, "chance", color=NULL["color"], fontsize=7.5,
            ha="center", va="bottom")   # the blank slot between the heads is the only clear space
    for pos, row in zip(x, rows):
        ax.bar(pos, row["auroc"], width=0.72, color=row["style"]["color"],
               edgecolor="white", linewidth=0.6, zorder=3)
        ax.errorbar(pos, row["auroc"], yerr=[[row["auroc"] - row["lo"]],
                                             [row["hi"] - row["auroc"]]],
                    color=MUTED, elinewidth=1.0, capsize=2.5, ls="none", zorder=4)
        ax.text(pos, row["hi"] + 0.02, f"{row['auroc']:.3f}", ha="center", va="bottom",
                fontsize=7.5, color=MUTED, zorder=4)

    for head in ["2-class", "4-class"]:
        at = x[[i for i, r in enumerate(rows) if r["head"] == head]]
        ax.text(at.mean(), 1.10, f"{head} probe", ha="center", va="bottom", fontsize=9)
        ax.plot([at[0] - 0.45, at[-1] + 0.45], [1.07, 1.07], color=MUTED, lw=0.8,
                clip_on=False)

    ax.set_xticks(x, [r["style"]["label"] for r in rows], fontsize=8.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ax.set_ylabel("AUROC")
    # n is one number for the whole panel now that the groups are equal, so it belongs on the
    # axis rather than repeated under every tick.
    ax.set_xlabel(f"at $t_0$, {EVAL_GAMES} games split evenly over the five conditions "
                  f"($n$ = {rows[0]['n_games']} per group)", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)

    fig.tight_layout()
    return save_figure(fig, FIGURES / f"{space}_detection_auroc", camera_ready)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default="raw")
    parser.add_argument("--camera_ready", action="store_true")
    args = parser.parse_args()

    apply_style(args.camera_ready)
    meta = localize.branches()
    selection = localize.arms(meta, SEED)
    split = evaluation_split(meta, selection, SEED)
    t0 = meta["t0"].numpy()
    print(f"{EVAL_GAMES} games, {len(split['control'])} per condition; "
          f"t0 spans {min(t0[r].min() for r in split.values())}-"
          f"{max(t0[r].max() for r in split.values())} in every group")
    drawn, reference = bars(args.space, meta, selection, split)

    print(draw(args.space, drawn, args.camera_ready))
    out = PAPER_RESULTS if args.space == "raw" else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{"space": args.space, **{k: v for k, v in r.items() if k != "style"}}
                          for r in drawn + reference])
    csv = out / f"detection_auroc_{args.space}.csv"
    table.to_csv(csv, index=False)
    print(f"wrote {csv}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
