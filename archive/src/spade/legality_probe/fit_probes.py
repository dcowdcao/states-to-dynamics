"""Step 4: fit the four legality probes in each space and persist them.

Four targets x eleven spaces, each saved under `study_dir("legality")/probes`, so the
read-out loads weights instead of refitting 44 probes in memory and a published number is
reproducible from an artifact rather than from a seed.

`legality` is the study's 4-way target; `board` is the 3-class occupancy
`bracket_flip/sae_arm.py` fits, kept because reproducing that arm is this pipeline's
cross-check; `legality_wrong_player` (the same labels computed for the player who just
moved) and `legality_shuffled` are the two floors that show the *level* of a legality
read-out is not a result. `board` and `legality_wrong_player` are derived here by
replaying the saved tokens; the fit set stores only the mover-frame legality labels.

Every fit runs `probe.fit`: train on 128 of the 160 fit games, early stop on the other 32,
score on the 40 eval games that no fit, no stopping rule and no hyperparameter selection
ever sees. Weight decay comes from `weight_decay.selected`, per space, so run that sweep
first. probe_fit.csv carries the held-out scores with a bootstrap over those 40 games;
training_curves.csv carries the per-epoch curve, so the plateau is visible rather than
asserted.

Usage: python -m spade.legality_probe.fit_probes [--spaces raw] [--targets legality]
"""

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from spade.common.device import select_device
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.legality_probe import probe as legality_probe
from spade.legality_probe import spaces
from spade.legality_probe.bootstrap import (
    accuracy_of, by_game, ci, f1_of, recall_of, resample, tally)
from spade.legality_probe.labels import STARTING_SQUARES, disc_labels, opponent_labels
from spade.legality_probe.weight_decay import selected
from spade.probes import tpr

from circuits.othello_engine_utils import OthelloBoardState, to_string

BOARD_CLASSES = ("empty", "theirs", "mine")
NON_START = np.array(sorted(set(range(64)) - set(STARTING_SQUARES)))
FIT_COLUMNS = ["space", "d_space", "target", "n_classes", "filler_dim",
               "metric", "class", "value", "lo", "hi", "n"]


@dataclass(frozen=True)
class Target:
    labels: Tensor  # [n, 64] class indices, fit games first
    classes: tuple[str, ...]
    filler_dim: int


def targets() -> tuple[dict[str, Target], Tensor]:
    """The four fit-set targets and each row's source game."""
    d = study_dir("legality") / "fit_set"
    tokens = torch.load(d / "tokens.pt", weights_only=True).long()
    legality = torch.load(d / "labels.pt", weights_only=True).long()
    position = torch.load(d / "position.pt", weights_only=True).long()
    game = torch.load(d / "game_id.pt", weights_only=True).long()

    board = np.empty((len(position), 64), dtype=np.int8)
    wrong = np.empty_like(board)
    for i, (g, p) in enumerate(zip(game.tolist(), position.tolist())):
        state = OthelloBoardState()
        state.update(to_string(tokens[g, : p + 1].tolist()))
        board[i], wrong[i] = disc_labels(state), opponent_labels(state)

    n_fit = int((game < legality_probe.TRAIN_GAMES).sum())
    permutation = torch.randperm(n_fit, generator=torch.Generator().manual_seed(0))
    classes, filler = legality_probe.CLASSES, legality_probe.FILLER_DIM
    return {
        "legality": Target(legality, classes, filler),
        "board": Target(torch.from_numpy(board).long(), BOARD_CLASSES, tpr.FILLER_DIM),
        "legality_wrong_player": Target(torch.from_numpy(wrong).long(), classes, filler),
        "legality_shuffled": Target(
            torch.cat([legality[permutation], legality[n_fit:]]), classes, filler),
    }, game


def held_out(arm, spec: Target, code: Tensor, te: Tensor, game: Tensor,
             device: str) -> tuple[list[dict], float]:
    """probe_fit.csv rows for one fitted probe, scored on the test games it never saw."""
    acts, truth = code[te].float(), spec.labels[te]
    scores = legality_probe.macro_f1(arm, acts, truth, spec.classes)

    n_classes = len(spec.classes)
    _, test_game = np.unique(game[te].numpy(), return_inverse=True)
    n_games = int(test_game.max()) + 1
    preds = torch.cat([arm(acts[i:i + legality_probe.EVAL_BATCH].to(device)).argmax(-1).cpu()
        for i in range(0, len(acts), legality_probe.EVAL_BATCH)]).numpy().astype(np.int8)
    truth = truth.numpy()
    counts = by_game(tally(preds, truth, n_classes), test_game, n_games)
    pooled, boot = counts.sum(0), resample(counts)
    assert abs(f1_of(pooled).mean() - scores.macro_f1) < 1e-6, "counts disagree with macro_f1"
    assert abs(accuracy_of(pooled) - scores.accuracy) < 1e-5, "counts disagree with the accuracy"

    keys = {"d_space": code.shape[-1], "n_classes": n_classes,
            "filler_dim": spec.filler_dim, "n": n_games}
    rows = [{**keys, "metric": "macro_f1", "class": "", "value": f1_of(pooled).mean(),
             **dict(zip(("lo", "hi"), ci(f1_of(boot).mean(-1))))},
            {**keys, "metric": "accuracy", "class": "", "value": accuracy_of(pooled),
             **dict(zip(("lo", "hi"), ci(accuracy_of(boot))))}]
    for metric, statistic in (("per_class_f1", f1_of), ("per_class_recall", recall_of)):
        point, draws = statistic(pooled), statistic(boot)
        rows += [{**keys, "metric": metric, "class": c, "value": point[i],
                  **dict(zip(("lo", "hi"), ci(draws[:, i])))}
                 for i, c in enumerate(spec.classes)]
    if "R1" in spec.classes:
        # 4 of 64 squares are the starting squares: R1 by construction, and readable off the
        # role embedding alone. Same metrics over the 60 squares a move can be played on.
        i = spec.classes.index("R1")
        open_counts = by_game(tally(preds[..., NON_START], truth[..., NON_START], n_classes),
                              test_game, n_games)
        open_pooled, open_boot = open_counts.sum(0), resample(open_counts)
        rows += [{**keys, "metric": metric, "class": "R1_no_start",
                  "value": statistic(open_pooled)[i],
                  **dict(zip(("lo", "hi"), ci(statistic(open_boot)[:, i])))}
                 for metric, statistic in (("per_class_f1", f1_of), ("per_class_recall", recall_of))]
    return rows, scores.macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default="", help="comma-separated filter")
    parser.add_argument("--targets", default="", help="comma-separated filter")
    parser.add_argument("--seed", type=int, default=legality_probe.SEED)
    args = parser.parse_args()

    device = select_device()
    latents = study_dir("legality") / "latents"
    fingerprint = spaces.fingerprint(spaces.probe_acts())
    target, game = targets()
    if args.targets:
        target = {k: v for k, v in target.items() if k in set(args.targets.split(","))}
    tr, va, te = legality_probe.split(game, seed=args.seed)
    print(f"fingerprint {fingerprint}\nrows: train {len(tr)} val {len(va)} test {len(te)}",
          flush=True)

    chosen = spaces.roster() + [(n, None) for n in spaces.NULLS]
    if args.spaces:
        keep = set(args.spaces.split(","))
        chosen = [s for s in chosen if s[0] in keep]

    rows_dir = study_dir("legality") / "probes" / "fit_rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    for name, _ in chosen:
        fit, curves = [], []
        row = json.loads((latents / f"row_{name}.json").read_text())
        assert row["probe_fingerprint"] == fingerprint, (
            f"{latents}/probe_{name}.pt encodes probe set {row['probe_fingerprint']}, not the "
            f"current {fingerprint}; re-run spade.legality_probe.spaces rather than mixing the two")
        code = torch.load(latents / f"probe_{name}.pt", map_location="cpu", weights_only=True)

        weight_decay = selected(name)
        for arm_name, spec in target.items():
            print(f"{name} / {arm_name} at weight_decay {weight_decay}", flush=True)
            arm, result = legality_probe.fit(
                code, spec.labels, tr, va, weight_decay, classes=spec.classes,
                filler_dim=spec.filler_dim, seed=args.seed, device=device)
            rows, macro = held_out(arm, spec, code, te, game, device)
            legality_probe.save(arm, name, arm_name, {
                "seed": args.seed, "classes": list(spec.classes),
                "d_space": code.shape[-1],
                "best_epoch": result.best_epoch, "epochs_run": len(result.curve),
                "val_accuracy": result.val_accuracy, "test_macro_f1": macro,
                "n_train_games": legality_probe.TRAIN_POOL,
                "probe_fingerprint": fingerprint,
                "dataset": "adamkarvonen/othello_45MB_games"})
            fit += [{"space": name, "target": arm_name, **r} for r in rows]
            curves += [dict(space=name, target=arm_name, **c) for c in result.curve]
            print(f"{name:28} {arm_name:22} macro_f1 {macro:.4f}  "
                  f"val_acc {result.val_accuracy:.4f}  best epoch {result.best_epoch}"
                  f" of {len(result.curve)}", flush=True)
        pd.DataFrame(fit)[FIT_COLUMNS].to_csv(rows_dir / f"{name}.csv", index=False)
        pd.DataFrame(curves).to_csv(rows_dir / f"{name}_curve.csv", index=False)
        del code

    # Per space, so that a --spaces run refits its spaces without truncating the study's
    # table to them; both outputs are whatever fit_rows/ holds, in roster order.
    order = [n for n, _ in spaces.roster()] + list(spaces.NULLS)
    have = [n for n in order if (rows_dir / f"{n}.csv").exists()]
    fit = pd.concat([pd.read_csv(rows_dir / f"{n}.csv") for n in have])
    out = repo_root() / "analysis" / "legality_probe" / "results"
    out.mkdir(parents=True, exist_ok=True)
    fit.to_csv(out / "probe_fit.csv", index=False)
    pd.concat([pd.read_csv(rows_dir / f"{n}_curve.csv") for n in have]).to_csv(
        study_dir("legality") / "probes" / "training_curves.csv", index=False)
    print(f"wrote {out}/probe_fit.csv ({len(fit)} rows from {len(have)} spaces)")
    report(fit)


def report(fit: pd.DataFrame) -> None:
    macro = fit[fit.metric == "macro_f1"].pivot(index="space", columns="target", values="value")
    print("\nheld-out macro-F1 by arm\n", macro.to_string())

    arm = pd.read_csv(repo_root() / "analysis" / "bracket_flip" / "results" / "sae_arm.csv")
    board = fit[(fit.target == "board") & (fit.metric == "accuracy")].set_index("space")["value"]
    cross = arm.set_index("space")[["acc_rel_L5"]].join(board.rename("board_accuracy"), how="inner")
    cross["delta"] = cross["board_accuracy"] - cross["acc_rel_L5"]
    print("\nboard arm vs analysis/bracket_flip/results/sae_arm.csv\n", cross.to_string())


if __name__ == "__main__":
    main()
