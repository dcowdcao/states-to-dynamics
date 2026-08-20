"""Step 5: fit the legality probe in each space, read it out over the branch window, write
the two CSVs the figures consume.

Four probes per space on one held-out split. `legality` is the study's 4-way target;
`board` is the 3-class occupancy `bracket_flip/sae_arm.py` fits, kept because reproducing
that arm is this pipeline's cross-check; `legality_wrong_player` (the same labels computed
for the player who just moved) and `legality_shuffled` are the two floors that show the
*level* of a legality read-out is not a result.

The read-out scores every branch of a game against ONE target: the reference game's labels
at t0 + r, in the reference game's own mover frame. After an illegal move whose turn it is
is precisely what is in dispute, so a per-branch frame would score each branch against a
different board -- `analysis/violation_divergence/probe_readout/decode.py` makes the same
choice for the same reason. Positions where the reference game forfeits are masked, since
`taxonomy.classify() == LEGAL` stops matching the engine there.

r < 0 is the paired control. A game's branches share their prefix bitwise and every encoder
preserves that (`spaces.prefix_bitwise`, re-checked here *through the probe*), so damage at
r < 0 is ordinary lossiness and the excess at r = 0 is what the violation costs.
`drop_macro_f1` is the within-space drop; `excess_macro_f1` subtracts raw's on the same
bootstrap draws, so the two are paired.

macro_f1 against r is not a decay curve: R1 is exactly (ply+5)/64 of squares and r shifts
absolute ply. That drift is common to every space, so between-space contrasts at fixed r
are clean; `macro_f1_minus_raw` is the per-game paired version, and the per-class rows show
which class moves.

Usage: python -m spade.legality_probe.readout [--smoke] [--spaces raw,TopKTrainer-othello]
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from spade.common.device import select_device
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.legality_probe import probe as legality_probe
from spade.legality_probe import spaces
from spade.legality_probe.fit_set import N_TRAIN
from spade.legality_probe.labels import (
    LABEL_OF_CATEGORY, STARTING_SQUARES, board_labels, disc_labels, opponent_labels)
from spade.othello.taxonomy import CATEGORIES, LEGAL, RULE_MEMBERS, RULE_OF, classify_all
from spade.probes.tpr import FILLER_DIM as BOARD_FILLER_DIM
from spade.violation_divergence.branches import role
from spade.violation_divergence.divergence import WINDOW

from circuits.othello_engine_utils import OthelloBoardState, to_string

N_BOOT = 1000
R0 = WINDOW.index(0)
BOARD_CLASSES = ("empty", "theirs", "mine")
EMPTY_CATEGORIES = ["empty_detached", "empty_own_only", "empty_no_bracket", "opponent_only"]
NON_START = np.array(sorted(set(range(64)) - set(STARTING_SQUARES)))
SMOKE_GAMES = 50
FIT_COLUMNS = ["space", "d_space", "target", "n_classes", "filler_dim",
               "metric", "class", "value", "lo", "hi", "n"]
READOUT_COLUMNS = ["space", "category", "rule", "role", "r", "metric", "mean", "lo", "hi", "n"]


# ---------------------------------------------------------------- bootstrap over games

@lru_cache(maxsize=None)
def resample_counts(n: int, seed: int = 0) -> np.ndarray:
    """One bootstrap design per sample size, shared by every space and every metric -- which
    is what makes `excess_macro_f1` and `macro_f1_minus_raw` paired rather than a difference
    of two independent intervals."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(np.float32)


def boot_ci(values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mean and percentile 95% CI over source games (axis 0), per column.

    Resampling as multinomial row counts rather than an index array: the same draw, but a
    [N_BOOT, n] @ [n, k] matmul instead of an [N_BOOT, n, k] gather.
    """
    valid = np.isfinite(values).astype(np.float32)
    counts = resample_counts(len(values))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (counts @ np.nan_to_num(values).astype(np.float32)) / (counts @ valid)
        mean = np.nansum(values, axis=0) / valid.sum(0)
        return (mean, *np.nanpercentile(draws, [2.5, 97.5], axis=0))


def series(values: np.ndarray, **keys) -> list[dict]:
    """One row per window offset from a [games, 33] per-game quantity."""
    mean, lo, hi = boot_ci(values)
    return [{**keys, "r": r, "mean": mean[j], "lo": lo[j], "hi": hi[j],
             "n": int(np.isfinite(values[:, j]).sum())}
            for j, r in enumerate(WINDOW)]


def resample(counts: np.ndarray) -> np.ndarray:
    """[N_BOOT, ...] re-pooled (tp, fp, fn) totals. F1 is a ratio of sums, so a resample
    re-pools the counts instead of averaging a per-game statistic."""
    draws = resample_counts(len(counts))
    return (draws @ counts.reshape(len(counts), -1).astype(np.float32)
            ).reshape(N_BOOT, *counts.shape[1:])


def f1_of(counts: np.ndarray) -> np.ndarray:
    """Per-class F1 from [..., 3, C] of (tp, fp, fn); an absent class scores 0, which is
    upstream's `macro_f1_score` definition and `probe.macro_f1`'s."""
    tp, fp, fn = counts[..., 0, :], counts[..., 1, :], counts[..., 2, :]
    denominator = 2 * tp + fp + fn
    return np.where(denominator > 0, 2 * tp / np.where(denominator > 0, denominator, 1), 0.0)


def recall_of(counts: np.ndarray) -> np.ndarray:
    tp, fn = counts[..., 0, :], counts[..., 2, :]
    return np.where(tp + fn > 0, tp / np.where(tp + fn > 0, tp + fn, 1), 0.0)


def accuracy_of(counts: np.ndarray) -> np.ndarray:
    """Every square carries exactly one class, so the true totals are tp + fn."""
    tp, fn = counts[..., 0, :], counts[..., 2, :]
    return tp.sum(-1) / (tp + fn).sum(-1)


def tally(preds: np.ndarray, truth: np.ndarray, n_classes: int) -> np.ndarray:
    """(tp, fp, fn) per class over the squares axis: [..., 64] -> [..., 3, n_classes]."""
    out = np.empty((*preds.shape[:-1], 3, n_classes), dtype=np.int32)
    for c in range(n_classes):
        predicted, actual = preds == c, truth == c
        out[..., 0, c] = (predicted & actual).sum(-1)
        out[..., 1, c] = (predicted & ~actual).sum(-1)
        out[..., 2, c] = (~predicted & actual).sum(-1)
    return out


def by_game(values: np.ndarray, game: np.ndarray, n_games: int) -> np.ndarray:
    """Sum each game's rows. `game` is non-decreasing -- rows are grouped by game and no
    span splits one -- so this is a segmented reduction, not a scatter."""
    assert (np.diff(game) >= 0).all(), "rows are not grouped by game"
    present, start = np.unique(game, return_index=True)
    out = np.zeros((n_games, *values.shape[1:]), dtype=values.dtype)
    out[present] = np.add.reduceat(values, start, axis=0)
    return out


def per_game(values: np.ndarray, game: np.ndarray, n_games: int) -> np.ndarray:
    """Mean over each game's rows, NaN for games with none -- the bootstrap's unit."""
    total = by_game(values.astype(np.float64), game, n_games)
    count = by_game(np.ones((len(values), 1)), game, n_games)
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


def ci(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return lo, hi


# ---------------------------------------------------------------- the four probe targets

@dataclass(frozen=True)
class Target:
    labels: torch.Tensor  # [n, 64] class indices, train rows first
    classes: tuple[str, ...]
    filler_dim: int


def targets() -> tuple[dict[str, Target], np.ndarray]:
    """The four fit-set targets and each row's source game.

    `board` and `legality_wrong_player` are derived here by replaying the saved tokens;
    the fit set stores only the mover-frame legality labels.
    """
    d = study_dir("legality") / "fit_set"
    tokens = torch.load(d / "tokens.pt", weights_only=True).long()
    legality = torch.load(d / "labels.pt", weights_only=True).long()
    position = torch.load(d / "position.pt", weights_only=True).long()
    game = torch.load(d / "game_id.pt", weights_only=True).long().numpy()

    board = np.empty((len(position), 64), dtype=np.int8)
    wrong = np.empty_like(board)
    for i, (g, p) in enumerate(zip(game, position.tolist())):
        state = OthelloBoardState()
        state.update(to_string(tokens[g, : p + 1].tolist()))
        board[i], wrong[i] = disc_labels(state), opponent_labels(state)

    permutation = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(0))
    classes, filler = legality_probe.CLASSES, legality_probe.FILLER_DIM
    return {
        "legality": Target(legality, classes, filler),
        "board": Target(torch.from_numpy(board).long(), BOARD_CLASSES, BOARD_FILLER_DIM),
        "legality_wrong_player": Target(torch.from_numpy(wrong).long(), classes, filler),
        "legality_shuffled": Target(
            torch.cat([legality[permutation], legality[N_TRAIN:]]), classes, filler),
    }, game


def decode_argmax(probe: legality_probe.LegalityProbe, acts: torch.Tensor,
                  device: str) -> np.ndarray:
    return torch.cat([probe.decode(acts[i:i + legality_probe.EVAL_BATCH].to(device)).argmax(-1).cpu()
        for i in range(0, len(acts), legality_probe.EVAL_BATCH)]).numpy().astype(np.int8)


def fit_rows(space: str, name: str, target: Target, code: torch.Tensor, game: np.ndarray,
             device: str) -> tuple[legality_probe.LegalityProbe, list[dict]]:
    probe = legality_probe.fit(code[:N_TRAIN].float(), target.labels[:N_TRAIN],
                               filler_dim=target.filler_dim, classes=target.classes,
                               device=device)
    held, truth = code[N_TRAIN:].float(), target.labels[N_TRAIN:]
    scores = legality_probe.macro_f1(probe, held, truth)

    n_classes = len(target.classes)
    _, eval_game = np.unique(game[N_TRAIN:], return_inverse=True)
    n_games = int(eval_game.max()) + 1
    preds, truth = decode_argmax(probe, held, device), truth.numpy()
    counts = by_game(tally(preds, truth, n_classes), eval_game, n_games)
    pooled, boot = counts.sum(0), resample(counts)
    assert abs(f1_of(pooled).mean() - scores.macro_f1) < 1e-6, "counts disagree with probe.macro_f1"
    assert abs(accuracy_of(pooled) - scores.accuracy) < 1e-5, "counts disagree with the accuracy"

    keys = {"space": space, "d_space": code.shape[-1], "target": name,
            "n_classes": n_classes, "filler_dim": target.filler_dim, "n": n_games}
    rows = [{**keys, "metric": "macro_f1", "class": "", "value": f1_of(pooled).mean(),
             **dict(zip(("lo", "hi"), ci(f1_of(boot).mean(-1))))},
            {**keys, "metric": "accuracy", "class": "", "value": accuracy_of(pooled),
             **dict(zip(("lo", "hi"), ci(accuracy_of(boot))))}]
    for metric, statistic in (("per_class_f1", f1_of), ("per_class_recall", recall_of)):
        point, draws = statistic(pooled), statistic(boot)
        rows += [{**keys, "metric": metric, "class": c, "value": point[i],
                  **dict(zip(("lo", "hi"), ci(draws[:, i])))}
                 for i, c in enumerate(target.classes)]
    if "R1" in target.classes:
        # 4 of 64 squares are the starting squares: R1 by construction, and readable off the
        # role embedding alone. Same metrics over the 60 squares a move can be played on.
        i = target.classes.index("R1")
        open_counts = by_game(tally(preds[..., NON_START], truth[..., NON_START], n_classes),
                              eval_game, n_games)
        open_pooled, open_boot = open_counts.sum(0), resample(open_counts)
        rows += [{**keys, "metric": metric, "class": "R1_no_start", "value": statistic(open_pooled)[i],
                  **dict(zip(("lo", "hi"), ci(statistic(open_boot)[:, i])))}
                 for metric, statistic in (("per_class_f1", f1_of), ("per_class_recall", recall_of))]
    return probe, rows


# ---------------------------------------------------------------- the branch window

def reference_truth(seq: list[int], t0: int, skips: Counter) -> tuple[np.ndarray, np.ndarray]:
    """[33, 64] legality labels of the reference game at t0 + WINDOW, and which are usable.

    One target per source game, in that game's own mover frame, for every branch alike.
    A forfeit position is masked rather than labelled.
    """
    offset_of = {t0 + r: j for j, r in enumerate(WINDOW)}
    board = OthelloBoardState()
    truth = np.zeros((len(WINDOW), 64), dtype=np.int8)
    valid = np.zeros(len(WINDOW), dtype=bool)
    for s, token in enumerate(seq[: max(offset_of) + 1]):
        board.umpire(to_string(token))
        if s not in offset_of:
            continue
        if not any(board.tentative_move(m) == 1 for m in board.get_valid_moves()):
            skips["forfeit"] += 1
            continue
        labels = board_labels(board, classify_all(board))
        assert set(np.flatnonzero(labels == 0).tolist()) == set(board.get_valid_moves()), \
            f"taxonomy/engine disagreement at board position {s}"
        truth[offset_of[s]], valid[offset_of[s]] = labels, True
    return truth, valid


@dataclass(frozen=True)
class Window:
    """Everything about the read-out set that does not depend on the space."""
    game_index: np.ndarray  # [rows] -> 0..n_games-1
    n_games: int
    truth: np.ndarray       # [n_games, 33, 64]
    valid: np.ndarray       # [n_games, 33]
    square: np.ndarray      # [rows] the branch token's square
    rule_class: np.ndarray  # [rows] the class the branch's own move carries before t0
    sets: list["Branches"]


@dataclass(frozen=True)
class Branches:
    category: str
    rule: str
    role: str
    rows: np.ndarray


def window(meta: dict, skips: Counter) -> Window:
    games = sorted(meta["game_id"].unique().tolist())
    game_of = meta["game_id"].numpy()
    is_reference = ((meta["category"] == LEGAL) & (meta["is_control"] == 0)).numpy()
    truth = np.zeros((len(games), len(WINDOW), 64), dtype=np.int8)
    valid = np.zeros((len(games), len(WINDOW)), dtype=bool)
    for i, g in enumerate(games):
        rows = np.flatnonzero(game_of == g)
        reference = rows[is_reference[rows]][0]
        truth[i], valid[i] = reference_truth(
            meta["tokens"][reference].long().tolist(), int(meta["t0"][reference]), skips)

    roles = np.array([role(int(c), int(k))
                      for c, k in zip(meta["category"], meta["is_control"])])
    pools = ([(c, RULE_OF[c], [c]) for c in CATEGORIES[1:]]
             + [("", rule, members) for rule, members in RULE_MEMBERS.items()]
             + [("empty_pooled", "", EMPTY_CATEGORIES), ("illegal_pooled", "", CATEGORIES[1:])])
    sets = [Branches("legal", "", r, np.flatnonzero(roles == r))
            for r in ("reference", "control_legal", "control_legal_2")]
    sets += [Branches(name, rule, "illegal", np.flatnonzero(np.isin(roles, members)))
             for name, rule, members in pools]
    assert all(len(s.rows) for s in sets), [s.category for s in sets if not len(s.rows)]

    return Window(
        game_index=np.searchsorted(games, game_of), n_games=len(games), truth=truth, valid=valid,
        square=np.array([to_string(t) for t in meta["branch_token"].tolist()]),
        rule_class=np.array([LABEL_OF_CATEGORY[c] for c in meta["category"].tolist()]),
        sets=sets)


def decode_window(name: str, entry, branch: torch.Tensor, game_id: torch.Tensor,
                  probe: legality_probe.LegalityProbe, device: str) -> np.ndarray:
    """[rows, 33, 64] int8 argmax, one `spans` batch at a time so the [rows, 33, d] code
    never exists whole, with the prefix check section C depends on."""
    preds = np.empty((len(branch), len(WINDOW), 64), dtype=np.int8)
    for rows, code in spaces.stream_branch(name, entry, branch, game_id, device):
        span = probe.decode(code.to(device).float()).argmax(-1).to(torch.int8).cpu()
        assert spaces.prefix_bitwise(span, game_id[rows]), \
            f"{name}: branches of a game do not decode identically before t0"
        preds[rows] = span.numpy()
    return preds


@dataclass
class Baseline:
    """Raw's per-set quantities, kept so the paired metrics are differences of the *same*
    bootstrap draws rather than of two independent intervals."""
    macro: dict[tuple, np.ndarray]  # [games, 33] per-game macro-F1
    drop: dict[tuple, tuple]        # (point, [N_BOOT] draws) of the r<0 -> r=0 drop


def readout_rows(space: str, preds: np.ndarray, raw_preds: np.ndarray, w: Window,
                 raw: Baseline) -> tuple[list[dict], Baseline]:
    """Every readout.csv row for one space, plus the baselines raw hands to the others."""
    classes = legality_probe.CLASSES
    truth = w.truth[w.game_index]
    keep = w.valid[w.game_index][..., None, None]
    counts_of = tally(preds, truth, len(classes)) * keep
    open_counts_of = tally(preds[..., NON_START], truth[..., NON_START], len(classes)) * keep
    agree_of = (preds == raw_preds).mean(-1)
    played = preds[np.arange(len(preds)), :, w.square]
    legal_of = (played == 0).astype(np.float64)
    rule_of = (played == w.rule_class[:, None]).astype(np.float64)

    rows, mine = [], Baseline({}, {})
    for s in w.sets:
        key = (s.category, s.rule, s.role)
        keys = {"space": space, "category": s.category, "rule": s.rule, "role": s.role}
        game = w.game_index[s.rows]
        counts = by_game(counts_of[s.rows], game, w.n_games)
        point, draws = f1_of(counts.sum(0)), f1_of(resample(counts))
        n = int((counts.sum((1, 2, 3)) > 0).sum())

        open_counts = by_game(open_counts_of[s.rows], game, w.n_games)
        r1 = classes.index("R1")
        curves = [("macro_f1", point.mean(-1), draws.mean(-1))]
        curves += [(f"f1_{c}", point[:, i], draws[:, :, i]) for i, c in enumerate(classes)]
        curves += [("f1_R1_no_start", f1_of(open_counts.sum(0))[:, r1],
                    f1_of(resample(open_counts))[:, :, r1])]
        for metric, value, spread in curves:
            lo, hi = ci(spread)
            rows += [{**keys, "metric": metric, "r": r, "mean": value[j],
                      "lo": lo[j], "hi": hi[j], "n": n} for j, r in enumerate(WINDOW)]

        agree = per_game(agree_of[s.rows], game, w.n_games)
        rows += series(agree, **keys, metric="agreement_with_raw")
        rows += series(per_game(legal_of[s.rows], game, w.n_games), **keys,
                       metric="p_pred_legal_at_played_square")
        if s.role == "illegal":
            rows += series(per_game(rule_of[s.rows], game, w.n_games), **keys,
                           metric="p_pred_correct_rule")

        # Paired within game against raw: what cancels the ply drift in the macro-F1 curve.
        labelled = (counts[..., 0, :] + counts[..., 2, :]).sum(-1) > 0
        macro = np.where(labelled, f1_of(counts).mean(-1), np.nan)
        rows += series(macro - raw.macro.get(key, macro), **keys, metric="macro_f1_minus_raw")
        mine.macro[key] = macro

        # Section C: the drop at the violation against this space's own r < 0 baseline, and
        # the excess of that drop over raw's, which separates violation-specific failure
        # from ordinary lossiness.
        before, at_zero = counts[:, :spaces.N_BEFORE].sum(1), counts[:, R0]
        drop = (f1_of(before.sum(0)).mean() - f1_of(at_zero.sum(0)).mean(),
                f1_of(resample(before)).mean(-1) - f1_of(resample(at_zero)).mean(-1))
        reference = raw.drop.get(key, drop)
        mine.drop[key] = drop
        for metric, value, spread in [("drop_macro_f1", *drop),
                                      ("excess_macro_f1", drop[0] - reference[0],
                                       drop[1] - reference[1])]:
            lo, hi = ci(spread)
            rows.append({**keys, "metric": metric, "r": 0, "mean": float(value),
                         "lo": float(lo), "hi": float(hi), "n": n})
        agree_drop = agree[:, :spaces.N_BEFORE].mean(1) - agree[:, R0]
        mean, lo, hi = boot_ci(agree_drop[:, None])
        rows.append({**keys, "metric": "drop_agreement_with_raw", "r": 0, "mean": mean[0],
                     "lo": lo[0], "hi": hi[0], "n": int(np.isfinite(agree_drop).sum())})
    return rows, mine


# ---------------------------------------------------------------- driver

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default="", help="comma-separated filter; raw is always kept")
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE_GAMES} source games, into study_dir('legality')/smoke/")
    args = parser.parse_args()

    device = select_device()
    latents = study_dir("legality") / "latents"
    fingerprint = spaces.fingerprint(spaces.probe_acts())
    target, fit_game = targets()

    meta = torch.load(study_dir("divergence") / "branches" / "branches.pt", weights_only=True)
    branch, game_id = spaces.branch_window()
    if args.smoke:
        keep = (meta["game_id"] < SMOKE_GAMES).numpy()
        meta = {k: v[keep] for k, v in meta.items()}
        branch, game_id = branch[keep], game_id[keep]
    skips = Counter()
    w = window(meta, skips)
    masked = int((~w.valid).sum())
    print(f"probe set {spaces.PROBE_ACTS}\nfingerprint {fingerprint}\n"
          f"{len(branch)} branches from {w.n_games} games; "
          f"{masked}/{w.valid.size} window positions masked as forfeits", flush=True)

    chosen = spaces.roster() + [(n, None) for n in spaces.NULLS]
    if args.spaces:
        keep = set(args.spaces.split(",")) | {"raw"}
        chosen = [s for s in chosen if s[0] in keep]
    assert chosen[0][0] == "raw", "raw runs first: every other space is scored against it"

    fit, readout, raw_preds, baseline = [], [], None, Baseline({}, {})
    for name, entry in chosen:
        row = json.loads((latents / f"row_{name}.json").read_text())
        assert row["probe_fingerprint"] == fingerprint, (
            f"{latents}/probe_{name}.pt encodes probe set {row['probe_fingerprint']}, not the "
            f"current {fingerprint}; re-run spade.legality_probe.spaces rather than mixing the two")
        code = torch.load(latents / f"probe_{name}.pt", map_location="cpu", weights_only=True)

        probes = {}
        for arm, spec in target.items():
            probes[arm], rows = fit_rows(name, arm, spec, code, fit_game, device)
            fit += rows
            print(f"{name:28} {arm:22} macro_f1 {rows[0]['value']:.4f}  "
                  f"acc {rows[1]['value']:.4f}", flush=True)
        del code

        preds = decode_window(name, entry, branch, game_id, probes["legality"], device)
        raw_preds = preds if raw_preds is None else raw_preds
        rows, mine = readout_rows(name, preds, raw_preds, w, baseline)
        baseline = mine if name == "raw" else baseline
        readout += rows
        print(f"{name:28} read out over {len(branch)} branches", flush=True)

    fit = pd.DataFrame(fit)[FIT_COLUMNS]
    readout = pd.DataFrame(readout)[READOUT_COLUMNS]
    out = study_dir("legality") / "smoke" if args.smoke \
        else repo_root() / "analysis" / "legality_probe" / "results"
    out.mkdir(parents=True, exist_ok=True)
    fit.to_csv(out / "probe_fit.csv", index=False)
    readout.to_csv(out / "readout.csv", index=False)
    print(f"wrote {out}/probe_fit.csv ({len(fit)} rows), {out}/readout.csv ({len(readout)} rows)")
    report(fit, readout)


def report(fit: pd.DataFrame, readout: pd.DataFrame) -> None:
    macro = fit[(fit.metric == "macro_f1")].pivot(index="space", columns="target", values="value")
    print("\nheld-out macro-F1 by arm\n", macro.to_string())

    arm = pd.read_csv(repo_root() / "analysis" / "bracket_flip" / "results" / "sae_arm.csv")
    board = fit[(fit.target == "board") & (fit.metric == "accuracy")].set_index("space")["value"]
    cross = arm.set_index("space")[["acc_rel_L5"]].join(board.rename("board_accuracy"), how="inner")
    cross["delta"] = cross["board_accuracy"] - cross["acc_rel_L5"]
    print("\nboard arm vs analysis/bracket_flip/results/sae_arm.csv\n", cross.to_string())

    reference = readout[(readout.role == "reference") & (readout.metric == "macro_f1")
                        & (readout.space == "raw")]
    assert reference["mean"].min() > 0.7, \
        f"raw decodes its own reference game at only {reference['mean'].min():.3f}: the " \
        "ground-truth construction is wrong"
    print(f"\nraw, reference role, macro_f1 over r: min {reference['mean'].min():.4f} "
          f"max {reference['mean'].max():.4f}")
    agreement = readout[(readout.space == "raw") & (readout.metric == "agreement_with_raw")]
    assert (agreement["mean"] == 1.0).all(), "raw does not agree with itself"
    print(f"agreement_with_raw for raw against itself: {agreement['mean'].min()} on "
          f"{len(agreement)} cells")

    rules = readout[(readout.category == "") & (readout.r == 0)
                    & (readout.metric == "p_pred_correct_rule")]
    print("\np_pred_correct_rule at r = 0, by rule (members: "
          + ", ".join(f"{r} n={len(m)}" for r, m in RULE_MEMBERS.items()) + ")\n",
          rules.pivot(index="space", columns="rule", values="mean").to_string())

    headline = readout[(readout.category == "illegal_pooled")
                       & readout.metric.isin(["drop_macro_f1", "excess_macro_f1",
                                              "drop_agreement_with_raw"])]
    print("\nillegal_pooled, r = 0 against the space's own r < 0 baseline\n",
          headline.pivot(index="space", columns="metric", values="mean").to_string())


if __name__ == "__main__":
    main()
