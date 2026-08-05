"""Ground-truth checks that need no activations: labels, forfeit masking, the r = -1
identity the R3 claim rests on, and the branch-set bookkeeping.

Everything is re-derived with `labels_indep`; `spade.legality_probe.*` is never imported
except where a shipped helper is the thing being tested (`labels_at_ply`, section 4).

Usage: python analysis/legality_probe/redteam/checks_cpu.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels_indep import CATEGORY_NAMES, legal_squares, legality  # noqa: E402

from spade.common.paths import add_karvonen_to_path  # noqa: E402
from spade.common.study import study_dir  # noqa: E402

add_karvonen_to_path()

from circuits.othello_engine_utils import OthelloBoardState, to_string  # noqa: E402

WINDOW = tuple(range(-8, 25))
OUT = Path(__file__).resolve().parent / "results"


def fit_set_labels() -> dict:
    """Re-derive all 10000 fit-set label rows from the saved tokens."""
    d = study_dir("legality") / "fit_set"
    tokens = torch.load(d / "tokens.pt", weights_only=True).long().numpy()
    labels = torch.load(d / "labels.pt", weights_only=True).numpy()
    position = torch.load(d / "position.pt", weights_only=True).long().numpy()
    game = torch.load(d / "game_id.pt", weights_only=True).long().numpy()

    mismatch = 0
    played_legal = 0
    engine_disagree = 0
    for i, (g, p) in enumerate(zip(game, position)):
        board = OthelloBoardState()
        board.update(to_string(tokens[g, : p + 1].tolist()))
        mine = legality(board.state, board.next_hand_color)
        mismatch += int((mine != labels[i]).sum())
        played_legal += int(mine[to_string(int(tokens[g, p + 1]))] == 0)
        engine_disagree += int(
            legal_squares(board.state, board.next_hand_color)
            != {m for m in board.get_valid_moves() if board.tentative_move(m) == 1})
    return {"rows": len(game), "square_mismatches": mismatch,
            "played_move_labelled_legal": played_legal,
            "engine_vs_my_legal_set_disagreements": engine_disagree}


def reference_truth_indep(seq, t0):
    """[33, 64] labels and [33] validity, replayed independently."""
    offset_of = {t0 + r: j for j, r in enumerate(WINDOW)}
    board = OthelloBoardState()
    truth = np.zeros((len(WINDOW), 64), dtype=np.int8)
    valid = np.zeros(len(WINDOW), dtype=bool)
    for s, token in enumerate(seq[: max(offset_of) + 1]):
        board.umpire(to_string(token))
        if s not in offset_of:
            continue
        if not legal_squares(board.state, board.next_hand_color):
            continue
        truth[offset_of[s]] = legality(board.state, board.next_hand_color)
        valid[offset_of[s]] = True
    return truth, valid


def branch_checks() -> dict:
    meta = torch.load(study_dir("divergence") / "branches" / "branches.pt", weights_only=True)
    game_of = meta["game_id"].numpy()
    category = meta["category"].numpy()
    is_control = meta["is_control"].numpy()
    t0 = meta["t0"].numpy()
    tokens = meta["tokens"].numpy()
    branch_token = meta["branch_token"].numpy()
    square = np.array([to_string(int(t)) for t in branch_token])
    label_of = np.array([0, 1, 1, 2, 2, 2, 3], dtype=np.int8)
    rule_class = label_of[category]

    games = sorted(set(game_of.tolist()))
    index = {g: i for i, g in enumerate(games)}
    truth = np.zeros((len(games), len(WINDOW), 64), dtype=np.int8)
    valid = np.zeros((len(games), len(WINDOW)), dtype=bool)
    is_reference = (category == 0) & (is_control == 0)
    for g in games:
        rows = np.flatnonzero(game_of == g)
        reference = rows[is_reference[rows]][0]
        truth[index[g]], valid[index[g]] = reference_truth_indep(
            tokens[reference].tolist(), int(t0[reference]))

    gi = np.array([index[g] for g in game_of])
    r_minus_1 = WINDOW.index(-1)
    at = truth[gi, r_minus_1, square]
    illegal = (category != 0)
    agree = at[illegal] == rule_class[illegal]

    # the same identity at r = 0, where the metric keeps using the pre-move class
    at0 = truth[gi, WINDOW.index(0), square]
    agree0 = at0[illegal] == rule_class[illegal]

    per_category = {CATEGORY_NAMES[c]: {
        "n": int((category == c).sum()),
        "truth_at_r-1_equals_rule_class": float(
            (truth[gi, r_minus_1, square][category == c] == rule_class[category == c]).mean()),
        "truth_at_r0_equals_rule_class": float(
            (truth[gi, WINDOW.index(0), square][category == c] == rule_class[category == c]).mean()),
    } for c in range(1, 7)}

    return {
        "n_branch_rows": int(len(game_of)),
        "n_games": len(games),
        "n_illegal_rows": int(illegal.sum()),
        "window_positions": int(valid.size),
        "masked_as_forfeit": int((~valid).sum()),
        "games_with_any_mask": int((~valid).any(1).sum()),
        "masked_by_offset": {int(r): int((~valid[:, j]).sum())
                             for j, r in enumerate(WINDOW) if (~valid[:, j]).any()},
        "truth_at_r-1_equals_rule_class": float(agree.mean()),
        "truth_at_r0_equals_rule_class": float(agree0.mean()),
        "per_category": per_category,
        "rows_per_game": {"min": int(np.bincount(gi).min()), "max": int(np.bincount(gi).max()),
                          "mean": float(np.bincount(gi).mean())},
    }


def labels_at_ply_gap() -> dict:
    """The known open item: `labels_at_ply` cannot label the last window position.

    Compares the shipped shared helper against the incremental replay for every
    (game, offset) the read-out labels, and reports where the two paths differ.
    """
    from spade.legality_probe.labels import labels_at_ply

    meta = torch.load(study_dir("divergence") / "branches" / "branches.pt", weights_only=True)
    category, is_control = meta["category"].numpy(), meta["is_control"].numpy()
    game_of, t0, tokens = meta["game_id"].numpy(), meta["t0"].numpy(), meta["tokens"].numpy()
    is_reference = (category == 0) & (is_control == 0)

    disagree, none_from_helper, compared = 0, Counter(), 0
    for g in sorted(set(game_of.tolist())):
        rows = np.flatnonzero(game_of == g)
        reference = rows[is_reference[rows]][0]
        seq, start = tokens[reference].tolist(), int(t0[reference])
        truth, valid = reference_truth_indep(seq, start)
        for j, r in enumerate(WINDOW):
            # position p sees moves 0..p, so its labels are labels_at_ply(game, p + 1)
            helper = labels_at_ply(seq, start + r + 1, Counter())
            if helper is None:
                none_from_helper[r] += 1
                continue
            if not valid[j]:
                continue
            compared += 1
            disagree += int((helper != truth[j]).any())
    return {"offsets_where_helper_returns_None": {int(r): int(n)
                                                  for r, n in sorted(none_from_helper.items())},
            "positions_compared": compared, "label_row_disagreements": disagree}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"fit_set": fit_set_labels()}
    print(json.dumps(report["fit_set"], indent=2), flush=True)
    report["branch_window"] = branch_checks()
    print(json.dumps(report["branch_window"], indent=2), flush=True)
    report["labels_at_ply"] = labels_at_ply_gap()
    print(json.dumps(report["labels_at_ply"], indent=2), flush=True)
    (OUT / "checks_cpu.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT/'checks_cpu.json'}")


if __name__ == "__main__":
    main()
