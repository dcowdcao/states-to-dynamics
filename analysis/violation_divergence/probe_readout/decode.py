"""M6: read the layer-5 board probe out on every branch of the violation-divergence study.

The probe is `spade.probes.load.board(5)`, the project's one board read-out. It is
mover-relative -- {empty, theirs, mine} with "mine" = the player to
move at the *next* position -- and after an illegal move whose turn it is is
exactly what is in dispute. So both branches are decoded and scored against one
common target: the **reference game's board**, labelled in the **reference
game's frame**, with the mover taken from engine replay rather than from t % 2
(forfeits are unmarked in this corpus, so parity desynchronises from the mover).
"theirs" at offset r is therefore whichever colour the reference game's previous
mover holds at position t0 + r, for every branch alike.

Writes results/violation_divergence/probe_*.csv, tidy long format; F7 is drawn
from them by make_figures.py in this directory.

`--variant rollout` reads the tree where every branch, the reference included,
samples its own continuation, and writes `probe_rollout_*.csv`. Two things
change. Each of the K rollouts of a game carries its own reference, so a
comparison group is a (game, rollout) pair, and a branch's K rows are averaged
before any interval -- otherwise a branch is one item K times and the CI is ~2x
too tight. And the reference game is now *sampled*, so its board exists only
while OthelloGPT keeps playing legally: `reference_boards` stops at the first
illegal move and every read-out after it is NaN, not a guess. `probe_rollout_
sanity.csv` carries the resulting per-offset n.

Usage: python analysis/violation_divergence/probe_readout/decode.py [--smoke]
                                                                   [--variant rollout]
"""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spade.common.config import CTX_LEN
from spade.common.paths import repo_root
from spade.common.study import variant_dir
from spade.othello.boards import flip_set, regular_moves
from spade.othello.taxonomy import CATEGORIES, LEGAL
from spade.probes.load import board as board_probe
from spade.violation_divergence.compute_divergence import group_key
from spade.violation_divergence.divergence import WINDOW

from circuits.othello_engine_utils import OthelloBoardState, to_string

EMPTY, THEIRS, MINE = 0, 1, 2  # spade.probes.load.CLASSES order
PROBE_LAYER = 5
N_BOOT = 1000
R0 = WINDOW.index(0)
# Square classes at a window position, from the reference board. `disputed` is
# every square some branch of the game plays or flips, so the untouched classes
# are exactly the squares no branching decision could have touched.
SQUARE_CLASSES = ["empty", "untouched_theirs", "untouched_mine", "reference_sq",
                  "reference_flipped", "disputed"]
EMPTY_SQUARE_CATEGORIES = ["empty_detached", "empty_own_only", "empty_no_bracket",
                           "opponent_only"]
PERSIST_AT = [1, 2, 4, 8, 16, 25]  # plies of unbroken disagreement, 25 = the whole window
VARIANTS = ("spliced", "rollout")


def role(category: int, is_control: int) -> str:
    if is_control:
        return "control_legal"
    return "reference" if category == LEGAL else CATEGORIES[category]


@lru_cache(maxsize=None)
def resample_counts(n: int, seed: int = 0) -> np.ndarray:
    """One bootstrap design per sample size, shared by every curve of that size."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1 / n), size=N_BOOT).astype(np.float32)


def boot_ci(values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mean and percentile 95% CI over source games (axis 0), per window offset.

    Resampling as multinomial row counts rather than an index array: the same
    draw, but a [N_BOOT, n] @ [n, offsets] matmul instead of a [N_BOOT, n,
    offsets] gather, which is what makes 130 of these cheap.
    """
    valid = np.isfinite(values).astype(np.float32)
    counts = resample_counts(len(values))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (counts @ np.nan_to_num(values).astype(np.float32)) / (counts @ valid)
        mean = np.nansum(values, axis=0) / valid.sum(0)
        return (mean, *np.nanpercentile(draws, [2.5, 97.5], axis=0))


def series(values: np.ndarray, **keys) -> list[dict]:
    mean, lo, hi = boot_ci(values)
    return [{**keys, "r": r, "n": int(np.isfinite(values[:, j]).sum()),
             "mean": mean[j], "lo": lo[j], "hi": hi[j]}
            for j, r in enumerate(WINDOW)]


def reference_boards(seq: list[int],
                     sampled: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Absolute board after each move, the colour to move next, and where a board exists.

    `sampled` says the continuation came from the model rather than the source game,
    so it can play an illegal move, after which no board follows. `valid` is False
    from there on and every read-out at those positions is NaN rather than a guess.
    """
    board = OthelloBoardState()
    states = np.zeros((CTX_LEN, 64), dtype=np.int8)
    movers = np.zeros(CTX_LEN, dtype=np.int8)
    valid = np.ones(CTX_LEN, dtype=bool)
    for s, token in enumerate(seq):
        square = to_string(token)
        if sampled and square not in (regular_moves(board.state, board.next_hand_color)
                                      or regular_moves(board.state, -board.next_hand_color)):
            valid[s:] = False
            break
        board.umpire(square)
        states[s] = board.state.reshape(64)
        movers[s] = board.next_hand_color
    return states, movers, valid


def defined(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Float view with the offsets where the reference has no board set to NaN."""
    return np.where(valid, values.astype(float), np.nan)


def per_branch(values: np.ndarray, k: int) -> np.ndarray:
    """Average a per-row score over the K rollouts of its source branch.

    The rollout tree stores them as contiguous rows and every selection here is on
    columns constant within a branch, so a reshape is the grouping. k = 1 in the
    spliced tree, where this is the identity.
    """
    if k == 1:
        return values
    g = values.reshape(-1, k, *values.shape[1:])
    ok = np.isfinite(g)
    n = ok.sum(1)
    return np.where(n > 0, np.where(ok, g, 0).sum(1) / np.maximum(n, 1), np.nan)


def label(states: np.ndarray, movers: np.ndarray) -> np.ndarray:
    """{empty, theirs, mine} per square, in the frame `movers` defines."""
    return np.where(states == 0, EMPTY,
                    np.where(states == movers[:, None], MINE, THEIRS)).astype(np.int8)


def group_tables(items: dict, group_of: np.ndarray, sampled: bool) -> dict[str, np.ndarray]:
    """Per comparison group: window labels in both frames, square classes, the t0
    mover's class, and where the reference still has a board.

    A group is a source game, or in the rollout tree one (game, rollout) sample,
    since there each rollout carries its own reference.
    """
    tokens, t0s = items["tokens"].long().numpy(), items["t0"].numpy()
    is_reference = ((items["category"] == LEGAL) & (items["is_control"] == 0)).numpy()
    branch_square = np.array([to_string(t) for t in items["branch_token"].tolist()])
    parity_mover = np.where(np.arange(CTX_LEN) % 2 == 0, -1, 1).astype(np.int8)

    out = {k: [] for k in ("lab", "lab_parity", "cls", "mover_colour", "valid",
                           "reference_sq", "reference_row", "control_row")}
    for g in range(group_of.max() + 1):
        rows = np.flatnonzero(group_of == g)
        ref = rows[is_reference[rows]][0]
        t0 = int(t0s[ref])
        seq = tokens[ref].tolist()
        states, movers, valid = reference_boards(seq, sampled)
        pos = t0 + np.asarray(WINDOW)

        pre = OthelloBoardState()
        pre.update([to_string(t) for t in seq[:t0]])
        mover_at_t0 = pre.next_hand_color
        reference_sq = branch_square[ref]
        reference_flipped = flip_set(pre.state, reference_sq, mover_at_t0)
        disputed = set(branch_square[rows].tolist())
        for row in rows[(items["category"].numpy() == LEGAL)[rows]]:  # only legal moves flip
            disputed |= set(flip_set(pre.state, branch_square[row], mover_at_t0))

        lab = label(states, movers)[pos]
        cls = lab.copy()  # SQUARE_CLASSES[:3] is the probe's class order, so lab is already the class
        cls[:, sorted(disputed)] = SQUARE_CLASSES.index("disputed")
        cls[:, reference_flipped] = SQUARE_CLASSES.index("reference_flipped")
        cls[:, reference_sq] = SQUARE_CLASSES.index("reference_sq")

        out["lab"].append(lab)
        out["lab_parity"].append(label(states, parity_mover)[pos])
        out["cls"].append(cls)
        out["mover_colour"].append(np.where(movers[pos] == mover_at_t0, MINE, THEIRS).astype(np.int8))
        out["valid"].append(valid[pos])
        out["reference_sq"].append(reference_sq)
        out["reference_row"].append(ref)
        out["control_row"].append(rows[(items["is_control"].numpy() == 1)[rows]][0])
    return {k: np.asarray(v) for k, v in out.items()}


def decode(acts_path: Path, t0: torch.Tensor) -> np.ndarray:
    """argmax probe class per (row, window offset, square). fp32 -- fp16 overflows."""
    probe = board_probe(PROBE_LAYER, flat=True).float()
    # mmap: the rollout tree's activation tensor is 4.2 GB and is only gathered
    # from, one batch of window positions at a time.
    acts = torch.load(acts_path, map_location="cpu", weights_only=True, mmap=True)
    pos = t0.unsqueeze(1) + torch.as_tensor(WINDOW)
    out = torch.empty(len(acts), len(WINDOW), 64, dtype=torch.int8)
    for start in range(0, len(acts), 512):
        sl = slice(start, min(start + 512, len(acts)))
        z = acts[torch.arange(sl.start, sl.stop).unsqueeze(1), pos[sl]].float()
        out[sl] = torch.einsum("nwd,dqk->nwqk", z, probe).argmax(-1).to(torch.int8)
    return out.numpy()


def masked_rate(hit: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Share of masked squares where `hit`, NaN where the mask selects none."""
    n = mask.sum(-1)
    return np.where(n > 0, (hit & mask).sum(-1) / np.where(n > 0, n, 1), np.nan)


def sanity_rows(dec: np.ndarray, tables: dict, k: int) -> list[dict]:
    """Reference-branch accuracy under the two candidate frames, overall and by class.

    `frame_diagnostic` is the only comparison that separates them: the squares
    whose label the two frames disagree on, which exist exactly where the
    reference game has taken a forfeit before this position.
    """
    ref, valid = tables["reference_row"], tables["valid"]
    diagnostic = (tables["lab"] != tables["lab_parity"]) & valid[..., None]
    rows = []
    for frame, lab in (("engine_replay", tables["lab"]), ("parity", tables["lab_parity"])):
        correct = dec[ref] == lab
        rows += series(per_branch(defined(correct.mean(-1), valid), k),
                       frame=frame, square_class="all")
        for c, name in enumerate(SQUARE_CLASSES):
            rows += series(per_branch(masked_rate(correct, (tables["cls"] == c)
                                                  & valid[..., None]), k),
                           frame=frame, square_class=name)
        rows += series(per_branch(masked_rate(correct, diagnostic), k), frame=frame,
                       square_class="frame_diagnostic")
    rows += series(per_branch(defined(diagnostic.any(-1), valid), k),
                   frame="frames_differ", square_class="all")
    return rows


def parity(dec: np.ndarray, lab: np.ndarray, cls: np.ndarray,
           valid: np.ndarray) -> dict[str, np.ndarray]:
    """Decoded mover vs the reference's, voted over the squares no branch touched.

    A branch reading the board in the opposite frame swaps theirs and mine on
    every untouched disc at once, so the vote is a direct read of whose turn the
    branch thinks it is -- R3 turn confusion off the representation.
    """
    untouched = ((cls == SQUARE_CLASSES.index("untouched_theirs"))
                 | (cls == SQUARE_CLASSES.index("untouched_mine"))) & valid[..., None]
    n = untouched.sum(-1)
    agree = ((dec == lab) & untouched).sum(-1)
    swap = ((dec == 3 - lab) & untouched).sum(-1)
    return {"p_aligned": np.where(agree + swap > 0, agree > swap, np.nan),
            "frac_agree": np.where(n > 0, agree / np.where(n > 0, n, 1), np.nan),
            "frac_swap": np.where(n > 0, swap / np.where(n > 0, n, 1), np.nan)}


def per_game(values: np.ndarray, game_index: np.ndarray, n_games: int) -> np.ndarray:
    """Mean over each game's rows, NaN for games with none -- the pooled-anchor unit.

    A source game contributes one row per empty-square category, and in the rollout
    tree K of each, so this collapses the categories and the rollouts in one step.
    Rows where the reference has no board are NaN and drop out of both sums.
    """
    ok = np.isfinite(values)
    total = np.zeros((n_games, values.shape[1]))
    count = np.zeros((n_games, values.shape[1]))
    np.add.at(total, game_index, np.where(ok, values, 0.0))
    np.add.at(count, game_index, ok)
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="the 50-game copy under $DIR/smoke/")
    parser.add_argument("--variant", default="spliced", choices=VARIANTS,
                        help="rollout: every branch samples its own continuation "
                             "($DIR[/smoke]/rollout/), K per branch")
    args = parser.parse_args()
    variant = args.variant
    sampled = variant == "rollout"

    study = variant_dir("divergence", args.smoke, "" if variant == "spliced" else variant)
    items = torch.load(study / "branches" / "branches.pt", weights_only=True)
    games = sorted(items["game_id"].unique().tolist())
    group_index = group_key(items).numpy()
    k = int(items["rollout"].max()) + 1 if sampled else 1
    tables = group_tables(items, group_index, sampled)
    dec = decode(study / "acts" / f"resid_post_{PROBE_LAYER}.pt", items["t0"])

    game_index = np.searchsorted(games, items["game_id"].numpy())
    branch_square = np.array([to_string(t) for t in items["branch_token"].tolist()])
    roles = np.array([role(int(c), int(j)) for c, j in zip(items["category"], items["is_control"])])
    lab, cls = tables["lab"][group_index], tables["cls"][group_index]
    valid = tables["valid"][group_index]
    votes = parity(dec, lab, cls, valid)

    assert (dec[:, :R0] == dec[tables["reference_row"][group_index], :R0]).all(), \
        "branches of a group decode differently before t0"
    sanity = pd.DataFrame(sanity_rows(dec, tables, k))
    engine_all = sanity[(sanity.frame == "engine_replay") & (sanity.square_class == "all")]
    assert engine_all["mean"].min() > 0.95, \
        f"reference branch decodes the true board at only {engine_all['mean'].min():.3f}"

    out = repo_root() / "results" / "violation_divergence"
    out.mkdir(parents=True, exist_ok=True)
    # a smoke run never overwrites the real CSVs, and neither does a variant
    stem = "probe_smoke" if args.smoke else "probe"
    stem = stem if variant == "spliced" else f"{stem}_{variant}"
    print(f"{len(games)} games, {len(items['t0'])} rows, K={k}, "
          f"{1 - tables['valid'].mean():.2%} of window positions have no reference board")
    sanity.to_csv(out / f"{stem}_sanity.csv", index=False)

    parity_rows = []
    for branch in ["reference", "control_legal", *CATEGORIES[1:]]:
        rows = np.flatnonzero(roles == branch)
        for metric, values in votes.items():
            parity_rows += series(per_branch(values[rows], k), branch=branch, metric=metric)
    pd.DataFrame(parity_rows).to_csv(out / f"{stem}_parity.csv", index=False)

    square_rows, class_rows, persistence_rows, contract = [], [], [], []
    aligned_r0 = votes["p_aligned"][:, R0] == 1  # NaN (no untouched disc decoded) is not aligned

    # The ceiling anchor: a disc the reference branch really did see placed.
    square_rows += series(
        per_branch(defined(dec[tables["reference_row"], :, tables["reference_sq"]] != EMPTY,
                           tables["valid"]), k),
        category="reference_move", role="reference", metric="p_disc")

    pooled: dict[tuple[str, str], list] = {}
    for category in CATEGORIES[1:]:
        rows = np.flatnonzero(roles == category)
        g, sq, v = game_index[rows], branch_square[rows], valid[rows]
        target = {"illegal": dec[rows, :, sq],
                  "reference": dec[tables["reference_row"][group_index[rows]], :, sq],
                  "control": dec[tables["control_row"][group_index[rows]], :, sq]}
        truth, mover_colour = lab[rows, :, sq], tables["mover_colour"][group_index[rows]]
        for role_name, d in target.items():
            metrics = {"p_disc": d != EMPTY,
                       "p_mover_colour": d == mover_colour,
                       "p_disagree_reference": d != truth}
            for metric, values in metrics.items():
                values = defined(values, v)
                square_rows += series(per_branch(values, k), category=category,
                                      role=role_name, metric=metric)
                if category in EMPTY_SQUARE_CATEGORIES:
                    pooled.setdefault((role_name, metric), []).append((g, values))
            square_rows += series(
                per_branch(np.where(aligned_r0[rows, None] & v, d == mover_colour, np.nan), k),
                category=category, role=role_name, metric="p_mover_colour_frame_aligned")

        for c, name in enumerate(SQUARE_CLASSES[:-1]):
            mask = (cls[rows] == c) & v[..., None]
            for role_name, source in (("illegal", rows),
                                      ("reference", tables["reference_row"][group_index[rows]])):
                class_rows += series(per_branch(masked_rate(dec[source] == lab[rows], mask), k),
                                     category=category, role=role_name, square_class=name)

        # An offset with no reference board breaks the run: we stop counting there
        # rather than call an undefined comparison a disagreement.
        run = np.cumprod((target["illegal"][:, R0:] != truth[:, R0:]) & v[:, R0:], axis=1).sum(1)
        n_branch = len(rows) // k
        persistence_rows += [{"category": category, "n": n_branch, "stat": stat, "value": value}
            for stat, value in [("median_run", float(np.median(run))),
                                ("q25_run", float(np.quantile(run, 0.25))),
                                ("q75_run", float(np.quantile(run, 0.75))),
                                ("mean_run", float(run.mean()))]
            ] + [{"category": category, "n": n_branch, "stat": f"p_run_ge_{plies}",
                  "value": float((run >= plies).mean())} for plies in PERSIST_AT]
        contract += [
            {"key": "p_illegal_disc_decoded", "category": category, "n": n_branch,
             "value": float((target["illegal"][:, R0] != EMPTY).mean())},
            {"key": "parity_agree_r0", "category": category, "n": int(
                np.isfinite(votes["p_aligned"][rows, R0]).sum()) // k,
             "value": float(np.nanmean(votes["p_aligned"][rows, R0]))},
            {"key": "p_mover_colour_r0", "category": category, "n": n_branch,
             "value": float((target["illegal"][:, R0] == mover_colour[:, R0]).mean())},
            {"key": "p_mover_colour_r0_frame_aligned", "category": category,
             "n": int(aligned_r0[rows].sum()) // k,
             "value": float((target["illegal"][aligned_r0[rows], R0]
                             == mover_colour[aligned_r0[rows], R0]).mean())},
            {"key": "median_persistence_plies", "category": category, "n": n_branch,
             "value": float(np.median(run))}]

    for (role_name, metric), parts in pooled.items():
        square_rows += series(
            per_game(np.concatenate([v for _, v in parts]),
                     np.concatenate([g for g, _ in parts]), len(games)),
            category="empty_pooled", role=role_name, metric=metric)

    pd.DataFrame(square_rows).to_csv(out / f"{stem}_illegal_square.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out / f"{stem}_square_class.csv", index=False)
    pd.DataFrame(persistence_rows).to_csv(out / f"{stem}_persistence.csv", index=False)
    contract += [{"key": "n_games", "category": "", "n": len(games), "value": len(games)},
                 {"key": "n_branches", "category": "", "n": len(items["t0"]) // k,
                  "value": len(items["t0"]) // k}]
    pd.DataFrame(contract).to_csv(out / f"{stem}_contract.csv", index=False)
    print(pd.DataFrame(contract).pivot_table(index="category", columns="key", values="value"))


if __name__ == "__main__":
    main()
