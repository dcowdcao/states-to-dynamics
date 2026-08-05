"""Does a branch's continuation stay legal on the board that branch actually produced?

The spliced branch set gives every branch the *source game's* continuation,
`tokens[t0+1:]`. That continuation is a legal game only on the reference's board.
This script measures, from tokens and the engine alone, how much of it survives on
each branch's own board -- the defect that motivates the rollout variant -- and,
once the rollout set exists, how legal the model's own sampled continuations are.

Two things are deliberately *not* measured on the spliced set:

  * Suffix legality for an illegal branch. No board follows an illegal move, so
    there is nothing to replay the suffix on. The undefinedness is the finding;
    `suffix_legality_undefined.csv` reports what the engine does when asked to
    play the branch move instead of a number that pretends a board exists.
  * Anything needing a convention for what an illegal move does to the board.
    The one convention-free statement available -- occupancy only grows in
    Othello, so a move onto an occupied square is illegal under every convention
    -- is reported as `p_continuation_replays_branch_square`.

Usage: python analysis/violation_divergence/suffix_legality/measure.py [--smoke]
Writes results/violation_divergence/suffix_legality*.csv; see README.md here.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from spade.common.paths import results_dir
from spade.common.study import variant_dir
from spade.othello.boards import flip_set, replay
from spade.othello.games import engine_legal
from spade.othello.taxonomy import LEGAL, RULE_OF
from spade.violation_divergence.branches import ROLE_NAMES, role

from circuits.othello_engine_utils import to_string

N_BOOT = 1000
# analysis/violation_divergence/README.md, "What does not hold, and why", measured
# by the red team on the earlier one-control branch set (15531 rows).
README_CONTROL = {"p_first_move_legal": 0.768, "median_moves_legal": 2}


NO_BOARD, LEGAL_BOARD, PLACED_BOARD = "none", "legal", "placed convention"


@dataclass(frozen=True)
class Branch:
    game: int
    role: str
    rule: str
    n_continuation: int
    board: str                # NO_BOARD / LEGAL_BOARD / PLACED_BOARD
    moves_legal: int          # continuation moves playable before the first illegal one
    verdict: str              # what the engine does with the branch move itself
    replays_branch_square: bool
    moves_before_reuse: int   # convention-free: before landing on an occupied square


def verdict(board, square: int) -> str:
    """What the engine makes of the branch move, for the branches that have no board."""
    if board.state[square // 8, square % 8] != 0:
        return "square already holds a disc"
    if board.tentative_move(square) == 0:
        return "flips nothing for either colour"
    return "engine forfeit fallback replays it as the opponent's move"


def moves_legal(board, continuation: list[int]) -> int:
    """Continuation moves playable in order from `board` before the first illegal one."""
    for n, token in enumerate(continuation):
        square = to_string(token)
        if not engine_legal(board, square):
            return n
        board.umpire(square)
    return len(continuation)


def moves_before_reuse(occupied: set[int], continuation: list[int]) -> int:
    """Moves before the first landing on a square that already holds a disc.

    Occupancy never shrinks in Othello, so this needs no convention for what an
    illegal move did to the board: it is a lower bound on illegality for the
    branches whose board is undefined. `occupied` excludes the branch square,
    which is exactly the square the conventions disagree about.
    """
    for n, token in enumerate(continuation):
        square = to_string(token)
        if square in occupied:
            return n
        occupied.add(square)
    return len(continuation)


def placed(board, square: int):
    """The board an illegal move gives if the disc simply lands: written in the
    mover's colour, brackets it closes flipped, turn passed.

    A STATED CONVENTION, not a fact -- no legal play reaches this state. It is the
    reading the study's probe read-out supports (the model decodes the illegal
    square as occupied by the mover), and rollout numbers for illegal branches are
    only as meaningful as it is. Nothing on the spliced set uses it.
    """
    out = deepcopy(board)
    colour = board.next_hand_color
    for flipped in flip_set(board.state, square, colour):
        out.state[flipped // 8, flipped % 8] = colour
    out.state[square // 8, square % 8] = colour
    out.next_hand_color = -colour
    return out


def branches(meta: dict[str, torch.Tensor], place_illegal: bool) -> pd.DataFrame:
    """One row per branch. `place_illegal` builds an illegal branch's board under
    the `placed()` convention instead of leaving it undefined."""
    out = []
    for i in range(len(meta["t0"])):
        t0, category = int(meta["t0"][i]), int(meta["category"][i])
        tokens = meta["tokens"][i].long().tolist()
        square, continuation = to_string(tokens[t0]), tokens[t0 + 1:]
        prefix = replay(tokens[:t0])
        if category == LEGAL:
            board, kind = deepcopy(prefix), LEGAL_BOARD
            board.umpire(square)
        elif place_illegal:
            board, kind = placed(prefix, square), PLACED_BOARD
        else:
            board, kind = None, NO_BOARD
        occupied = {sq for sq in range(64) if prefix.state[sq // 8, sq % 8] != 0} - {square}
        name = role(category, int(meta["is_control"][i]))
        out.append(Branch(
            game=int(meta["game_id"][i]),
            role=name,
            rule=RULE_OF.get(name, "none"),
            n_continuation=len(continuation),
            board=kind,
            moves_legal=moves_legal(board, continuation) if board is not None else -1,
            verdict="legal move" if category == LEGAL else verdict(prefix, square),
            replays_branch_square=square in {to_string(t) for t in continuation},
            moves_before_reuse=moves_before_reuse(occupied, continuation)))
    return pd.DataFrame([asdict(b) for b in out])


def boot_ci(values: np.ndarray, game: np.ndarray, rng) -> tuple[float, float]:
    """Resamples games, not rows: the rollout set holds K rows per game and they
    share a prefix, a t0 and a board, so rows are not independent draws."""
    by_game = pd.DataFrame({"v": values, "g": game}).groupby("g").v.agg(["sum", "count"])
    total, n = by_game["sum"].to_numpy(), by_game["count"].to_numpy()
    draws = rng.integers(0, len(total), size=(N_BOOT, len(total)))
    return tuple(np.percentile(total[draws].sum(1) / n[draws].sum(1), [2.5, 97.5]))


def summarise(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Per branch role, over its rows (one per game in the spliced set, K per game
    in the rollout set). Legality columns are NaN where the branch has no board;
    where `board` reads "placed convention" they are only as good as that convention."""
    rows = []
    for name in ROLE_NAMES:
        d = df[df.role == name]
        if not len(d):
            continue
        replayed = d.board.iloc[0] != NO_BOARD
        first_legal = (d.moves_legal >= 1).to_numpy(float)
        lo, hi = boot_ci(first_legal, d.game.to_numpy(), rng) if replayed else (np.nan, np.nan)
        rows.append({
            "role": name,
            "rule": d.rule.iloc[0],
            "n": len(d),
            "board": d.board.iloc[0],
            "p_first_move_legal": first_legal.mean() if replayed else np.nan,
            "p_first_move_legal_lo": lo,
            "p_first_move_legal_hi": hi,
            "mean_moves_legal": d.moves_legal.mean() if replayed else np.nan,
            "median_moves_legal": d.moves_legal.median() if replayed else np.nan,
            "p_all_moves_legal": (d.moves_legal == d.n_continuation).mean() if replayed else np.nan,
            "p_continuation_replays_branch_square": d.replays_branch_square.mean(),
            "p_reuse_before_end": (d.moves_before_reuse < d.n_continuation).mean(),
            "mean_moves_before_reuse": d.moves_before_reuse.mean(),
        })
    return pd.DataFrame(rows)


def survival(df: pd.DataFrame) -> pd.DataFrame:
    """P(the first d continuation moves are all legal), per role and offset after t0.

    n_at_risk falls with d because a game's continuation is 58 - t0 long; the
    proportion is over the rows that have a move at that offset at all.
    """
    rows = []
    for name in ROLE_NAMES:
        d = df[(df.role == name) & (df.board != NO_BOARD)]
        if not len(d):
            continue
        for offset in range(1, int(d.n_continuation.max()) + 1):
            at_risk = d[d.n_continuation >= offset]
            rows.append({"role": name, "offset_after_t0": offset, "n_at_risk": len(at_risk),
                         "p_legal_through_offset": (at_risk.moves_legal >= offset).mean()})
    return pd.DataFrame(rows)


def reproduction_check(table: pd.DataFrame) -> None:
    """The README's control_legal figures, re-derived. A miss means this engine
    handling and the red team's disagree, and one of them is wrong."""
    row = table[table.role == "control_legal"].iloc[0]
    inside = row.p_first_move_legal_lo <= README_CONTROL["p_first_move_legal"] <= row.p_first_move_legal_hi
    median_ok = row.median_moves_legal == README_CONTROL["median_moves_legal"]
    print(f"\ncontrol_legal reproduction check (README: "
          f"{README_CONTROL['p_first_move_legal']:.3f}, median {README_CONTROL['median_moves_legal']})\n"
          f"  p_first_move_legal {row.p_first_move_legal:.4f} "
          f"[{row.p_first_move_legal_lo:.4f}, {row.p_first_move_legal_hi:.4f}]  "
          f"{'contains' if inside else 'DOES NOT CONTAIN'} the README value\n"
          f"  median_moves_legal {row.median_moves_legal:.0f}  "
          f"{'matches' if median_ok else 'DOES NOT MATCH'} the README value")
    if not (inside and median_ok):
        raise SystemExit(
            "\nSTOP. This does not reproduce the README, so this script's engine handling and\n"
            "the red team's disagree about legality. Do not read the tables below as a\n"
            "measurement of the branch set until that is resolved -- one of the two is wrong.")


def report(meta, stem: str, place_illegal: bool, rng) -> pd.DataFrame:
    df = branches(meta, place_illegal)
    table = summarise(df, rng)
    out = results_dir("violation_divergence")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{stem}.csv", index=False, float_format="%.6g")
    survival(df).to_csv(out / f"{stem}_survival.csv", index=False, float_format="%.6g")
    print(f"\n{stem}\n{table.to_string(index=False)}")
    undefined = df[df.board == NO_BOARD]
    if len(undefined):
        counts = undefined.groupby(["role", "verdict"]).size().rename("n").reset_index()
        counts.to_csv(out / f"{stem}_undefined.csv", index=False)
        print(f"\n{stem}: no board follows these branches, so suffix legality is undefined "
              f"for them.\nThe engine's own verdict on the branch move:"
              f"\n{counts.to_string(index=False)}")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="the 50-game set under $DIR/smoke/")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    stem = "suffix_legality_smoke" if args.smoke else "suffix_legality"

    spliced = variant_dir("divergence", args.smoke) / "branches" / "branches.pt"
    meta = torch.load(spliced, weights_only=True)
    print(f"spliced: {len(meta['t0'])} branches over "
          f"{len(meta['game_id'].unique())} games from {spliced}")
    table = report(meta, f"{stem}_spliced", False, rng)
    if args.smoke:
        print("\nsmoke: 50 games, so the README's control_legal figures are not checked")
    else:
        reproduction_check(table)

    rollout = variant_dir("divergence", args.smoke, "rollout") / "branches" / "branches.pt"
    if not rollout.exists():
        print(f"\nno rollout branch set at {rollout} -- spliced numbers only")
        return
    meta = torch.load(rollout, weights_only=True)
    assert "rollout" in meta, f"{rollout} has no rollout column"
    print(f"\nrollout: {len(meta['t0'])} branches over {len(meta['game_id'].unique())} games, "
          f"K={int(meta['rollout'].max()) + 1} samples each, from {rollout}")
    report(meta, f"{stem}_rollout", True, rng)
    print("\nIllegal branches have no board, so their rollout rows above use the placed() "
          "convention; p_reuse_before_end is the convention-free reading.")


if __name__ == "__main__":
    main()
