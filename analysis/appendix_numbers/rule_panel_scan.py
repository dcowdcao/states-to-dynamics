"""A.9: the rule panel's position scan (Fig 1(a)), with a tie-break the criteria lacked.

Over t in [14, 26] of the first 800 seed-42 games, a position qualifies when the mover has a
legal move, the move the game played flips discs in more than one direction, and one square
of each drawn category (R1 `occupied_opp`, R2 `empty_no_bracket`, R3 `opponent_only`) can be
chosen so that all four squares are at least three cells apart (Chebyshev). Per position the
squares are the triple with the largest minimum pairwise distance, ties to the lowest squares.

The criteria admit ~1,900 positions and name no tie-break. This one ranks by that minimum
distance, then by the played move's closed directions, then by (game, t): squares far apart
and a move that visibly flanks. Fig 1(a)'s game 606, t = 16 qualifies, at the rank the CSV
gives, but no ranking over one or two of these features picks it: the original scan was never
committed and its pick is unrecorded, and its R1 square b2 (2 cells from b4) was moved by
hand afterwards.

Writes results/appendix_numbers/rule_panel_scan.csv: every qualifying position, ranked.

Usage: python analysis/appendix_numbers/rule_panel_scan.py
"""

from collections import Counter
from itertools import product

import pandas as pd

from spade.common.paths import results_dir
from spade.figures.figure_othello_rules import SEED, name
from spade.othello.boards import flip_set, rays, replay
from spade.othello.games import collect_legal, stream_games
from spade.othello.taxonomy import CATEGORIES, LEGAL, classify, flips

from circuits.othello_engine_utils import stoi_indices, to_string

OUT = results_dir("appendix_numbers") / "rule_panel_scan.csv"
DATASET = "adamkarvonen/othello_45MB_games"
N_GAMES, T_RANGE = 800, range(14, 27)
MIN_SEPARATION, MIN_DIRECTIONS = 3, 2
DRAWN = ("occupied_opp", "empty_no_bracket", "opponent_only")   # R1, R2, R3
RANKING = ["min_separation", "n_directions"]                       # descending; then (game, t)


def cells_apart(a: int, b: int) -> int:
    return max(abs(a // 8 - b // 8), abs(a % 8 - b % 8))


def separation(squares: tuple[int, ...]) -> int:
    return min(cells_apart(a, b) for i, a in enumerate(squares) for b in squares[i + 1:])


def scan(game_id: int, game: list[int], t: int) -> dict | None:
    board = replay(game[:t])
    mover = int(board.next_hand_color)
    if not any(board.state[s // 8, s % 8] == 0 and flips(board.state, s, mover) for s in range(64)):
        return None  # the mover forfeits
    played = to_string(game[t])
    n_directions = sum(ray.closed for ray in rays(board.state, played, mover))
    if classify(board, played) != LEGAL or n_directions < MIN_DIRECTIONS:
        return None
    category = {sq: CATEGORIES[classify(board, sq)] for sq in stoi_indices}
    candidates = [[sq for sq, c in category.items() if c == drawn] for drawn in DRAWN]
    triples = sorted(product(*candidates), key=lambda sq: (-separation((played, *sq)), sq))
    if not triples or separation((played, *triples[0])) < MIN_SEPARATION:
        return None
    r1, r2, r3 = triples[0]
    return {"game": game_id, "t": t, "mover": "black" if mover > 0 else "white",
            "played": name(played), "n_directions": n_directions,
            "n_flipped": len(flip_set(board.state, played, mover)),
            "min_separation": separation((played, r1, r2, r3)),
            "r1": name(r1), "r2": name(r2), "r3": name(r3)}


def main() -> None:
    games = collect_legal(stream_games(DATASET, SEED), N_GAMES, Counter())
    rows = [row for g, game in enumerate(games) for t in T_RANGE
            if (row := scan(g, game, t)) is not None]
    table = pd.DataFrame(rows).sort_values(
        RANKING + ["game", "t"], ascending=[False] * len(RANKING) + [True, True])
    table.insert(0, "rank", range(1, len(table) + 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    figure = table[(table.game == 606) & (table.t == 16)]
    print(f"{len(table)} qualifying positions; pick: game {table.iloc[0].game}, t {table.iloc[0].t}")
    print(f"Fig 1(a)'s 606/16:\n{figure.to_string(index=False)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
