"""Game corpora: stream records from HF and cut the prefixes every study branches from.

Imported by the rules, straightness, violation-divergence, manifold-capture and
bracket-flip pipelines — they all start from the same legal-prefix corpus, and they
must agree on it. `stream_games` and `classify_prefix` come from the rules study's
build_pairs.py; `collect_legal` and `first_illegal_pos` from the straightness study's
build_games.py.

Importing this module also forces IPv4 and puts the Karvonen fork on the path; several
source-repo modules imported it for that side effect alone. Keep those effects here and
explicit rather than scattered, but do not make them lazy — HF downloads on the cluster
depend on force_ipv4 having run before the first request.
"""

from collections import Counter

from spade.common.config import CTX_LEN
from spade.common.net import force_ipv4
from spade.common.paths import add_karvonen_to_path

force_ipv4()
add_karvonen_to_path()

from spade.othello.taxonomy import LEGAL, classify_all

from circuits.othello_engine_utils import OthelloBoardState, stoi, stoi_indices, to_string

T_MIN, T_MAX = 5, 57  # sequence length t+1 <= 58 < n_ctx = 59

# classify_all() index i must correspond to vocab token i+1
assert all(stoi[sq] == i + 1 for i, sq in enumerate(stoi_indices))


def stream_games(dataset_name: str, seed: int):
    from datasets import load_dataset
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    return iter(dataset.shuffle(seed=seed, buffer_size=10000))


def classify_prefix(game: list[int], t: int, skips: Counter):
    """Returns (board, square_cats) or None if the prefix is unusable."""
    if len(game) < t + 1:
        skips["short_game"] += 1
        return None
    if 0 in game[: t + 1]:
        skips["pass_token"] += 1
        return None
    board = OthelloBoardState()
    board.update(to_string(game[:t]))
    if not any(board.tentative_move(m) == 1 for m in board.get_valid_moves()):
        skips["forfeit_at_t"] += 1
        return None
    cats = classify_all(board)
    legal = {stoi_indices[i] for i, c in enumerate(cats) if c == LEGAL}
    assert legal == set(board.get_valid_moves()), \
        f"taxonomy/engine disagreement at t={t}: {legal} vs {board.get_valid_moves()}"
    return board, cats


def collect_legal(games, n: int, skips: Counter) -> list[list[int]]:
    out = []
    while len(out) < n:
        seq = next(games)["tokens"][:CTX_LEN]
        if len(seq) < CTX_LEN:
            skips["short_game"] += 1
        elif 0 in seq:
            skips["pass_token"] += 1
        else:
            out.append(seq)
    return out


def engine_legal(board: OthelloBoardState, square: int) -> bool:
    r, c = square // 8, square % 8
    if board.state[r, c] != 0:
        return False
    kind = board.tentative_move(square)
    if kind == 1:
        return True
    if kind == 0:
        return False
    # flips only for the opponent: engine-legal iff the mover must forfeit
    return all(board.tentative_move(m) != 1 for m in range(64))


def first_illegal_pos(seq: list[int]) -> int:
    board = OthelloBoardState()
    for pos, tok in enumerate(seq):
        sq = to_string(tok)
        if not engine_legal(board, sq):
            return pos
        board.umpire(sq)
    return -1
