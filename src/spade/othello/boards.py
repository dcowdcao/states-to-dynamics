"""Engine helpers shared by the rules causal experiments.

All functions speak vocab tokens (1..60) at the boundary and engine squares
(0..63) internally. "Regular" moves flip for the nominal mover — no engine
forfeit fallback; forfeit handling is explicit at each call site.
"""

from dataclasses import dataclass

import numpy as np

from spade.common.paths import add_karvonen_to_path

add_karvonen_to_path()

from spade.othello.taxonomy import flips

from circuits.othello_engine_utils import OthelloBoardState, eights, stoi, to_string

VOCAB_SQUARES = sorted(to_string(tok) for tok in range(1, 61))


def replay(tokens: list[int]) -> OthelloBoardState:
    board = OthelloBoardState()
    if tokens:
        board.update(to_string(tokens))
    return board


@dataclass(frozen=True)
class Ray:
    """One of the eight directions out of a square, as R2 sees it.

    `run` is the contiguous opponent discs, nearest first. Exactly one of `terminator` (the
    mover's own disc, so the bracket closes) and `blocker` (the empty square that breaks it)
    is set; both are None when the run walks off the board.
    """
    run: list[int]
    terminator: int | None
    blocker: int | None

    @property
    def closed(self) -> bool:
        return self.terminator is not None and bool(self.run)


def rays(state: np.ndarray, square: int, color: int) -> list[Ray]:
    """The eight rays out of `square` for `color`, in `eights` order."""
    r, c = square // 8, square % 8
    out = []
    for dr, dc in eights:
        run, cr, cc = [], r + dr, c + dc
        while 0 <= cr <= 7 and 0 <= cc <= 7 and state[cr, cc] != 0 and state[cr, cc] != color:
            run.append(8 * cr + cc)
            cr, cc = cr + dr, cc + dc
        on_board = 0 <= cr <= 7 and 0 <= cc <= 7
        stop = 8 * cr + cc if on_board else None
        closed = on_board and state[cr, cc] == color
        out.append(Ray(run, stop if closed else None, None if closed else stop))
    return out


def flip_set(state: np.ndarray, square: int, color: int) -> list[int]:
    """Squares flipped when `color` plays `square`."""
    return [sq for ray in rays(state, square, color) if ray.closed for sq in ray.run]


def regular_moves(state: np.ndarray, color: int) -> set[int]:
    """Empty squares where `color` flips at least one disc."""
    return {sq for sq in VOCAB_SQUARES
        if state[sq // 8, sq % 8] == 0 and flips(state, sq, color)}


def next_mover(state: np.ndarray, nominal: int) -> int:
    """R4: the colour that actually moves next, skipping a nominal mover who cannot flip.

    `nominal` is the engine's `next_hand_color`. Every mover-relative label needs this
    rather than move-index parity, which a single skip puts out of phase for the rest of
    the game.
    """
    return nominal if regular_moves(state, nominal) else -nominal


def rays_touch(square: int, targets: set[int]) -> bool:
    """Does any target lie on one of `square`'s eight full rays?"""
    r, c = square // 8, square % 8
    for dr, dc in eights:
        cr, cc = r + dr, c + dc
        while 0 <= cr <= 7 and 0 <= cc <= 7:
            if 8 * cr + cc in targets:
                return True
            cr, cc = cr + dr, cc + dc
    return False


def tokens_of(squares) -> list[int]:
    return sorted(stoi[sq] for sq in squares)
