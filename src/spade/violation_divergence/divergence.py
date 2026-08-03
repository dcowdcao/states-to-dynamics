"""Divergence of a branch from its game's reference, in one representation space.

Δ(s) = z^b_s − z^ref_s at board position s; r = s − t0 is the offset from the
branch position. Inputs are [n, T, D] (activations or SAE codes) or [n, T]
(a per-position value), n branch rows, T board positions, D the space's width;
every kernel returns [n, T] or [n, len(window)]. Feed float32: an fp16 norm over
512 activation dims overflows.

Undefined is NaN rather than a filled-in default — a zero reference norm, an
empty support union, an all-zero Δ. Causal attention makes Δ(s < t0) exactly
zero, so `relative_divergence` is 0 there while `direction_cosine`, which needs
a direction, is NaN.

`topk_share` and `participation_ratio` read Δ through its squared coordinates:
both are functionals of one energy distribution p_j = Δ_j² / ‖Δ‖₂², the k-th
partial sum and the inverse-Simpson effective count. Sign is therefore
irrelevant — a feature switched off counts like one switched on.
"""

from collections.abc import Sequence

import torch

WINDOW = tuple(range(-8, 25))  # offsets r; the design's no-attrition window
PLATEAU_R = (20, 24)           # M3's late-game plateau, inclusive


def _ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """num/den, NaN where den is 0."""
    return torch.where(den > 0, num / den, torch.nan)


def relative_divergence(Zb: torch.Tensor, Zref: torch.Tensor) -> torch.Tensor:
    """‖Δ‖ / ‖z_ref‖ at each position."""
    return _ratio((Zb - Zref).norm(dim=-1), Zref.norm(dim=-1))


def direction_cosine(Zb: torch.Tensor, Zref: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
    """cos(Δ(s), Δ(t0)): does the offset introduced at t0 persist or rotate away?"""
    delta = Zb - Zref
    at_t0 = delta[torch.arange(len(delta), device=delta.device), t0].unsqueeze(1)
    return _ratio((delta * at_t0).sum(-1), delta.norm(dim=-1) * at_t0.norm(dim=-1))


def branch_cosine(Fb: torch.Tensor, Fc: torch.Tensor, Fref: torch.Tensor) -> torch.Tensor:
    """cos(Δ_b, Δ_c): do two branches leave the reference in the same direction?

    The illegal-vs-control form of M2. Near 1 means an illegal move displaces the
    state the way a legal one does and only the magnitude differs; near 0 means
    it leaves along a direction legal play does not use.
    """
    db, dc = Fb - Fref, Fc - Fref
    return _ratio((db * dc).sum(-1), db.norm(dim=-1) * dc.norm(dim=-1))


def support_jaccard(Fb: torch.Tensor, Fref: torch.Tensor) -> torch.Tensor:
    """|S_b ∩ S_ref| / |S_b ∪ S_ref| over the nonzero supports of two codes."""
    sb, sref = Fb != 0, Fref != 0
    return _ratio((sb & sref).sum(-1).float(), (sb | sref).sum(-1).float())


def topk_share(Fb: torch.Tensor, Fref: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Share of Δ's squared norm carried by its k largest |Δ_j|."""
    energy = (Fb - Fref).square()
    return _ratio(energy.topk(k, dim=-1).values.sum(-1), energy.sum(-1))


def participation_ratio(Fb: torch.Tensor, Fref: torch.Tensor) -> torch.Tensor:
    """(Σ Δ_j²)² / Σ Δ_j⁴: how many coordinates Δ effectively occupies."""
    energy = (Fb - Fref).square()
    return _ratio(energy.sum(-1).square(), energy.square().sum(-1))


def align(values: torch.Tensor, t0: torch.Tensor, window: Sequence[int]) -> torch.Tensor:
    """Event-lock a per-position quantity: column j is `values[i, t0[i] + window[j]]`.

    `values[:, s]` must be the quantity at board position s — passing an array
    whose column 0 is position 1 shifts every offset. Positions off the board
    are NaN; negative ones would otherwise wrap silently.
    """
    T = values.shape[1]
    pos = t0.unsqueeze(1) + torch.as_tensor(window, device=values.device)
    on_board = (pos >= 0) & (pos < T)
    return values.gather(1, pos.clamp(0, T - 1)).masked_fill(~on_board, torch.nan)


def persistence(delta_hat: torch.Tensor, window: Sequence[int]) -> dict[str, torch.Tensor]:
    """Peak, its offset, half-life and plateau of event-locked δ̂, over r >= 0.

    `half_life` is the number of plies **after the peak** that δ̂ takes to reach
    ½ peak, and +inf where it never does inside the window. Elapsed, not the
    absolute offset: it is a decay timescale, and `peak_r` is stored beside it,
    so the absolute position is a sum away while the reverse would silently
    inflate every game whose peak is late.

    The +inf is right-censoring, not a missing value: it keeps the column
    ordered, so a median or a censored fraction stays honest while a mean loudly
    returns inf instead of the downward-biased number NaN-dropping would give.
    """
    r = torch.as_tensor(window, device=delta_hat.device)
    post = delta_hat.masked_fill(r < 0, torch.nan)
    peak_j = torch.nan_to_num(post, nan=-torch.inf).argmax(1)
    peak = post.gather(1, peak_j.unsqueeze(1)).squeeze(1)
    peak_r = r[peak_j]
    decayed = (post <= 0.5 * peak.unsqueeze(1)) & (r > peak_r.unsqueeze(1))
    return {
        "peak": peak,
        "peak_r": peak_r,
        "half_life": torch.where(decayed.any(1),
                                 (r[decayed.int().argmax(1)] - peak_r).float(), torch.inf),
        "plateau": delta_hat.masked_fill(
            (r < PLATEAU_R[0]) | (r > PLATEAU_R[1]), torch.nan).nanmean(1),
    }
