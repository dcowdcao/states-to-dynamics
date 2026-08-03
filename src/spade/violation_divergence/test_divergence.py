"""Hand-worked tests for the divergence kernels: no data, no model, no GPU.

Usage: python -m spade.violation_divergence.test_divergence
"""

import torch

from spade.violation_divergence.divergence import (
    WINDOW,
    align,
    branch_cosine,
    direction_cosine,
    participation_ratio,
    persistence,
    relative_divergence,
    support_jaccard,
    topk_share,
)

T0_RANGE = (10, 34)  # branches.py's round-robin range; the window claim below rests on it


def test_identical_branches_do_not_diverge():
    torch.manual_seed(0)
    Z = torch.randn(4, 12, 8)
    assert (relative_divergence(Z, Z) == 0).all()
    F = Z * (torch.rand(4, 12, 8) > 0.5)  # a sparse code with the same support twice
    assert (support_jaccard(F, F) == 1).all()
    assert torch.isnan(support_jaccard(torch.zeros(2, 3, 8), torch.zeros(2, 3, 8))).all()


def test_zero_reference_norm_is_undefined():
    Zref = torch.zeros(1, 2, 4)
    Zref[0, 1] = torch.tensor([3.0, 4.0, 0.0, 0.0])
    Zb = Zref.clone()
    Zb[0, :] += torch.tensor([0.0, 0.0, 6.0, 8.0])
    d = relative_divergence(Zb, Zref)
    assert torch.isnan(d[0, 0]), d          # ‖z_ref‖ = 0
    assert d[0, 1] == 2.0, d                # ‖Δ‖ = 10, ‖z_ref‖ = 5


def test_constructed_delta_recovers_by_hand():
    # z_ref = (3,4,0,0) everywhere: ‖z_ref‖ = 5. Δ = (0,0,6,8) at every position.
    Zref = torch.tensor([3.0, 4.0, 0.0, 0.0]).expand(2, 6, 4).contiguous()
    Zb = Zref + torch.tensor([0.0, 0.0, 6.0, 8.0])
    assert (relative_divergence(Zb, Zref) == 2.0).all()
    assert torch.allclose(direction_cosine(Zb, Zref, torch.tensor([0, 3])),
                          torch.ones(2, 6))
    # Δ energy is (0,0,36,64): top-1 share 64/100, PR 100²/(36²+64²) = 10000/5392
    assert torch.allclose(topk_share(Zb, Zref, k=1), torch.full((2, 6), 0.64))
    assert torch.allclose(participation_ratio(Zb, Zref),
                          torch.full((2, 6), 10000 / 5392))


def test_direction_cosine_persistent_rotated_and_undefined():
    t0 = 2
    Zref = torch.zeros(1, 5, 4)
    Zb = Zref.clone()
    Zb[0, t0] = torch.tensor([1.0, 0.0, 0.0, 0.0])          # Δ(0)
    Zb[0, 3] = torch.tensor([7.0, 0.0, 0.0, 0.0])           # same direction, longer
    Zb[0, 4] = torch.tensor([0.5, 3.0 ** 0.5 / 2, 0.0, 0.0])  # rotated 60°
    cos = direction_cosine(Zb, Zref, torch.tensor([t0]))
    assert torch.isnan(cos[0, :t0]).all(), cos               # Δ = 0 before t0
    assert torch.allclose(cos[0, t0:], torch.tensor([1.0, 1.0, 0.5]), atol=1e-6), cos


def test_support_jaccard_matches_the_hand_count():
    Fb, Fref = torch.zeros(1, 1, 8), torch.zeros(1, 1, 8)
    Fb[0, 0, [0, 1, 2, 3]] = 1.0
    Fref[0, 0, [2, 3, 4]] = 1.0
    assert support_jaccard(Fb, Fref)[0, 0] == 2 / 5  # |∩| = 2, |∪| = 5


def test_delta_on_exactly_k_coordinates():
    k, F = 5, 64
    Fref = torch.zeros(1, 1, F)
    Fb = Fref.clone()
    Fb[0, 0, torch.arange(k) * 7] = 2.0  # equal magnitudes, exact in binary
    assert topk_share(Fb, Fref, k=k)[0, 0] == 1.0
    assert participation_ratio(Fb, Fref)[0, 0] == float(k)
    assert topk_share(Fb, Fref, k=k - 1)[0, 0] == (k - 1) / k
    assert torch.isnan(participation_ratio(Fref, Fref))[0, 0]


def test_align_maps_offset_r_to_position_t0_plus_r():
    values = torch.arange(3 * 59, dtype=torch.float).reshape(3, 59) % 59
    values += 100 * torch.arange(3).unsqueeze(1)  # values[i, s] = 100 i + s
    t0 = torch.tensor([10, 34, 20])
    out = align(values, t0, (-2, -1, 0, 1, 24))
    assert out[0].tolist() == [8, 9, 10, 11, 34], out[0]
    assert out[1].tolist() == [132, 133, 134, 135, 158], out[1]
    assert out[2].tolist() == [218, 219, 220, 221, 244], out[2]


def test_align_never_wraps_off_the_board():
    values = torch.arange(2 * 59, dtype=torch.float).reshape(2, 59)
    out = align(values, torch.tensor([1, 57]), (-2, 0, 24))
    assert torch.isnan(out[0, 0]), out  # position -1 must not wrap to 58
    assert torch.isnan(out[1, 2]), out  # position 81 is off the board
    assert out[0, 1] == 1.0 and out[1, 1] == 59 + 57


def test_window_stays_on_the_board_for_every_t0():
    """The design's no-attrition claim: t0 ∈ [10, 34] and r ∈ [-8, 24] never
    leave positions [2, 58], so n is constant across the x-axis."""
    lo = T0_RANGE[0] + min(WINDOW)
    hi = T0_RANGE[1] + max(WINDOW)
    assert (lo, hi) == (2, 58), (lo, hi)
    assert lo >= 1, "position 0 is the attention sink and is never used"


def test_persistence_peak_half_life_and_plateau():
    curve = torch.zeros(3, len(WINDOW))
    post = {r: i for i, r in enumerate(WINDOW) if r >= 0}
    # row 0: peaks 1.0 at r = 2, first drops to <= 0.5 at r = 5 (3 plies after
    # the peak, which is what half_life reports), plateaus at 0.2
    profile = [0.5, 0.8, 1.0, 0.9, 0.6, 0.45]
    for r, v in enumerate(profile):
        curve[0, post[r]] = v
    for r in range(len(profile), 25):
        curve[0, post[r]] = 0.2
    # row 1: never decays. row 2: a pre-t0 spike that must not become the peak.
    for r in range(25):
        curve[1, post[r]] = 1.0
        curve[2, post[r]] = 0.25
    curve[2, WINDOW.index(-8)] = 5.0

    p = persistence(curve, WINDOW)
    assert p["peak"].tolist() == [1.0, 1.0, 0.25], p["peak"]
    assert p["peak_r"].tolist() == [2, 0, 0], p["peak_r"]
    # elapsed since the peak, not the absolute offset: r = 5 minus peak_r = 2
    assert p["half_life"].tolist() == [3.0, torch.inf, torch.inf], p["half_life"]
    assert torch.allclose(p["plateau"], torch.tensor([0.2, 1.0, 0.25])), p["plateau"]


def test_branch_cosine_agrees_opposes_and_self_identifies():
    ref = torch.zeros(4, 1, 3)
    control = torch.tensor([[[1.0, 0.0, 0.0]]]).expand(4, 1, 3)
    branch = torch.tensor([
        [[2.0, 0.0, 0.0]],    # same direction as the control, larger
        [[-1.0, 0.0, 0.0]],   # opposite
        [[0.0, 1.0, 0.0]],    # orthogonal
        [[1.0, 0.0, 0.0]],    # is the control
    ])
    cos = branch_cosine(branch, control, ref)
    assert torch.allclose(cos.flatten(), torch.tensor([1.0, -1.0, 0.0, 1.0]), atol=1e-6), cos
    # a branch that does not move has no direction, so the cosine is undefined
    assert branch_cosine(ref, control, ref).isnan().all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"{len(tests)} tests passed")
