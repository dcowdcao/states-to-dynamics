"""A5 red team: independent re-derivation of the bracket-flip numbers, and the attacks on them.

Every board fact is recomputed from the event prefixes with the engine, and every
decode is written from the saved probe tensors, so nothing here depends on the
analysis agents' modules. A third probe is fitted from scratch on A2's probe set,
because the two committed probes were both trained on ARENA games while the events
come from the Karvonen corpus.

`SPADE_REPO` points at a checkout carrying `external/`, `reference/` and
`analysis/linear_probe/activations/`; a worktree branched before the bracket-flip
prep commits does not have them.

Usage: SPADE_REPO=<checkout> python -m spade.bracket_flip.redteam --out report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spade.common.paths import repo_root

REPO = Path(os.environ.get("SPADE_REPO", repo_root()))
sys.path.insert(0, str(REPO / "external" / "Karvonen_Research"))

from spade.common.study import study_dir  # noqa: E402
from spade.othello.boards import flip_set, rays_touch, regular_moves, replay  # noqa: E402

from circuits.othello_engine_utils import to_string  # noqa: E402

EMPTY, THEIRS, MINE = 0, 1, 2
BLACK, WHITE = 1, 2  # absolute-basis classes; 0 stays empty
CODE_EMPTY, CODE_PLACED, CODE_FLIPPED, CODE_MOVER, CODE_OPP = 0, 1, 2, 3, 4
LAYER, N_LAYERS = 5, 8
ARENA = REPO / ("reference/research_materials/ARENA/chapter1_transformer_interp"
                "/exercises/part53_othellogpt/board_seqs_id_small.npy")
LINEAR_ACTS = REPO / "analysis" / "linear_probe" / "activations"
ARENA_GAMES, ARENA_TRAIN_GAMES = 200, 160  # the committed probe's 8000/2000 split
PROBE_TRAIN_GAMES = 320  # of A2's 400, leaving 80 games held out for my own probe
BOOT = 2000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- loading

def condition(name: str) -> torch.Tensor:
    return torch.load(study_dir("bracket") / "events" / f"{name}.pt", map_location="cpu",
                      weights_only=True)


def probe_set() -> dict[str, torch.Tensor]:
    d = study_dir("bracket") / "probe"
    names = ("all_layer_activations", "board_states_relative", "board_states_absolute",
             "game_id", "mover_next", "tokens")
    return {n: torch.load(d / f"{n}.pt", map_location="cpu", weights_only=True) for n in names}


def tpr_checkpoint(layer: int) -> dict:
    """The retired 8,000-row ARENA fits, by raw path: this file audits what was
    shipped, so it must not follow the study onto the canonical probe."""
    return torch.load(study_dir("bracket") / "tpr_probes" / f"tpr_r52_f2_relative_layer{layer}.pt",
                      map_location="cpu", weights_only=True)


# ---------------------------------------------------------------- engine re-derivation

def relative(state: np.ndarray, mine: int) -> np.ndarray:
    flat = state.reshape(64)
    return np.where(flat == 0, EMPTY, np.where(flat == mine, MINE, THEIRS))


def absolute(state: np.ndarray) -> np.ndarray:
    flat = state.reshape(64)
    return np.where(flat == 0, EMPTY, np.where(flat == 1, BLACK, WHITE))


def codes_of(state: np.ndarray, move_square: int, square_s: int, mover: int) -> str:
    flat = state.reshape(64)
    out = []
    for sq in range(64):
        if sq == move_square:
            out.append(CODE_PLACED)
        elif sq == square_s:
            out.append(CODE_FLIPPED)
        elif flat[sq] == 0:
            out.append(CODE_EMPTY)
        elif flat[sq] == mover:
            out.append(CODE_MOVER)
        else:
            out.append(CODE_OPP)
    return "".join(map(str, out))


def rederive(ev: pd.DataFrame, prefs: list[list[int]]) -> tuple[dict, dict]:
    """Oracle labels for the three conditions, and every integrity check re-run from scratch."""
    n = len(ev)
    orc = {k: np.zeros((n, 64), dtype=np.int64)
           for k in ("before", "after", "control", "before_abs", "after_abs")}
    orc["mover"] = np.zeros(n, dtype=np.int64)
    bad = dict.fromkeys((
        "mover_mismatch", "t_below_5", "forfeit_at_t", "forfeit_at_t1", "control_forfeits",
        "flip_size", "s_mismatch", "s_not_opponent_before", "s_not_mover_after",
        "two_squares_changed", "control_flips_s", "control_flip_set", "n_legal",
        "kind_mismatch", "codes_mismatch", "s_not_adjacent_to_m", "move_not_regular"), 0)
    parity_wrong = 0

    for i, (row, P) in enumerate(zip(ev.itertuples(), prefs)):
        b = replay(P)
        e = b.next_hand_color
        m, mp = to_string(int(row.move)), to_string(int(row.control_move))
        legal = regular_moves(b.state, e)
        flips = flip_set(b.state, m, e)
        after = replay(P + [int(row.move)]).state
        control = replay(P + [int(row.control_move)]).state
        ctrl_flips = flip_set(b.state, mp, e)

        orc["mover"][i] = e
        orc["before"][i] = relative(b.state, e)
        orc["after"][i] = relative(after, -e)
        orc["control"][i] = relative(control, -e)
        orc["before_abs"][i] = absolute(b.state)
        orc["after_abs"][i] = absolute(after)

        bad["mover_mismatch"] += e != row.mover
        bad["t_below_5"] += row.t < 5
        bad["forfeit_at_t"] += not legal
        bad["move_not_regular"] += m not in legal
        bad["forfeit_at_t1"] += not regular_moves(after, -e)
        bad["control_forfeits"] += not regular_moves(control, -e)
        bad["flip_size"] += len(flips) != 1
        bad["s_mismatch"] += not flips or flips[0] != row.square_s
        bad["s_not_opponent_before"] += b.state.reshape(64)[row.square_s] != -e
        bad["s_not_mover_after"] += after.reshape(64)[row.square_s] != e
        bad["two_squares_changed"] += int((b.state != after).sum()) != 2
        bad["control_flips_s"] += row.square_s in ctrl_flips
        bad["control_flip_set"] += (" ".join(map(str, sorted(ctrl_flips)))
                                    != str(row.control_flip_set))
        bad["n_legal"] += len(legal) != row.n_legal
        bad["kind_mismatch"] += ("loose" if rays_touch(row.square_s, {mp, *ctrl_flips})
                                 else "strict") != row.control_kind
        bad["codes_mismatch"] += codes_of(b.state, m, row.square_s, e) != row.control_squares
        bad["s_not_adjacent_to_m"] += max(abs(m // 8 - row.square_s // 8),
                                          abs(m % 8 - row.square_s % 8)) != 1
        parity_wrong += (1 if row.t % 2 == 0 else -1) != e

    checks = {"n": n, "failures": {k: int(v) for k, v in bad.items()},
              "parity_names_wrong_mover": int(parity_wrong)}
    return orc, checks


# ---------------------------------------------------------------- decoders

def linear_logits(h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """h [n, d] against a [d, 8, 8, 3] or [d, 64, 3] probe -> [n, 64, 3]."""
    return torch.einsum("nd,dsc->nsc", h, w.reshape(w.shape[0], 64, 3))


def tpr_logits(h: torch.Tensor, ckpt: dict) -> torch.Tensor:
    """logit(s, c) = r_s^T M(h) f_c, from the saved tensors alone."""
    sd = ckpt["probe_state_dict"]
    binding = torch.einsum("nd,drf->nrf", h, sd["binding_map"])
    return torch.einsum("nrf,sr,cf->nsc", binding,
                        sd["role_embeddings"].reshape(64, -1), sd["filler_embeddings"])


def fit_linear(acts: torch.Tensor, labels: torch.Tensor, epochs: int, seed: int = 0) -> torch.Tensor:
    """[d, 64, 3] probe by full-batch Adam -- train_linear_probe.py's recipe, my own code."""
    torch.manual_seed(seed)
    w = (torch.randn(acts.shape[1], 64, 3) * 0.01).requires_grad_(True)
    opt = torch.optim.Adam([w], lr=1e-2)
    target = labels.reshape(-1)
    for _ in range(epochs):
        loss = torch.nn.functional.cross_entropy(
            torch.einsum("nd,dsc->nsc", acts, w).reshape(-1, 3), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return w.detach()


# ---------------------------------------------------------------- statistics

def boot(mask: np.ndarray, groups: np.ndarray | None = None) -> list[float]:
    """[point, lo, hi]; `groups` clusters by event so square-events are not counted as n."""
    if groups is None:
        stats = mask[RNG.integers(0, len(mask), size=(BOOT, len(mask)))].mean(1)
    else:
        sizes = np.bincount(groups)
        assert sizes.min() > 0, "an event contributed no squares; reduceat would misalign"
        sums = np.add.reduceat(mask.astype(np.float64), np.concatenate([[0], np.cumsum(sizes)[:-1]]))
        draws = RNG.integers(0, len(sizes), size=(BOOT, len(sizes)))
        stats = sums[draws].sum(1) / sizes[draws].sum(1)
    return [float(mask.mean()), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def confusion(pred: np.ndarray, true: np.ndarray) -> list[list[int]]:
    m = np.zeros((3, 3), dtype=np.int64)
    np.add.at(m, (true.reshape(-1), pred.reshape(-1)), 1)
    return m.tolist()


def discs(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(codes, (CODE_MOVER, CODE_OPP))
    return mask, np.repeat(np.arange(len(codes)), mask.sum(1))


def contract(pred_b: np.ndarray, pred_a: np.ndarray, orc: dict, codes: np.ndarray,
             s: np.ndarray) -> dict:
    """The design doc's cross-validation keys, each beside its stricter reading."""
    rows = np.arange(len(s))
    sb, sa = pred_b[rows, s], pred_a[rows, s]
    ok_b, ok_a = sb == orc["before"][rows, s], sa == orc["after"][rows, s]
    mask, ev_idx = discs(codes)
    db, da = pred_b[mask], pred_a[mask]
    tb, ta = orc["before"][mask], orc["after"][mask]

    unchanged_discs = np.zeros(len(s), dtype=np.int64)
    np.add.at(unchanged_discs, ev_idx, (db == da).astype(np.int64))
    empty = codes == CODE_EMPTY
    placed = codes == CODE_PLACED

    return {
        "n_events": int(len(s)),
        "p_s_unchanged_rel": boot(sb == sa),
        # relabelling to absolute colour is a bijection given the mover, so the contract's
        # p_s_absolute_correct is exactly "s decoded right before and after"; triviality() checks it
        "p_s_absolute_correct": boot(ok_b & ok_a),
        "p_s_correct_before": boot(ok_b),
        "p_s_correct_after": boot(ok_a),
        "p_unflipped_swap_rel": boot((db != da) & (db != EMPTY) & (da != EMPTY), ev_idx),
        "p_unflipped_swap_rel_strict": boot((db == tb) & (da == ta), ev_idx),
        "p_unflipped_changed": boot(db != da, ev_idx),
        "p_s_unique_unchanged_disc": boot((sb == sa) & (unchanged_discs == 0)),
        "mean_other_unchanged_discs": float(unchanged_discs.mean()),
        "n_discs_per_event": float(mask.sum(1).mean()),
        "independence_product": float(ok_b.mean() * ok_a.mean()),
        # empties must NOT swap: a blanket position-parity bias in the probe would move them too
        "p_empty_unchanged": boot(pred_b[empty] == pred_a[empty], np.repeat(rows, empty.sum(1))),
        "p_placed_empty_then_theirs": boot(
            (pred_b[placed] == EMPTY) & (pred_a[placed] == THEIRS)),
    }


def to_absolute(pred: np.ndarray, mine: np.ndarray) -> np.ndarray:
    """Relative prediction -> absolute colour, composed with the colour that moves next."""
    colour = np.where(pred == MINE, mine[:, None], -mine[:, None])
    return np.where(pred == EMPTY, EMPTY, np.where(colour == 1, BLACK, WHITE))


def triviality(pred_b: np.ndarray, pred_a: np.ndarray, pred_abs_b: np.ndarray,
               pred_abs_a: np.ndarray, orc: dict, codes: np.ndarray, s: np.ndarray) -> dict:
    """What the measurement adds over a board-state oracle, which scores 1.0 on every key.

    The absolute-colour statement is measured twice: by composing the relative decode
    with the mover, and with a probe trained directly on absolute labels. The first is a
    bijection of the relative decode and is expected to be identical to it; the second is
    the design doc's basis test and is the only one that reports the model.
    """
    rows = np.arange(len(s))
    mask, _ = discs(codes)
    opp = codes == CODE_OPP
    e = orc["mover"]  # s holds -e before and e after

    relabel_ok = ((to_absolute(pred_b, e)[rows, s] == orc["before_abs"][rows, s])
                  & (to_absolute(pred_a, -e)[rows, s] == orc["after_abs"][rows, s]))
    direct_ok = ((pred_abs_b[rows, s] == orc["before_abs"][rows, s])
                 & (pred_abs_a[rows, s] == orc["after_abs"][rows, s]))

    changed = (pred_b != pred_a).sum(1)
    oracle_changed = (orc["before"] != orc["after"]).sum(1)
    return {
        "oracle_value_of_every_key": 1.0,
        "p_s_absolute_correct_by_relabel": boot(relabel_ok),
        "p_s_absolute_correct_by_absolute_probe": boot(direct_ok),
        "relabel_equals_relative_both_correct": bool((relabel_ok == (
            (pred_b[rows, s] == orc["before"][rows, s])
            & (pred_a[rows, s] == orc["after"][rows, s]))).all()),
        "acc_at_s_after": boot(pred_a[rows, s] == orc["after"][rows, s]),
        "acc_at_unflipped_opp_after": boot(
            (pred_a[opp] == orc["after"][opp]), np.repeat(rows, opp.sum(1))),
        "acc_at_all_discs_after": boot(pred_a[mask] == orc["after"][mask],
                                       np.repeat(rows, mask.sum(1))),
        "squares_changing_decode_mean": float(changed.mean()),
        "squares_changing_oracle_mean": float(oracle_changed.mean()),
    }


# ---------------------------------------------------------------- attacks

def leakage(prefs: list[list[int]], probe: dict, t: np.ndarray) -> tuple[dict, np.ndarray]:
    """Identical prefixes give identical activations, so a shared prefix is the leak channel.

    Both committed probes were fitted on ARENA games and A1 selected `weight_decay`
    on ARENA games 140-159, all inside the first 160 games; A2's own probe set is a
    disjoint slice of the Karvonen stream. Games being disjoint is not the same as
    prefixes being disjoint, which is what determines the activation.
    """
    arena = np.load(ARENA)[:ARENA_GAMES]
    train_pref = {tuple(g[:k]) for g in arena[:ARENA_TRAIN_GAMES] for k in range(5, 58)}
    tuned_pref = {tuple(g[:k]) for g in arena[140:ARENA_TRAIN_GAMES] for k in range(5, 58)}
    all_pref = {tuple(g[:k]) for g in arena for k in range(5, 58)}
    a2 = probe["tokens"].numpy()
    a2_pref = {tuple(int(x) for x in g[:k]) for g in a2 for k in range(5, 58)}
    keys = [tuple(P) for P in prefs]
    hit = np.array([k in train_pref or k in a2_pref for k in keys])
    return {
        "event_prefix_in_arena_train_160": int(sum(k in train_pref for k in keys)),
        "event_prefix_in_arena_wd_tuning_140_159": int(sum(k in tuned_pref for k in keys)),
        "event_prefix_in_arena_all_200": int(sum(k in all_pref for k in keys)),
        "event_prefix_in_a2_probe_400": int(sum(k in a2_pref for k in keys)),
        "any_contaminated": int(hit.sum()),
        "contaminated_t": sorted(int(v) for v in t[hit]),
        "a2_probe_prefix_in_arena_200": int(sum(
            tuple(int(x) for x in g[:k]) in all_pref for g in a2 for k in (5, 25, 55))),
        "duplicate_event_prefixes": int(pd.Series(keys).duplicated().sum()),
        "arena_openings": sorted({int(g[0]) for g in arena}),
        "karvonen_openings": sorted({P[0] for P in prefs}),
    }, hit


def alignment(before: torch.Tensor, after: torch.Tensor, ctrl: torch.Tensor, w: torch.Tensor,
              tpr: dict, codes: np.ndarray, s: np.ndarray, strict: np.ndarray) -> dict:
    """cos(delta, predicted flip direction) under both readings of "the flip direction".

    A: the control contrast act(P+m) - act(P+m'). s reads *theirs* in one branch and
    *mine* in the other, so the predicted direction is non-degenerate.
    B: before -> after. s's relative class does not change, so the direction the design
    doc names, pinv(M)[r_s (f_after - f_before)^T], is identically zero; what is
    measured instead is whether s swaps like everything else.
    """
    rows = np.arange(len(s))
    wl = w.reshape(w.shape[0], 64, 3)
    sd = tpr["probe_state_dict"]
    pinv = torch.linalg.pinv(sd["binding_map"].reshape(512, -1).T)
    roles, fillers = sd["role_embeddings"].reshape(64, -1), sd["filler_embeddings"]

    def tpr_dir(sq, frm, to):
        db = torch.einsum("nr,f->nrf", roles[sq], fillers[to] - fillers[frm])
        return db.reshape(len(sq), -1) @ pinv.T

    def cos(delta, direction):
        return torch.nn.functional.cosine_similarity(delta, direction, dim=-1).numpy()

    null_sq = np.array([RNG.choice(np.flatnonzero(codes[i] == CODE_OPP)) for i in rows])
    out = {}
    for tag, delta, frm, to in (("A_control_contrast", after - ctrl, MINE, THEIRS),
                                ("B_before_after", after - before, THEIRS, MINE)):
        flip_lin = cos(delta, wl[:, s, to].T - wl[:, s, frm].T)
        flip_tpr = cos(delta, tpr_dir(s, frm, to))
        out[tag] = {
            "cos_flip_median_linear": float(np.median(flip_lin)),
            "cos_null_median_linear": float(np.median(
                cos(delta, wl[:, null_sq, to].T - wl[:, null_sq, frm].T))),
            "cos_flip_median_tpr": float(np.median(flip_tpr)),
            "cos_null_median_tpr": float(np.median(cos(delta, tpr_dir(null_sq, frm, to)))),
            "cos_flip_median_tpr_strict": float(np.median(flip_tpr[strict])),
            "cos_flip_median_tpr_loose": float(np.median(flip_tpr[~strict])),
            "cos_flip_median_linear_strict": float(np.median(flip_lin[strict])),
            "cos_flip_median_linear_loose": float(np.median(flip_lin[~strict])),
        }
    d_ba, d_cb = after - before, ctrl - before
    swap = d_ba.mean(0)
    unit_cb = d_cb / d_cb.norm(dim=-1, keepdim=True)
    out["global_swap_variance_share"] = float(
        (d_ba @ (swap / swap.norm())).pow(2).sum() / d_ba.pow(2).sum())
    out["per_event_control_swap_variance_share"] = float(
        (d_ba * unit_cb).sum(-1).pow(2).sum() / d_ba.pow(2).sum())
    out["cos_delta_ba_delta_cb_median"] = float(np.median(cos(d_ba, d_cb)))
    out["mean_norms"] = {"h": float(before.norm(dim=-1).mean()),
                         "delta_before_after": float(d_ba.norm(dim=-1).mean()),
                         "delta_control": float((after - ctrl).norm(dim=-1).mean()),
                         "delta_control_minus_before": float(d_cb.norm(dim=-1).mean())}
    return out


def depth(before: torch.Tensor, after: torch.Tensor, orc: dict, codes: np.ndarray,
          s: np.ndarray, decode) -> dict:
    """Per-layer readout under four definitions of "strongest", which need not agree."""
    rows = np.arange(len(s))
    mask, _ = discs(codes)
    per_layer = {}
    for layer in range(N_LAYERS):
        pb = decode(before[:, layer], layer).argmax(-1).numpy()
        pa = decode(after[:, layer], layer).argmax(-1).numpy()
        sb, sa = pb[rows, s], pa[rows, s]
        db, da = pb[mask], pa[mask]
        per_layer[layer] = {
            "p_s_unchanged": float((sb == sa).mean()),
            "p_s_correct_both": float(((sb == orc["before"][rows, s])
                                       & (sa == orc["after"][rows, s])).mean()),
            "p_unflipped_swap": float(((db != da) & (db != EMPTY) & (da != EMPTY)).mean()),
            "p_disc_unchanged": float((db == da).mean()),
            "board_acc": float(((pb == orc["before"]).mean() + (pa == orc["after"]).mean()) / 2),
        }
    best = lambda key: int(max(per_layer, key=lambda l: per_layer[l][key]))  # noqa: E731
    return {"per_layer": per_layer, "peak_layer": {
        "by_p_s_unchanged": best("p_s_unchanged"),
        "by_p_s_correct_both": best("p_s_correct_both"),
        "by_p_unflipped_swap": best("p_unflipped_swap"),
        "by_board_acc": best("board_acc"),
        "by_specificity_margin": int(max(per_layer, key=lambda l: (
            per_layer[l]["p_s_unchanged"] - per_layer[l]["p_disc_unchanged"]))),
    }}


# ---------------------------------------------------------------- entrypoint

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    ev = pd.read_csv(study_dir("bracket") / "events" / "events.csv")
    prefs = [[int(tok) for tok in p.split()] for p in ev["prefix"]]
    s = ev["square_s"].to_numpy()
    codes = np.array([[int(ch) for ch in row] for row in ev["control_squares"]])
    orc, checks = rederive(ev, prefs)
    before, after, ctrl = (condition(k) for k in ("before", "after", "control"))
    report = {"integrity": checks, "row_alignment": {
        "identical_before_after_rows": int((before[:, LAYER] == after[:, LAYER]).all(1).sum()),
        "mean_abs_delta_L5": float((after[:, LAYER] - before[:, LAYER]).abs().mean())}}

    probe = probe_set()
    train = (probe["game_id"] < PROBE_TRAIN_GAMES).numpy()
    with torch.enable_grad():
        mine = {l: fit_linear(probe["all_layer_activations"][l][train],
                              probe["board_states_relative"][train].reshape(-1, 64), args.epochs)
                for l in range(N_LAYERS)}
        mine_abs = fit_linear(probe["all_layer_activations"][LAYER][train],
                              probe["board_states_absolute"][train].reshape(-1, 64), args.epochs)

    committed = torch.load(LINEAR_ACTS / "linear_probe.pt", map_location="cpu", weights_only=True)
    tpr = {l: tpr_checkpoint(l) for l in range(N_LAYERS)}
    decoders = {
        "committed_linear_ARENA": lambda h, l: linear_logits(h, committed),
        "tpr_A1_ARENA": lambda h, l: tpr_logits(h, tpr[l]),
        "redteam_linear_karvonen": lambda h, l: linear_logits(h, mine[l]),
    }

    held = ~train
    acts_held = probe["all_layer_activations"][LAYER][held]
    rel_held = probe["board_states_relative"][held].reshape(-1, 64).numpy()
    abs_held = probe["board_states_absolute"][held].reshape(-1, 64).numpy()
    report["heldout_accuracy_karvonen"] = {
        name: float((fn(acts_held, LAYER).argmax(-1).numpy() == rel_held).mean())
        for name, fn in decoders.items()}
    report["heldout_accuracy_karvonen"]["redteam_absolute"] = float(
        (linear_logits(acts_held, mine_abs).argmax(-1).numpy() == abs_held).mean())
    acts_train = probe["all_layer_activations"][LAYER][train]
    report["train_accuracy_karvonen"] = {
        "redteam_relative": float((linear_logits(acts_train, mine[LAYER]).argmax(-1).numpy()
                                   == probe["board_states_relative"][train].reshape(-1, 64).numpy()
                                   ).mean()),
        "redteam_absolute": float((linear_logits(acts_train, mine_abs).argmax(-1).numpy()
                                   == probe["board_states_absolute"][train].reshape(-1, 64).numpy()
                                   ).mean())}
    report["confusion_heldout"] = {
        name: confusion(fn(acts_held, LAYER).argmax(-1).numpy(), rel_held)
        for name, fn in decoders.items()}

    strict = (ev["control_kind"] == "strict").to_numpy()
    t = ev["t"].to_numpy()
    rows = np.arange(len(s))
    for key in ("contract", "confusion_events", "control_pair", "by_control_kind", "by_t"):
        report[key] = {}
    for name, fn in decoders.items():
        pb = fn(before[:, LAYER], LAYER).argmax(-1).numpy()
        pa = fn(after[:, LAYER], LAYER).argmax(-1).numpy()
        pc = fn(ctrl[:, LAYER], LAYER).argmax(-1).numpy()
        report["contract"][name] = contract(pb, pa, orc, codes, s)
        report["confusion_events"][name] = {"before": confusion(pb, orc["before"]),
                                            "after": confusion(pa, orc["after"])}
        report["control_pair"][name] = {
            "p_s_unchanged_before_vs_control": float((pb[rows, s] == pc[rows, s]).mean()),
            "p_s_correct_in_control": float((pc[rows, s] == orc["control"][rows, s]).mean()),
            "p_s_unchanged_before_vs_after": float((pb[rows, s] == pa[rows, s]).mean())}
        report["by_control_kind"][name] = {
            kind: contract(pb[m], pa[m], {k: v[m] for k, v in orc.items()}, codes[m], s[m])
            for kind, m in (("strict", strict), ("loose", ~strict))}
        report["by_t"][name] = {f"{lo}-{lo + 12}": float(
            (pb[rows, s] == pa[rows, s])[(t >= lo) & (t < lo + 13)].mean()) for lo in (5, 18, 31, 44)}

    pb5 = decoders["redteam_linear_karvonen"](before[:, LAYER], LAYER).argmax(-1).numpy()
    pa5 = decoders["redteam_linear_karvonen"](after[:, LAYER], LAYER).argmax(-1).numpy()
    report["triviality"] = {
        "redteam_linear_karvonen": triviality(
            pb5, pa5, linear_logits(before[:, LAYER], mine_abs).argmax(-1).numpy(),
            linear_logits(after[:, LAYER], mine_abs).argmax(-1).numpy(), orc, codes, s)}
    report["alignment"] = alignment(before[:, LAYER], after[:, LAYER], ctrl[:, LAYER],
                                    mine[LAYER], tpr[LAYER], codes, s, strict)
    report["depth"] = {name: depth(before, after, orc, codes, s, decoders[name])
                       for name in ("tpr_A1_ARENA", "redteam_linear_karvonen")}
    report["leakage"], hit = leakage(prefs, probe, t)
    clean = ~hit
    report["contract_uncontaminated"] = {
        name: contract(fn(before[:, LAYER], LAYER).argmax(-1).numpy()[clean],
                       fn(after[:, LAYER], LAYER).argmax(-1).numpy()[clean],
                       {k: v[clean] for k, v in orc.items()}, codes[clean], s[clean])
        for name, fn in decoders.items()}

    text = json.dumps(report, indent=2, default=float)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
