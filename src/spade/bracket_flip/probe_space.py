"""M1, M2 and M5 of the bracket-flip study: what the TPR reads off the flanked square.

Writes CSVs to results/fig07_flanking/ (m1_decode.csv, which the paper does not read, to
analysis/bracket_flip/results/); make_figures.py draws them and never recomputes anything.

Three conventions meet here and all three are load-bearing.

1. The probe's classes are mover-relative {empty, theirs, mine} with "mine" the
   player to move next. Playing m flips that perspective, so `before` is read
   from the mover's side and `after` and `control` from the opponent's. ORIENT
   carries that single fact into the absolute relabeling of the read-out.
2. Panel coordinates are gauge-invariant logit differences, never filler
   coordinates: the bilinear form is invariant under f -> Af, B -> BA^-1, and
   results/fig07_flanking/PROBE_REPORT.md measures the filler angle moving
   131.3 -> 104.5 degrees across seeds at constant accuracy.
3. Events are the independent unit (one per game), squares within an event are
   not, so every interval here is a cluster bootstrap over events.

The mover-relative read-out is the canonical probe, `spade.probes.load`. The
absolute basis and the layer -1 input space have no canonical corpus behind them,
so they are fit here, on this study's probe set, under `spade.probes.train`'s
protocol -- split by game, early stop on validation accuracy, best-validation
weights kept.

Usage: python -m spade.bracket_flip.probe_space
"""

import argparse
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from spade.common.config import MODEL_NAME
from spade.common.device import select_device
from spade.common.paths import add_karvonen_to_path, repo_root, results_dir
from spade.common.study import study_dir, variant_dir
from spade.probes import load as probes
from spade.probes.tpr import TPR
from spade.probes.train import CLASSES, Board, fit

add_karvonen_to_path()

from circuits.othello_engine_utils import to_string

EMPTY, THEIRS, MINE = 0, 1, 2  # index into CLASSES
BLACK, WHITE = 1, 2  # index into the absolute probe's classes (probe/manifest.json)
CONDITIONS = ("before", "after", "control")

# The pair of classes whose logit difference says who owns a square, per probe.
# The relative probe answers in the frame of whoever is to move; the absolute one
# answers in colours, and has to be oriented by the event's mover (+1 black) or
# black and white cancel across events.
OWNER = {"relative": (MINE, THEIRS), "absolute": (BLACK, WHITE)}

# +1 where "mine" is the mover, -1 where the move has handed the turn over.
# events.py drops forfeits at t and t+1, so `after` and `control` share the sign.
ORIENT = {"before": 1, "after": -1, "control": -1}

# events.csv `control_squares` codes, read on the board before m.
ROLE_CODE = {"empty": 0, "placed": 1, "flipped": 2, "unflipped_own": 3, "unflipped_opp": 4}
INPUT_LAYER = -1  # W_E[token] + W_pos[pos]: the input-space anchor of FIGURE_GUIDELINES 4.1
BLOCKS = tuple(range(8))
LAYERS = (INPUT_LAYER, *BLOCKS)
BASES = ("relative", "absolute")
MAIN_LAYER = 5
N_BOOT = 2000
SEED = 42               # the canonical protocol's seed, and this study's data seed
# The arrow panel's event set: a fixed quota per move index over the same [10, 34] window,
# in the same unit steps, that `move_legality` and `changepoint_localization` evaluate on, so
# the three figures rest on one t distribution. Move index is the strongest nuisance variable
# in this tree, and taking whatever the corpus happens to hold lets it back in.
FIELD_T_RANGE = (10, 34)
FIELD_PER_T = 40        # 25 indices x 40 = the 1000 events the panel averages over
FIELD_VARIANT = "field" # the larger corpus that quota needs, built by events.py --variant
N_TRAIN = 8000          # rows 0:8000 = games 0-159; games 160-399 are the eval half
VAL_GAMES = 0.15        # carved out of the training half, by game, to stop on
BATCH = 512             # 6.8k fitting rows, so the canonical 8192 would be one step per epoch

CACHE = study_dir("bracket") / "probe_space"
RESULTS = repo_root() / "analysis" / "bracket_flip" / "results"
FLANKING = results_dir("fig07_flanking")


# ---------------------------------------------------------------- data

def events(variant: str = "") -> pd.DataFrame:
    return pd.read_csv(variant_dir("bracket", variant=variant) / "events" / "events.csv")


def role_codes(ev: pd.DataFrame) -> np.ndarray:
    """[n, 64] over ROLE_CODE, engine-square order (square = 8 * row + col)."""
    return np.array([[int(c) for c in s] for s in ev.control_squares])


def event_acts(variant: str = "") -> dict[str, Tensor]:
    d = variant_dir("bracket", variant=variant) / "events"
    return {c: torch.load(d / f"{c}.pt", weights_only=True, map_location="cpu")
            for c in CONDITIONS}


@lru_cache(maxsize=1)
def embed_weights() -> tuple[Tensor, Tensor]:
    """(W_E, W_pos), cached so repeat runs touch neither the network nor the model."""
    path = CACHE / "embed_weights.pt"
    if path.exists():
        w = torch.load(path, weights_only=True)
        return w["W_E"], w["W_pos"]

    # The only two tensors this study needs from the model itself, fetched once.
    from spade.common.net import force_ipv4
    force_ipv4()
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
    w = {"W_E": model.W_E.clone(), "W_pos": model.W_pos.clone()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(w, path)
    return w["W_E"], w["W_pos"]


def input_event_acts(ev: pd.DataFrame) -> dict[str, Tensor]:
    """Residual stream before block 0, per condition. Exact, and needs no forward pass.

    `before` sits at position t-1 of run(P + m) and reads token P[t-1]; `after`
    and `control` sit at position t and read m and m'.
    """
    w_e, w_pos = embed_weights()
    t = torch.tensor(ev.t.to_numpy())
    token = {"before": torch.tensor([int(p.split()[-1]) for p in ev.prefix]),
             "after": torch.tensor(ev.move.to_numpy()),
             "control": torch.tensor(ev.control_move.to_numpy())}
    pos = {"before": t - 1, "after": t, "control": t}
    return {c: w_e[token[c]] + w_pos[pos[c]] for c in CONDITIONS}


@lru_cache(maxsize=1)
def _block_acts() -> Tensor:
    return torch.load(study_dir("bracket") / "probe" / "all_layer_activations.pt",
                      weights_only=True)


@lru_cache(maxsize=None)
def probe_acts(layer: int) -> Tensor:
    """[20000, 512] probe-set activations at `layer`, or in the input space."""
    if layer != INPUT_LAYER:
        return _block_acts()[layer]
    w_e, w_pos = embed_weights()
    tokens = torch.load(study_dir("bracket") / "probe" / "tokens.pt", weights_only=True).long()
    idx = torch.load(study_dir("bracket") / "probe" / "move_indices.pt",
                     weights_only=True).reshape(tokens.shape[0], -1)
    return (w_e[torch.gather(tokens, 1, idx)] + w_pos[idx]).reshape(-1, w_e.shape[1])


@lru_cache(maxsize=None)
def probe_labels(basis: str) -> Tensor:
    return torch.load(study_dir("bracket") / "probe" / f"board_states_{basis}.pt",
                      weights_only=True)


# ---------------------------------------------------------------- probes

def fit_split(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Fitting and early-stopping row indices, split BY GAME inside the training
    half. The eval half (games 160-399) is never touched here."""
    games = torch.load(study_dir("bracket") / "probe" / "game_id.pt",
                       weights_only=True).numpy()[:N_TRAIN]
    ids = np.unique(games)
    order = np.random.default_rng(seed).permutation(len(ids))
    held = np.isin(games, ids[order[int((1 - VAL_GAMES) * len(ids)):]])
    return np.flatnonzero(~held), np.flatnonzero(held)


def local(arm_class, layer: int, basis: str) -> torch.nn.Module:
    """One arm fit on this study's probe set under `spade.probes.train`'s protocol.

    Only the arms the canonical corpus cannot supply come through here: the
    absolute-colour basis, and layer -1, which is the input space rather than a
    block output. Cached under $SPADE_BRACKET_DIR, never a canonical path.
    """
    kind = arm_class.__name__.lower()
    path = CACHE / f"{kind}_{basis}_layer{layer}.pt"
    acts = probe_acts(layer)
    arm = arm_class(acts.shape[-1], SEED)
    if path.exists():
        arm.load_state_dict(torch.load(path, weights_only=True))
        return arm.requires_grad_(False)

    tr, va = fit_split()
    y = probe_labels(basis).reshape(-1, 64)
    print(f"fitting {kind} {basis} layer {layer}: {len(tr)} rows, {len(va)} held to stop on",
          flush=True)
    with torch.enable_grad():
        fit(arm, acts[tr], y[tr], acts[va], y[va], select_device(), BATCH)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(arm.state_dict(), path)
    return arm.cpu().requires_grad_(False)


def tpr(layer: int, basis: str) -> torch.nn.Module:
    """The bilinear read-out: one probe per type, so the mover-relative blocks are
    the canonical artifact and nothing here refits them."""
    if basis == "relative" and layer != INPUT_LAYER:
        return probes.tpr(layer)
    return local(TPR, layer, basis)


def board(layer: int, basis: str) -> torch.nn.Module:
    """The per-square linear read-out, the capacity reference for the TPR."""
    if basis == "relative" and layer != INPUT_LAYER:
        weight = probes.board(layer, flat=True)
        arm = Board(weight.shape[0], SEED)
        arm.weight.data = weight
        return arm.requires_grad_(False)
    return local(Board, layer, basis)


def occupancy_ceiling(labels: Tensor) -> float:
    """Best accuracy reachable without reading colour: call empty/occupied perfectly,
    then always answer the majority colour. A property of the labels, not of a probe,
    so it is what an accuracy in either basis has to be read against."""
    share = [(labels == c).float().mean().item() for c in range(len(CLASSES))]
    return share[EMPTY] + max(share[1:])


def accuracies(layer: int) -> dict:
    """Board-state accuracy over rows N_TRAIN: (games 160-399), all 64 squares.

    The relative arms are the canonical probes, whose corpus is the same Karvonen
    stream this study samples, so most of these 240 games are in their training
    split and their number here is not held out. `*_canonical_test` is, and is the
    one to quote; the study-set column exists so relative and absolute are read off
    one population.
    """
    acts = probe_acts(layer)[N_TRAIN:]
    out = {"layer": layer, "n_train": N_TRAIN, "n_eval": acts.shape[0]}
    for basis in BASES:
        labels = probe_labels(basis)[N_TRAIN:].reshape(-1, 64)
        for name, arm in (("tpr", tpr(layer, basis)), ("linear", board(layer, basis))):
            out[f"acc_{name}_{basis}"] = round(
                (arm(acts).argmax(-1) == labels).float().mean().item(), 4)
        # A digit finer than the accuracies: the absolute basis sits 0.004 points
        # under its ceiling, which four decimals cannot show.
        out[f"ceiling_{basis}"] = round(occupancy_ceiling(labels), 5)
    if layer != INPUT_LAYER:
        out["acc_tpr_relative_canonical_test"] = round(
            probes.card(layer, "tpr")["test_accuracy"], 4)
        out["acc_linear_relative_canonical_test"] = round(
            probes.card(layer)["test_accuracy"], 4)
    return out


# ---------------------------------------------------------------- read-out

@dataclass(frozen=True)
class Decode:
    """One probe's read-out of every event at one layer."""
    layer: int
    logits: dict[str, np.ndarray]  # condition -> [n_events, 64, 3]
    codes: np.ndarray  # [n_events, 64] over ROLE_CODE
    square_s: np.ndarray  # [n_events]

    @property
    def pred(self) -> dict[str, np.ndarray]:
        return {c: l.argmax(-1) for c, l in self.logits.items()}

    def at_s(self, per_square: np.ndarray) -> np.ndarray:
        return per_square[np.arange(len(self.square_s)), self.square_s]


def decode(layer: int, ev: pd.DataFrame, acts: dict[str, Tensor],
           input_acts: dict[str, Tensor], basis: str = "relative") -> Decode:
    probe = tpr(layer, basis)
    source = input_acts if layer == INPUT_LAYER else {c: a[:, layer] for c, a in acts.items()}
    return Decode(layer, {c: probe(h).numpy() for c, h in source.items()},
                  role_codes(ev), ev.square_s.to_numpy())


def owner(pred: np.ndarray, condition: str) -> np.ndarray:
    """Decoded absolute owner: +1 the mover's colour, -1 the opponent's, 0 empty.

    This is the design's "compose the relative label with whose turn it is", and
    it is exact rather than fitted, so absolute correctness at s reduces to
    reading `theirs` at both ends.
    """
    sign = ORIENT[condition]
    return np.where(pred == EMPTY, 0, np.where(pred == MINE, sign, -sign))


def coords(logits: np.ndarray, basis: str, orient: np.ndarray | int = 1) -> np.ndarray:
    """[..., 2] gauge-invariant panel coordinates for the probe that produced `logits`.

    x is who owns the square, y = mean of the two owner logits - l_empty is
    whether it is occupied at all. `orient` re-signs x per event and is the
    mover's colour for the absolute basis, where it is what stops black and white
    from cancelling; the relative basis is already written from a mover's side.
    """
    hi, lo = OWNER[basis]
    x = (logits[..., hi] - logits[..., lo]) * orient
    y = (logits[..., hi] + logits[..., lo]) / 2 - logits[..., EMPTY]
    return np.stack([x, y], -1)


# ---------------------------------------------------------------- statistics

def boot_ci(numer: np.ndarray, denom: np.ndarray,
            rng: np.random.Generator) -> tuple[float, float]:
    """95% percentile CI for numer.sum()/denom.sum(), resampling events with replacement."""
    idx = rng.integers(0, len(numer), size=(N_BOOT, len(numer)))
    lo, hi = np.percentile(numer[idx].sum(1) / denom[idx].sum(1), [2.5, 97.5])
    return float(lo), float(hi)


def ratio_row(name: str, layer: int, numer: np.ndarray, denom: np.ndarray,
              rng: np.random.Generator) -> dict:
    lo, hi = boot_ci(numer, denom, rng)
    return {"layer": layer, "metric": name, "estimate": numer.sum() / denom.sum(),
            "ci_lo": lo, "ci_hi": hi, "n_events": int((denom > 0).sum()),
            "n_units": int(denom.sum())}


def swapped(pred_a: np.ndarray, pred_b: np.ndarray) -> np.ndarray:
    """theirs -> mine or mine -> theirs; strictly stronger than "the class changed"."""
    return (((pred_a == THEIRS) & (pred_b == MINE))
            | ((pred_a == MINE) & (pred_b == THEIRS)))


def readout(d: Decode, rng: np.random.Generator) -> list[dict]:
    """The M1 proportions at one layer, each with a cluster-bootstrap interval."""
    pred = d.pred
    s = {c: d.at_s(p) for c, p in pred.items()}
    ones = np.ones(len(d.square_s))
    per_event_flags = {
        "p_s_unchanged_rel": s["before"] == s["after"],
        "p_s_theirs_both": (s["before"] == THEIRS) & (s["after"] == THEIRS),
        "p_s_unchanged_ctrl": s["before"] == s["control"],
        "p_s_absolute_correct": (owner(s["before"], "before") == -1)
                                & (owner(s["after"], "after") == 1),
        "p_s_absolute_correct_before": owner(s["before"], "before") == -1,
        "p_s_absolute_correct_after": owner(s["after"], "after") == 1,
    }
    rows = [ratio_row(name, d.layer, flag.astype(float), ones, rng)
            for name, flag in per_event_flags.items()]

    swaps = swapped(pred["before"], pred["after"])
    for name, roles in [("p_unflipped_swap_rel", ("unflipped_own", "unflipped_opp")),
                        ("p_unflipped_swap_own", ("unflipped_own",)),
                        ("p_unflipped_swap_opp", ("unflipped_opp",))]:
        mask = np.isin(d.codes, [ROLE_CODE[r] for r in roles])
        rows.append(ratio_row(name, d.layer, (swaps & mask).sum(1).astype(float),
                              mask.sum(1).astype(float), rng))
    return rows


def per_event(d: Decode, ev: pd.DataFrame) -> pd.DataFrame:
    """One row per event at the main layer, so A5 can re-derive every proportion."""
    pred = d.pred
    s = {c: d.at_s(p) for c, p in pred.items()}
    unflipped = np.isin(d.codes, [ROLE_CODE["unflipped_own"], ROLE_CODE["unflipped_opp"]])
    swaps = swapped(pred["before"], pred["after"])
    return pd.DataFrame({
        "game_id": ev.game_id, "t": ev.t, "mover": ev.mover, "square_s": ev.square_s,
        "control_kind": ev.control_kind,
        **{f"s_{c}": [CLASSES[k] for k in s[c]] for c in CONDITIONS},
        "s_unchanged": s["before"] == s["after"],
        "s_unchanged_ctrl": s["before"] == s["control"],
        "s_absolute_correct": (owner(s["before"], "before") == -1)
                              & (owner(s["after"], "after") == 1),
        "n_unflipped": unflipped.sum(1),
        "n_unflipped_swapped": (swaps & unflipped).sum(1),
    })


# ---------------------------------------------------------------- M2 field

def paired(name: str, diff: np.ndarray, denom: np.ndarray,
           rng: np.random.Generator) -> dict:
    """A within-event contrast between two conditions, and its interval, as columns."""
    lo, hi = boot_ci(diff, denom, rng)
    return {name: diff.sum() / denom.sum(), f"{name}_lo": lo, f"{name}_hi": hi}


def field_events(ev: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """Strict-control events, FIELD_PER_T at every move index in FIELD_T_RANGE.

    A fixed quota rather than "whatever the corpus holds": move index is the strongest
    nuisance variable here, and the canonical 2,000-event corpus thins out past t = 30, so
    taking it as-is would weight the panel toward the middle of the game. Meeting the quota
    at every index needs the larger FIELD_VARIANT corpus; the assert names the shortfall
    rather than quietly averaging over an uneven set.
    """
    low, high = FIELD_T_RANGE
    strict = (ev.control_kind == "strict").to_numpy() & ev.t.between(low, high).to_numpy()
    values = range(low, high + 1)
    available = {t: int((strict & (ev.t == t).to_numpy()).sum()) for t in values}
    short = {t: n for t, n in available.items() if n < FIELD_PER_T}
    assert not short, f"corpus is short of {FIELD_PER_T} strict events at t={short}"
    quota = FIELD_PER_T
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(ev), dtype=bool)
    for t in values:
        at = np.flatnonzero(strict & (ev.t == t).to_numpy())
        keep[rng.permutation(at)[:quota]] = True
    assert keep.sum() == quota * len(values), "a move index came up short"
    return keep


def field(rel: Decode, abso: Decode, ev: pd.DataFrame,
          rng: np.random.Generator) -> pd.DataFrame:
    """Mean position per square role and condition, with paired-displacement CIs.

    128,000 arrows would be a smear, so the panel gets five aggregated ones per
    basis (FIGURE_GUIDELINES 4.7). Two choices the panel rests on:

    - **Strict-control events only.** The dashed control arrow is the panel's one
      contrast, and a control move touching s's rays is not a clean one. Before
      and after do not depend on m', but they are restricted too: the gap between
      the control head and the after head has to be read off one population.
    - **Intervals on the displacements, not the endpoints.** An arrow asserts a
      change, and the change is paired within event. Endpoints are not the claim,
      and their intervals do not combine into one for the arrow: on x the two
      conditions are strongly anticorrelated (the same event's discs read
      confidently `mine` before and confidently `theirs` after), so an endpoint
      interval understates the arrow by ~2x there and overstates it on y.
    """
    strict = field_events(ev)
    mover = ev.mover.to_numpy()[strict, None]
    rows = []
    for basis, d, orient in (("relative", rel, 1), ("absolute", abso, mover)):
        xy = {c: coords(l[strict], basis, orient) for c, l in d.logits.items()}
        for role, code in ROLE_CODE.items():
            mask = (d.codes[strict] == code).astype(float)
            n = mask.sum(1)
            row = {"layer": d.layer, "basis": basis, "role": role,
                   "n_events": int((n > 0).sum()), "n_squares": int(n.sum())}
            for axis, name in ((0, "x"), (1, "y")):
                total = {c: (v[..., axis] * mask).sum(1) for c, v in xy.items()}
                row |= {f"{c}_{name}": total[c].sum() / n.sum() for c in CONDITIONS}
                for c in ("after", "control"):
                    row |= paired(f"d_{c}_{name}", total[c] - total["before"], n, rng)
                row |= paired(f"gap_{name}", total["after"] - total["control"], n, rng)
            rows.append(row)
    return pd.DataFrame(rows)


def displacements(d: Decode) -> pd.DataFrame:
    """Per event, the median before->after displacement length of each role's squares.

    Taking the median over squares first keeps the event as the unit of analysis,
    so the marginal and its bootstrap band share one n. Roles holding a single
    square (flipped, placed) are that square's value.
    """
    step = np.linalg.norm(coords(d.logits["after"], "relative")
                          - coords(d.logits["before"], "relative"), axis=-1)
    return pd.DataFrame({role: np.nanmedian(np.where(d.codes == code, step, np.nan), axis=1)
                         for role, code in ROLE_CODE.items()})


# ---------------------------------------------------------------- F2 worked example

def canonical(ev: pd.DataFrame, lengths: pd.DataFrame) -> tuple[int, dict]:
    """A typical event, picked mechanically: among events at the median t holding a
    strict control, the one whose flipped-square displacement is closest to the
    median over all events. Typical in the measurement, not in the picture."""
    target = lengths["flipped"].median()
    pool = ev.index[(ev.t == int(ev.t.median())) & (ev.control_kind == "strict")]
    i = int(pool[np.argmin(np.abs(lengths.loc[pool, "flipped"] - target).to_numpy())])
    return i, {"n_pool": len(pool), "t": int(ev.t[i]), "median_t": int(ev.t.median()),
               "median_displacement": round(float(target), 4),
               "event_displacement": round(float(lengths.loc[i, "flipped"]), 4)}


def event_meta(ev: pd.DataFrame, i: int) -> pd.DataFrame:
    """What the figure needs about that event but the 64-square table cannot carry.

    `move` and `control_move` are vocab tokens; the panel draws squares, so both are
    converted here and nowhere else.
    """
    r = ev.loc[i]
    return pd.DataFrame([{
        "game_id": int(r.game_id), "t": int(r.t), "mover": int(r.mover),
        "square_s": int(r.square_s), "square_m": to_string(int(r.move)),
        "square_m_prime": to_string(int(r.control_move)), "control_kind": r.control_kind}])


def worked_example(d: Decode, ev: pd.DataFrame, i: int) -> pd.DataFrame:
    """The 64 squares of one event: true discs, decoded class, panel coordinates.

    Only m and s change between the two boards, so the role code and the mover
    determine both boards outright and no replay is needed.
    """
    mover = int(ev.mover[i])
    disc = {"before": {"empty": 0, "placed": 0, "flipped": -mover,
                       "unflipped_own": mover, "unflipped_opp": -mover},
            "after": {"empty": 0, "placed": mover, "flipped": mover,
                      "unflipped_own": mover, "unflipped_opp": -mover}}
    role = np.array(list(ROLE_CODE))[np.argmax(
        d.codes[i][:, None] == np.array(list(ROLE_CODE.values()))[None, :], axis=1)]
    row = {"square": np.arange(64), "role": role,
           **{f"disc_{c}": [disc[c][r] for r in role] for c in ("before", "after")}}
    for c in CONDITIONS:
        row[f"pred_{c}"] = [CLASSES[k] for k in d.pred[c][i]]
        xy = coords(d.logits[c][i], "relative")
        row |= {f"{c}_x": xy[:, 0], f"{c}_y": xy[:, 1]}
    return pd.DataFrame(row)


# ---------------------------------------------------------------- driver

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    RESULTS.mkdir(parents=True, exist_ok=True)
    FLANKING.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ev, acts = events(), event_acts()
    input_acts = input_event_acts(ev)

    basis = pd.DataFrame([accuracies(layer) for layer in LAYERS])
    basis.to_csv(FLANKING / "m1_basis.csv", index=False)
    print(basis.to_string(index=False), flush=True)

    rows, main_decode = [], None
    for layer in LAYERS:
        d = decode(layer, ev, acts, input_acts)
        rows += readout(d, rng)
        if layer == MAIN_LAYER:
            main_decode = d
        print(f"layer {layer} read out", flush=True)
    depth = pd.DataFrame(rows)
    depth.to_csv(RESULTS / "m1_decode.csv", index=False)

    per_event(main_decode, ev).to_csv(FLANKING / "m1_per_event.csv", index=False)
    absolute_decode = decode(MAIN_LAYER, ev, acts, input_acts, "absolute")
    # The arrow panel scores its own corpus: the quota it needs at every move index is more
    # events than the canonical set holds past t = 30. Same miner, same stream, same seed, so
    # it extends that set rather than replacing it, and nothing else here reads it.
    ev_field = events(FIELD_VARIANT)
    acts_field = event_acts(FIELD_VARIANT)
    input_field = input_event_acts(ev_field)
    field(decode(MAIN_LAYER, ev_field, acts_field, input_field),
          decode(MAIN_LAYER, ev_field, acts_field, input_field, "absolute"),
          ev_field, rng).to_csv(FLANKING / "m2_field.csv", index=False)
    lengths = displacements(main_decode)
    i, rule = canonical(ev, lengths)
    worked_example(main_decode, ev, i).to_csv(FLANKING / "f2_event.csv", index=False)
    event_meta(ev, i).to_csv(FLANKING / "f2_event_meta.csv", index=False)
    lengths.insert(0, "game_id", ev.game_id)
    lengths.to_csv(FLANKING / "m2_displacements.csv", index=False)

    blocks = depth[depth.layer >= 0]
    peak = blocks.loc[blocks[blocks.metric == "p_s_absolute_correct"].estimate.idxmax(), "layer"]
    headline = depth[depth.layer == MAIN_LAYER].set_index("metric").estimate
    accs = basis.set_index("layer")
    contract = {"n_events": len(ev), "peak_layer": int(peak), "main_layer": MAIN_LAYER,
                "p_s_unchanged_rel": headline["p_s_unchanged_rel"],
                "p_unflipped_swap_rel": headline["p_unflipped_swap_rel"],
                "p_s_absolute_correct": headline["p_s_absolute_correct"],
                "acc_rel_L5": accs.acc_tpr_relative[MAIN_LAYER],
                "acc_abs_L5": accs.acc_tpr_absolute[MAIN_LAYER],
                "canonical_event_row": i, **{f"canonical_{k}": v for k, v in rule.items()}}
    pd.Series(contract).rename("value").rename_axis("key").to_csv(FLANKING / "contract_a3.csv")
    print(pd.Series(contract).to_string(), flush=True)


if __name__ == "__main__":
    main()
