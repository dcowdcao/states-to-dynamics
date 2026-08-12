"""Layer-5 activation geometry of a bracketing flip (M3, M4 of docs/bracket_flip_design.md).

Everything here reads raw `resid_post`. The predicted flip direction comes from
either probe family, and both are fit on the study's own Karvonen probe set --
the corpus the events themselves are mined from, labelled by engine replay. The
older ARENA fits stay reachable through `corpus="arena"` as the off-corpus control.

The flip is the *absence* of a perspective swap, so the direction a square is
scored against is its **retain** direction: the residual step that keeps its
mover-relative class where the swap would have exchanged it. For the flanked
square s that step is what actually happens; for every unflipped disc it is the
opposite of what happens, which is the empirical null.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.signal import fftconvolve

from spade.common.config import D_MODEL, LAYER
from spade.common.paths import linear_probe_activations
from spade.common.study import study_dir

CLASSES = ("empty", "theirs", "mine")
# `control_squares` codes, read on the board before m
# (analysis/bracket_flip/reports/DATA_REPORT.md).
CODE = {"empty": 0, "placed": 1, "flipped": 2, "unflipped_own": 3, "unflipped_opp": 4}
NULL_CODES = (CODE["unflipped_own"], CODE["unflipped_opp"])
SOURCES = ("tpr", "linear")
CORPORA = ("karvonen", "arena")
PRIMARY_CORPUS = "karvonen"

ARENA_LINEAR_PROBE = linear_probe_activations() / "linear_probe.pt"


@dataclass(frozen=True)
class Events:
    """One row per flip event, activations already sliced to a single layer."""

    table: pd.DataFrame
    codes: np.ndarray   # [n, 64] role code per engine square
    before: np.ndarray  # [n, d_model]
    after: np.ndarray
    control: np.ndarray

    @property
    def n(self) -> int:
        return len(self.table)

    @property
    def square_s(self) -> np.ndarray:
        return self.table["square_s"].to_numpy()

    @property
    def strict(self) -> np.ndarray:
        """Controls whose placement and flips avoid all eight of s's rays."""
        return (self.table["control_kind"] == "strict").to_numpy()

    def delta(self, head: str, tail: str) -> np.ndarray:
        return getattr(self, head) - getattr(self, tail)


def load(layer: int = LAYER) -> Events:
    d = study_dir("bracket") / "events"
    table = pd.read_csv(d / "events.csv")
    acts = {c: torch.load(d / f"{c}.pt", map_location="cpu", weights_only=True)[:, layer]
            .double().numpy() for c in ("before", "after", "control")}
    codes = np.array([[int(ch) for ch in row] for row in table["control_squares"].astype(str)])
    assert (codes[np.arange(len(table)), table["square_s"]] == CODE["flipped"]).all()
    return Events(table=table, codes=codes, **acts)


def retain_directions(source: str, layer: int = LAYER,
                      corpus: str = PRIMARY_CORPUS) -> np.ndarray:
    """[64, d_model] unit rows: keep square q 'theirs' where the swap would make it 'mine'.

    A square that is 'mine' before the move takes the negative of its row — both
    probes are linear in the filler contrast, so one matrix covers both signs.

    `corpus` picks the fit both sources come from, so probe class is never confounded
    with training corpus: "karvonen" is the study's own 400-game engine-labelled probe
    set, "arena" the committed 200-game `t % 2`-labelled one, whose linear probe exists
    only at the primary layer.
    """
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}, expected one of {CORPORA}")
    # Deferred: both probe modules put the vendored TPR fork on sys.path at import.
    from spade.bracket_flip import probe_space, tpr_probe

    if source == "linear":
        if corpus == "arena":
            assert layer == LAYER, "the arena set commits one linear probe, at the primary layer"
            weights = torch.load(ARENA_LINEAR_PROBE, map_location="cpu", weights_only=True)
        else:
            weights = probe_space.linear(layer, "relative")
        w = weights.reshape(D_MODEL, 64, len(CLASSES)).double().numpy()
        rows = (w[:, :, CLASSES.index("theirs")] - w[:, :, CLASSES.index("mine")]).T
    elif source == "tpr":
        probe = tpr_probe.load(layer) if corpus == "arena" else probe_space.tpr(layer, "relative")
        rows = np.stack([probe.edit_direction(q, CLASSES.index("mine"),
                                              CLASSES.index("theirs")).numpy()
                         for q in range(64)]).astype(np.float64)
    else:
        raise ValueError(f"unknown direction source {source!r}, expected one of {SOURCES}")
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def cos_retain(delta: np.ndarray, dirs: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """[n, 64] cosine of each event's displacement with each square's retain direction."""
    unit = delta / np.linalg.norm(delta, axis=1, keepdims=True)
    return (unit @ dirs.T) * np.where(codes == CODE["unflipped_own"], -1.0, 1.0)


def flip_direction(events: Events, dirs: np.ndarray) -> np.ndarray:
    """[n, d_model] unit retain direction of each event's flanked square."""
    return dirs[events.square_s]


def null_direction(events: Events, dirs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One unflipped disc per event, so the null carries one direction per event too."""
    picks = np.array([rng.choice(np.flatnonzero(np.isin(row, NULL_CODES))) for row in events.codes])
    sign = np.where(events.codes[np.arange(events.n), picks] == CODE["unflipped_own"], -1.0, 1.0)
    return dirs[picks] * sign[:, None]


@dataclass(frozen=True)
class Projections:
    """Per-event squared lengths of one displacement along named directions.

    swap + flip + resid = total exactly, because the flip direction is
    orthogonalized against the swap direction before projecting. `flip_null` and
    `turn` are measured against the same total but are not part of that split:
    the first is the flip projection taken at an unflipped disc, the second is
    the one direction the whole event set shares.
    """

    total: np.ndarray
    swap: np.ndarray
    flip: np.ndarray
    flip_null: np.ndarray
    turn: np.ndarray

    @property
    def resid(self) -> np.ndarray:
        return self.total - self.swap - self.flip

    def shares(self, idx: np.ndarray) -> dict[str, float]:
        total = self.total[idx].sum()
        return {name: getattr(self, name)[idx].sum() / total
                for name in ("swap", "flip", "resid", "flip_null", "turn")}


def _orthogonal_projection(delta: np.ndarray, direction: np.ndarray,
                           against: np.ndarray) -> np.ndarray:
    d = direction - (direction * against).sum(1, keepdims=True) * against
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return (delta * d).sum(1) ** 2


def project(delta: np.ndarray, swap: np.ndarray, flip: np.ndarray,
            null: np.ndarray, turn: np.ndarray) -> Projections:
    """Split `delta` along the swap direction, then the flip direction orthogonal to it."""
    swap_unit = swap / np.linalg.norm(swap, axis=1, keepdims=True)
    return Projections(
        total=(delta ** 2).sum(1),
        swap=(delta * swap_unit).sum(1) ** 2,
        flip=_orthogonal_projection(delta, flip, swap_unit),
        flip_null=_orthogonal_projection(delta, null, swap_unit),
        turn=(delta @ turn) ** 2)


def bootstrap_shares(proj: Projections, idx: np.ndarray, rng: np.random.Generator,
                     n_boot: int = 1000) -> pd.DataFrame:
    """Percentile bootstrap over events; the resampling unit is one event."""
    draws = [proj.shares(rng.choice(idx, len(idx), replace=True)) for _ in range(n_boot)]
    full, spread = proj.shares(idx), pd.DataFrame(draws)
    return pd.DataFrame({
        "component": list(full),
        "share": [full[k] for k in full],
        "ci_lo": [spread[k].quantile(0.025) for k in full],
        "ci_hi": [spread[k].quantile(0.975) for k in full],
    })


def event_local_frame(events: Events, dirs: np.ndarray, idx: np.ndarray) -> pd.DataFrame:
    """Per-event 2-D coordinates with `before` at the origin.

    y is the predicted flip direction for s; x is what is left of the matched
    control's displacement once y is removed. Neither axis is defined by the
    `after` condition, so the vertical gap between the control and after heads is
    a measurement rather than a construction.
    """
    y_hat = flip_direction(events, dirs)[idx]
    swap = events.delta("control", "before")[idx]
    x_hat = swap - (swap * y_hat).sum(1, keepdims=True) * y_hat
    x_hat /= np.linalg.norm(x_hat, axis=1, keepdims=True)
    out = {}
    for condition in ("control", "after"):
        d = events.delta(condition, "before")[idx]
        out[f"x_{condition}"] = (d * x_hat).sum(1)
        out[f"y_{condition}"] = (d * y_hat).sum(1)
    return pd.DataFrame(out, index=idx)


def global_spectrum(delta: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Cumulative variance share of the best rank-k subspace shared by all events.

    `weights` are bootstrap resampling counts over events; going through the
    512x512 Gram keeps a whole bootstrap affordable.
    """
    gram = delta.T @ (delta if weights is None else delta * weights[:, None])
    spectrum = np.linalg.eigvalsh(gram)[::-1].clip(min=0)
    return np.cumsum(spectrum) / spectrum.sum()


def pooled_centered(events: Events) -> np.ndarray:
    """[3n, d_model] before, after and control stacked, each centered on its move number.

    Centering per t removes the trajectory drift that otherwise owns the leading
    components, so what is left is the condition structure.
    """
    pooled = np.concatenate([events.before, events.after, events.control])
    t = np.tile(events.table["t"].to_numpy(), 3)
    for value in np.unique(t):
        pooled[t == value] -= pooled[t == value].mean(0)
    return pooled


def to_move(events: Events) -> np.ndarray:
    """[3n] engine colour to move at the position each pooled row was read at.

    `before` is position t-1, so its mover is the one that plays m; `after` and
    `control` are position t, where the turn has already passed to the opponent.
    """
    mover = np.tile(events.table["mover"].to_numpy(), 3)
    return np.where(np.repeat(np.arange(3), events.n) == 0, mover, -mover)


def turn_axis(events: Events) -> np.ndarray:
    """Unit direction separating the two turn classes: the one axis every event shares."""
    pooled, side = pooled_centered(events), to_move(events)
    axis = pooled[side == 1].mean(0) - pooled[side == -1].mean(0)
    return axis / np.linalg.norm(axis)


def centered_pca(events: Events, n_components: int = 2
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, per-component variance ratios and the components themselves."""
    pooled = pooled_centered(events)
    _, s, vt = np.linalg.svd(pooled, full_matrices=False)
    return pooled @ vt[:n_components].T, s[:n_components] ** 2 / (s ** 2).sum(), vt[:n_components]


@dataclass(frozen=True)
class Density:
    """A binned Gaussian KDE with a percentile bootstrap band over events."""

    grid: np.ndarray
    density: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    bandwidth: float
    n_values: int


def kde_band(values: np.ndarray, event_id: np.ndarray, n_events: int, grid: np.ndarray,
             rng: np.random.Generator, n_boot: int = 1000) -> Density:
    """Scott's-rule Gaussian KDE, bandwidth fixed at the full-sample value.

    Holding h fixed across resamples keeps the band a statement about sampling
    noise rather than about the bandwidth tracking each resample's spread
    (FIGURE_GUIDELINES.md 4.4). Binning onto `grid` first turns each resample into
    one weighted sum of per-event histograms.
    """
    h = values.std(ddof=1) * len(values) ** -0.2
    step = grid[1] - grid[0]
    offsets = np.arange(-int(np.ceil(4 * h / step)), int(np.ceil(4 * h / step)) + 1) * step
    kernel = np.exp(-0.5 * (offsets / h) ** 2) / (h * np.sqrt(2 * np.pi))

    bins = np.clip(np.searchsorted(grid + step / 2, values), 0, len(grid) - 1)
    per_event = np.zeros((n_events, len(grid)))
    np.add.at(per_event, (event_id, bins), 1.0)

    weights = np.stack([np.bincount(rng.integers(0, n_events, n_events), minlength=n_events)
                        for _ in range(n_boot)]).astype(float)
    counts = np.vstack([per_event.sum(0), weights @ per_event])
    curves = fftconvolve(counts / counts.sum(1, keepdims=True), kernel[None, :],
                         mode="same", axes=1)
    lo, hi = np.percentile(curves[1:], [2.5, 97.5], axis=0)
    return Density(grid, curves[0], lo, hi, h, len(values))


def sanity(events: Events, layer: int = LAYER) -> pd.DataFrame:
    """Two checks that would fail loudly if the extraction were misaligned.

    `before` depends on the prefix alone, so events that share a prefix must share
    it exactly; and cancelling the prefix, move number and parity with the matched
    control must shrink the displacement.
    """
    d = study_dir("bracket") / "events"
    full_before = torch.load(d / "before.pt", map_location="cpu", weights_only=True)
    groups = events.table.groupby("prefix").indices
    pairs = [i for i in groups.values() if len(i) > 1]
    worst = max((full_before[i] - full_before[i[0]]).abs().max().item() for i in pairs) \
        if pairs else float("nan")

    strict = events.strict
    ba = np.linalg.norm(events.delta("after", "before")[strict], axis=1)
    ac = np.linalg.norm(events.delta("after", "control")[strict], axis=1)
    return pd.DataFrame([
        {"check": "before identical across events sharing a prefix",
         "n": sum(len(i) for i in pairs), "value": worst,
         "threshold": "max abs diff over all 8 layers < 1e-4", "pass": worst < 1e-4},
        {"check": "strict-control contrast shorter than the raw before/after difference",
         "n": int(strict.sum()), "value": float((ac < ba).mean()),
         "threshold": "fraction of events == 1.0", "pass": bool((ac < ba).all())},
        {"check": "median ||after - before|| (layer %d)" % layer,
         "n": int(strict.sum()), "value": float(np.median(ba)), "threshold": "", "pass": True},
        {"check": "median ||after - control|| (layer %d, strict)" % layer,
         "n": int(strict.sum()), "value": float(np.median(ac)), "threshold": "", "pass": True},
    ])
