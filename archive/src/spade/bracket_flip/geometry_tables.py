"""M3 and M4 of the bracket-flip study: the layer-5 activation-geometry tables.

Writes CSVs to analysis/bracket_flip/results/; make_figures.py draws them and never
recomputes anything.

Two per-event nulls, and they are not interchangeable. `cos_null1_*` scores one
unflipped disc drawn at random, so it is the same kind of quantity as `cos_flip_*` --
one direction, one cosine -- and is what a specificity claim has to be read against.
`cos_null_*` is the median over that event's unflipped discs (8 to 60 of them, 29
typically); it locates where the null centres, but the spread of a median is not the
spread of a draw, so it is not a yardstick for one cosine.

Usage: python -m spade.bracket_flip.geometry_tables
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spade.bracket_flip import act_geometry as ag
from spade.common.config import D_MODEL
from spade.common.paths import repo_root

SEED = 42  # one seed for the whole table, so `cos_null1_*` and `var_flip_null` draw the same discs
N_BOOT = 1000
CONTRASTS = {"after-before": ("after", "before"), "after-control": ("after", "control")}
RESULTS = repo_root() / "analysis" / "bracket_flip" / "results"


@dataclass(frozen=True)
class Measurements:
    """Keyed by (contrast, direction source); everything downstream reads from here."""

    dirs: dict[str, np.ndarray]
    turn: np.ndarray
    delta_pc1: np.ndarray
    strict: np.ndarray
    cos: dict[tuple[str, str], np.ndarray]        # [n, 64], every square
    cos_null1: dict[tuple[str, str], np.ndarray]  # [n], the one drawn unflipped disc
    proj: dict[tuple[str, str], ag.Projections]

    def index(self, contrast: str, n: int) -> np.ndarray:
        """The strict-control contrast is only defined where the control is strict."""
        return self.strict if contrast == "after-control" else np.arange(n)


def measure(events: ag.Events) -> Measurements:
    dirs = {source: ag.retain_directions(source) for source in ag.SOURCES}
    # Same seed per source, so both score the same drawn disc and differ only in direction.
    null = {source: ag.null_direction(events, d, np.random.default_rng(SEED))
            for source, d in dirs.items()}
    turn = ag.turn_axis(events)
    cos, cos_null1, proj = {}, {}, {}
    for contrast, (head, tail) in CONTRASTS.items():
        delta = events.delta(head, tail)
        unit = delta / np.linalg.norm(delta, axis=1, keepdims=True)
        for source, d in dirs.items():
            cos[contrast, source] = ag.cos_retain(delta, d, events.codes)
            cos_null1[contrast, source] = (unit * null[source]).sum(1)
            proj[contrast, source] = ag.project(
                delta, events.delta("control", "before"), ag.flip_direction(events, d),
                null[source], turn)
    pc1 = np.linalg.svd(events.delta("after", "before"), full_matrices=False)[2][0]
    return Measurements(dirs, turn, pc1, np.flatnonzero(events.strict), cos, cos_null1, proj)


def per_event_table(events: ag.Events, m: Measurements) -> pd.DataFrame:
    rows = events.table[["game_id", "t", "mover", "square_s", "control_kind"]].copy()
    rows["n_unflipped"] = np.isin(events.codes, ag.NULL_CODES).sum(1)
    for name, (head, tail) in [("after_before", ("after", "before")),
                               ("control_before", ("control", "before")),
                               ("after_control", ("after", "control"))]:
        rows[f"norm_{name}"] = np.linalg.norm(events.delta(head, tail), axis=1)
    swap = events.delta("control", "before")
    delta = events.delta("after", "before")
    rows["cos_delta_swap"] = ((delta * swap).sum(1)
                              / (np.linalg.norm(delta, axis=1) * np.linalg.norm(swap, axis=1)))
    unflipped = np.isin(events.codes, ag.NULL_CODES)
    for (contrast, source), cos in m.cos.items():
        tag = f"{contrast.replace('-', '_')}_{source}"
        rows[f"cos_flip_{tag}"] = cos[np.arange(events.n), events.square_s]
        rows[f"cos_null_{tag}"] = np.nanmedian(np.where(unflipped, cos, np.nan), axis=1)
        rows[f"cos_null1_{tag}"] = m.cos_null1[contrast, source]
    return rows


def shares_table(events: ag.Events, m: Measurements, rng: np.random.Generator) -> pd.DataFrame:
    frames = []
    for (contrast, source), proj in m.proj.items():
        idx = m.index(contrast, events.n)
        frame = ag.bootstrap_shares(proj, idx, rng, N_BOOT)
        frame.insert(0, "contrast", contrast)
        frame.insert(1, "direction_source", source)
        frame["n_events"] = len(idx)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def structure_rows(events: ag.Events, m: Measurements) -> list[dict]:
    """Descriptive numbers the write-up leans on, so none of them is asserted from prose."""
    tpr, linear = m.dirs["tpr"], m.dirs["linear"]
    gram = np.abs(tpr @ tpr.T)[~np.eye(64, dtype=bool)]
    delta = events.delta("after", "before")
    swap = events.delta("control", "before")
    swap /= np.linalg.norm(swap, axis=1, keepdims=True)
    across = (swap @ swap.T)[~np.eye(events.n, dtype=bool)]
    curve = ag.global_spectrum(delta)
    control_share = ((delta * swap).sum(1) ** 2).sum() / (delta ** 2).sum()
    return [
        {"key": "cos_tpr_vs_linear_direction", "value": float(np.median((tpr * linear).sum(1))),
         "n": 64, "definition": "median over squares of cos between the two probes' "
                                "retain directions"},
        {"key": "direction_overlap_median", "value": float(np.median(gram)), "n": 64 * 63,
         "definition": "off-diagonal |cos| between retain directions of different squares, "
                       "TPR; this is the floor the empirical null carries"},
        {"key": "direction_overlap_max", "value": float(gram.max()), "n": 64 * 63,
         "definition": "worst-case of the same"},
        {"key": "cos_turn_axis_vs_delta_pc1", "n": events.n,
         "value": float(abs(m.delta_pc1 @ m.turn)),
         "definition": "the leading direction of the after-before ensemble against the "
                       "to-move axis"},
        {"key": "cos_swap_across_events_median", "value": float(np.median(across)),
         "n": events.n * (events.n - 1),
         "definition": "cos between the matched-control displacements of two different events"},
        {"key": "rank_matching_control_swap", "n": events.n,
         "value": int(np.searchsorted(curve, control_share) + 1),
         "definition": "rank of the shared subspace needed to reach what one per-event "
                       "control direction reaches"},
        {"key": "norm_mean_delta", "value": float(np.linalg.norm(delta.mean(0))), "n": events.n,
         "definition": "length of the mean after-before displacement (activation units)"},
        {"key": "rms_norm_delta", "value": float(np.sqrt((delta ** 2).sum(1).mean())),
         "n": events.n,
         "definition": "root-mean-square length of the same displacements"},
    ]


DEFINITIONS = {
    "cos_flip_median": "median cos of the displacement with the retain direction at s",
    "cos_null_median": "the same at unflipped discs, pooled (empirical null)",
    "p_flip_positive": "events whose cos at s is positive",
    "var_swap": "energy share along the matched control's displacement",
    "var_flip": "energy share along the retain direction at s, swap removed",
    "var_resid": "energy left over: 1 - var_swap - var_flip",
    "var_flip_null": "var_flip taken at one unflipped disc per event (empirical null)",
    "var_turn": "energy share along the single to-move axis the whole event set shares",
}


def summary_table(events: ag.Events, m: Measurements) -> pd.DataFrame:
    """The cross-validation contract keys, each with the definition it was computed under."""
    unflipped = np.isin(events.codes, ag.NULL_CODES)
    rows = [{"key": "n_events", "value": events.n, "n": events.n,
             "definition": "headline single-flip events, layer 5"},
            {"key": "n_events_strict", "value": int(events.strict.sum()), "n": events.n,
             "definition": "events whose control avoids all eight of s's rays"},
            {"key": "var_isotropic_chance", "value": 1.0 / (D_MODEL - 1), "n": events.n,
             "definition": "share a random direction takes once the swap axis is removed"},
            *structure_rows(events, m)]
    for (contrast, source), cos in m.cos.items():
        idx = m.index(contrast, events.n)
        flip = cos[idx, events.square_s[idx]]
        null = cos[idx][unflipped[idx]]
        common = {"contrast": contrast, "direction_source": source}
        for key, value, n in [("cos_flip_median", float(np.median(flip)), len(idx)),
                              ("cos_null_median", float(np.median(null)), len(null)),
                              ("p_flip_positive", float((flip > 0).mean()), len(idx))]:
            rows.append({"key": key, **common, "value": value, "n": n,
                         "definition": DEFINITIONS[key]})
    for (contrast, source), proj in m.proj.items():
        idx = m.index(contrast, events.n)
        for component, value in proj.shares(idx).items():
            rows.append({"key": f"var_{component}", "contrast": contrast,
                         "direction_source": source, "value": value, "n": len(idx),
                         "definition": DEFINITIONS[f"var_{component}"]})
    return pd.DataFrame(rows)[["key", "contrast", "direction_source", "value", "n",
                               "definition"]].fillna({"contrast": "", "direction_source": ""})


def main() -> None:
    events = ag.load()
    m = measure(events)

    RESULTS.mkdir(parents=True, exist_ok=True)
    per_event_table(events, m).to_csv(RESULTS / "flip_cos_per_event.csv", index=False)
    shares_table(events, m, np.random.default_rng(SEED)).to_csv(
        RESULTS / "variance_shares.csv", index=False)
    summary_table(events, m).to_csv(RESULTS / "summary.csv", index=False)

    checks = ag.sanity(events)
    checks.to_csv(RESULTS / "sanity.csv", index=False)
    print(checks.to_string(index=False))
    assert checks["pass"].all(), "extraction sanity check failed"
    print(f"wrote flip_cos_per_event, variance_shares, summary and sanity to {RESULTS}")


if __name__ == "__main__":
    main()
