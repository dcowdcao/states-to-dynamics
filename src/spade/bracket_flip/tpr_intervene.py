"""M6: the paper's TPR intervention, cross-checking the linear-probe un-flip edit.

For each mined single-flip event, edit layer-5 `resid_post` at the injected move's
position so the flanked square s reads its pre-flip colour, then measure how
next-move probability mass moves between the real board's legal set and the
counterfactual (s un-flipped) board's legal set, on diagnostic squares only —
legal on exactly one of the two boards. Scoring is imported from
`spade.rules.causal.unflip_edit`, so the two methods report the same quantity.

Two edit families over the same events, both read off probes fit on the study's own
Karvonen probe set — the corpus these events are mined from — so the head-to-head is
probe class and not training corpus (`--corpus arena` is the off-corpus control):
  tpr     dB = r_s (f_mine - f_theirs)^T mapped back through pinv(M)
          (`TPRProbe.edit_direction`), swept over a multiplicative scale
  linear  that fit's mine-vs-theirs direction, moved to the target class's mean
          projection times `beta` — unflip_edit's protocol

The two knobs are not the same quantity, so a third family, `tpr_matched`, points
along the TPR direction at exactly the linear edit's per-event length: that is the
head-to-head at matched magnitude, with the scale sweep as the wider context.

Populations, all edited theirs -> mine at the same position:
  flipped    s, the disc m just flipped
  settled    a mover-coloured disc that was NOT flipped, untouched >= 10 plies
             (unflip_edit's `settled`, same definition, so the two are comparable)
  random     random direction at s, magnitude matched to the flipped edit

Each pass also records what the edit did to the read-out itself: whether the
target square's decoded class reached `mine`, and how many of the other 63 squares
changed class. Collateral is structural for the TPR edit — 64 roles live in 52
dimensions (analysis/bracket_flip/reports/PROBE_REPORT.md §5) — so an effect has to
be read next to it.

Usage: python -m spade.bracket_flip.tpr_intervene [--scales 1,2,3,4,6,8,12,16] [--betas 1,2]
       [--corpus karvonen|arena]
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from random import Random

import numpy as np
import pandas as pd
import torch

from spade.bracket_flip import probe_space, tpr_probe
from spade.bracket_flip.act_geometry import CORPORA, PRIMARY_CORPUS
from spade.common.config import D_MODEL, HOOK_POINT, LAYER, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.paths import linear_probe_activations, repo_root
from spade.common.study import study_dir
from spade.othello.boards import flip_set, regular_moves, replay, tokens_of
from spade.rules.causal.unflip_edit import set_logmass  # identical scoring, by import

force_ipv4()

from circuits.othello_engine_utils import to_string

EMPTY, THEIRS, MINE = 0, 1, 2
SETTLED_MIN_AGE = 10  # unflip_edit's `settled`; the engine zeroes age on place and on flip
BATCH_SIZE = 128
N_BOOT = 1000
SEED = 7


@dataclass(frozen=True)
class Row:
    """One edit: an event, a target square, and the two boards it discriminates."""

    event: int
    game_id: int
    t: int
    population: str
    square: int
    seq: list[int]
    real_only: list[int]
    cf_only: list[int]
    legal: list[int]


@dataclass(frozen=True)
class Condition:
    method: str  # base | tpr | linear
    scale: float


@dataclass(frozen=True)
class Directions:
    tpr: torch.Tensor      # [64, d_model] least-norm residual step realizing dB
    linear: torch.Tensor   # [64, d_model] unit mine-minus-theirs probe direction
    mu_mine: torch.Tensor  # [64] mean projection of true-`mine` activations onto it


@dataclass(frozen=True)
class Readout:
    """The two decoders, used as a common instrument for target flip and collateral."""

    tpr: tpr_probe.TPRProbe
    linear: torch.Tensor  # [64, 3, d_model]

    def classes(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"tpr": self.tpr.decode(h).argmax(-1),
                "lin": torch.einsum("bd,scd->bsc", h, self.linear).argmax(-1)}


@dataclass(frozen=True)
class Probes:
    """Both decoders at LAYER, plus the probe-set rows their `mine` mean is taken over."""

    tpr: tpr_probe.TPRProbe
    linear: torch.Tensor    # [64, 3, d_model]
    acts: torch.Tensor      # [n, d_model]
    labels: torch.Tensor    # [n, 64]
    provenance: dict[str, str]


def probes(corpus: str) -> Probes:
    """Both probe families fit on the same corpus, so probe class is never confounded with it.

    "karvonen" is the study's own 400-game probe set, labelled by engine replay and the
    corpus the events are mined from. "arena" is the committed 200-game `t % 2`-labelled
    set, kept as the off-corpus control.
    """
    if corpus == "karvonen":
        acts = probe_space.probe_acts(LAYER).float()
        return Probes(
            tpr=probe_space.tpr(LAYER, "relative"),
            linear=probe_space.linear(LAYER, "relative").reshape(D_MODEL, 64, 3).permute(1, 2, 0),
            acts=acts,
            labels=probe_space.probe_labels("relative").reshape(len(acts), 64),
            provenance={"tpr_probe": str(probe_space.path("tpr", LAYER, "relative")),
                        "linear_probe": str(probe_space.path("linear", LAYER, "relative"))})
    if corpus != "arena":
        raise ValueError(f"unknown corpus {corpus!r}, expected one of {CORPORA}")
    d = linear_probe_activations()
    acts = torch.load(d / "primary_layer_activations.pt", weights_only=True).float()
    return Probes(
        tpr=tpr_probe.load(LAYER),
        linear=torch.load(d / "linear_probe.pt", map_location="cpu", weights_only=False)
        .float().reshape(D_MODEL, 64, 3).permute(1, 2, 0),
        acts=acts,
        labels=torch.load(d / "board_states.pt", weights_only=True).reshape(len(acts), 64),
        provenance={"tpr_probe": str(tpr_probe.path(LAYER)),
                    "linear_probe": str(d / "linear_probe.pt")})


def mine_direction(fit: Probes) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit mine-vs-theirs direction per square, and the mean projection of true-`mine`
    rows onto it: `unflip_edit.probe_directions` family 0, restricted to M6's transition."""
    dhat = fit.linear[:, MINE] - fit.linear[:, THEIRS]
    dhat = dhat / dhat.norm(dim=-1, keepdim=True)
    proj = torch.einsum("nd,sd->ns", fit.acts, dhat)
    return dhat, torch.stack([proj[fit.labels[:, s] == MINE, s].mean() for s in range(64)])


def build_rows(events: pd.DataFrame, rng: Random) -> tuple[list[Row], dict]:
    """One row per (event, population), with the diagnostic token sets from the engine."""
    rows, skips = [], Counter()
    for event, e in events.iterrows():
        prefix = [int(tok) for tok in e["prefix"].split()]
        mover, s = int(e["mover"]), int(e["square_s"])
        board = replay(prefix)
        assert len(prefix) == int(e["t"]) and board.next_hand_color == mover
        move_square = to_string(int(e["move"]))
        assert flip_set(board.state, move_square, mover) == [s]
        board.umpire(move_square)
        real_legal = regular_moves(board.state, -mover)
        assert real_legal, "mining guarantees the opponent has a regular move after m"

        settled = [q for q in range(64)
            if board.state[q // 8, q % 8] == mover
            and board.age[q // 8, q % 8] >= SETTLED_MIN_AGE]
        targets = [("flipped", s), ("random", s)]
        if settled:
            targets.append(("settled", rng.choice(settled)))
        else:
            skips["no_settled_square"] += 1

        for population, q in targets:
            state_cf = board.state.copy()
            state_cf[q // 8, q % 8] = -mover
            cf_legal = regular_moves(state_cf, -mover)
            if not cf_legal or cf_legal == real_legal:
                skips[f"no_diagnostic_square_{population}"] += 1
                continue
            rows.append(Row(
                event=int(event), game_id=int(e["game_id"]), t=int(e["t"]),
                population=population, square=q, seq=prefix + [int(e["move"])],
                real_only=tokens_of(real_legal - cf_legal),
                cf_only=tokens_of(cf_legal - real_legal),
                legal=tokens_of(real_legal)))
    return rows, dict(skips)


def edit_vectors(h: torch.Tensor, squares: torch.Tensor, is_random: torch.Tensor,
                 cond: Condition, dirs: Directions, gen: torch.Generator) -> torch.Tensor:
    match cond.method:
        case "base":
            return torch.zeros_like(h)
        case "tpr":
            delta = cond.scale * dirs.tpr[squares]
        case "linear" | "tpr_matched":
            d = dirs.linear[squares]
            delta = (cond.scale * (dirs.mu_mine[squares] - (h * d).sum(-1))).unsqueeze(1) * d
            if cond.method == "tpr_matched":
                tpr = dirs.tpr[squares]
                delta = tpr / tpr.norm(dim=-1, keepdim=True) * delta.norm(dim=-1, keepdim=True)
    rand = torch.randn(len(squares), D_MODEL, generator=gen).to(h.device)
    rand = rand / rand.norm(dim=-1, keepdim=True) * delta.norm(dim=-1, keepdim=True)
    return torch.where(is_random.unsqueeze(1), rand, delta)


def run_pass(model, rows: list[Row], cond: Condition, dirs: Directions,
             readout: Readout, device: str) -> pd.DataFrame:
    by_len: dict[int, list[int]] = {}
    for j, row in enumerate(rows):
        by_len.setdefault(len(row.seq), []).append(j)
    gen = torch.Generator().manual_seed(13)
    records = []
    for idxs in by_len.values():
        for start in range(0, len(idxs), BATCH_SIZE):
            picks = idxs[start:start + BATCH_SIZE]
            batch = [rows[j] for j in picks]
            tokens = torch.tensor([r.seq for r in batch], device=device)
            squares = torch.tensor([r.square for r in batch], device=device)
            is_random = torch.tensor([r.population == "random" for r in batch], device=device)
            stash: dict[str, torch.Tensor] = {}

            def edit(resid, hook, squares=squares, is_random=is_random, stash=stash):
                h = resid[:, -1, :].clone()
                delta = edit_vectors(h, squares, is_random, cond, dirs, gen)
                resid[:, -1, :] = h + delta
                stash["h"], stash["delta"] = h, delta
                return resid

            logits = model.run_with_hooks(tokens, fwd_hooks=[(HOOK_POINT, edit)])
            logprobs = logits[:, -1, :].log_softmax(-1).cpu()
            h, delta = stash["h"], stash["delta"]
            pre, post = readout.classes(h), readout.classes(h + delta)
            item = torch.arange(len(batch), device=device)
            decoded = {}
            for name in pre:
                changed = (pre[name] != post[name]).long()
                decoded[f"pre_{name}"] = pre[name][item, squares].cpu().numpy()
                decoded[f"hit_{name}"] = (post[name][item, squares] == MINE).cpu().numpy()
                decoded[f"collateral_{name}"] = (
                    changed.sum(1) - changed[item, squares]).cpu().numpy()
            edit_norm = delta.norm(dim=-1).cpu().numpy()
            resid_norm = h.norm(dim=-1).cpu().numpy()

            for k, r in enumerate(batch):
                lp = logprobs[k:k + 1]
                records.append({
                    "row": picks[k],
                    "lp_real_only": float(set_logmass(lp, [r.real_only])),
                    "lp_cf_only": float(set_logmass(lp, [r.cf_only])),
                    "lp_legal": float(set_logmass(lp, [r.legal])),
                    "edit_norm": float(edit_norm[k]), "resid_norm": float(resid_norm[k]),
                    **{key: value[k].item() for key, value in decoded.items()},
                })
    return pd.DataFrame(records).set_index("row").sort_index()


def per_row_table(rows: list[Row], base: pd.DataFrame,
                  passes: dict[Condition, pd.DataFrame]) -> pd.DataFrame:
    """Deltas against the unedited run, one block per condition."""
    index = pd.DataFrame({
        "event": [r.event for r in rows], "game_id": [r.game_id for r in rows],
        "t": [r.t for r in rows], "population": [r.population for r in rows],
        "square": [r.square for r in rows],
        "n_real_only": [len(r.real_only) for r in rows],
        "n_cf_only": [len(r.cf_only) for r in rows],
    })
    frames = []
    for cond, table in passes.items():
        frame = index.copy()
        frame.insert(0, "method", cond.method)
        frame.insert(1, "scale", cond.scale)
        for name, column in [("cf", "lp_cf_only"), ("real", "lp_real_only"),
                             ("legal", "lp_legal")]:
            frame[f"d_lp_{name}"] = table[column].to_numpy() - base[column].to_numpy()
        frame["base_lp_cf"] = base["lp_cf_only"].to_numpy()
        frame["base_lp_real"] = base["lp_real_only"].to_numpy()
        for column in ("edit_norm", "resid_norm", "pre_tpr", "hit_tpr", "collateral_tpr",
                       "pre_lin", "hit_lin", "collateral_lin"):
            frame[column] = table[column].to_numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, int]:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    means = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), len(v)


def summarize(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (method, scale, population), g in table.groupby(["method", "scale", "population"],
                                                        sort=False):
        record = {"method": method, "scale": scale, "population": population, "n_events": len(g)}
        for key in ("d_lp_cf", "d_lp_real", "d_lp_legal", "collateral_tpr", "collateral_lin"):
            mean, lo, hi, n = boot_ci(g[key].to_numpy(float), rng)
            record |= {f"{key}_mean": mean, f"{key}_lo": lo, f"{key}_hi": hi, f"{key}_n": n}
        for key in ("hit_tpr", "hit_lin"):
            mean, lo, hi, n = boot_ci(g[key].to_numpy(float), rng)
            record |= {f"{key}_rate": mean, f"{key}_lo": lo, f"{key}_hi": hi, f"{key}_n": n}
        record |= {
            "clean_tpr": float((g["collateral_tpr"] == 0).mean()),
            "clean_lin": float((g["collateral_lin"] == 0).mean()),
            "pre_reads_theirs_tpr": float((g["pre_tpr"] == THEIRS).mean()),
            "pre_reads_theirs_lin": float((g["pre_lin"] == THEIRS).mean()),
            "edit_norm_mean": float(g["edit_norm"].mean()),
            "rel_edit_norm_mean": float((g["edit_norm"] / g["resid_norm"]).mean()),
        }
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_events", type=int, default=0, help="0 = every mined event")
    parser.add_argument("--scales", default="1,2,3,4,6,8,12,16")
    parser.add_argument("--betas", default="1,2")
    parser.add_argument("--corpus", default=PRIMARY_CORPUS, choices=CORPORA,
                        help="which corpus both probes are fit on; non-primary runs "
                             "write suffixed CSVs so the committed ones survive")
    args = parser.parse_args()
    tag = "" if args.corpus == PRIMARY_CORPUS else f"_{args.corpus}"

    device = select_device()
    events = pd.read_csv(study_dir("bracket") / "events" / "events.csv")
    if args.n_events:
        events = events.iloc[:args.n_events]
    rows, skips = build_rows(events, Random(SEED))
    counts = pd.Series([r.population for r in rows]).value_counts()
    print(f"{len(events)} events -> {len(rows)} rows: {counts.to_dict()}; skips {skips}",
          flush=True)

    fit = probes(args.corpus)
    tpr_dirs = torch.stack([fit.tpr.edit_direction(q, THEIRS, MINE) for q in range(64)])
    dhat, mu = mine_direction(fit)
    fit.tpr.module.to(device)
    dirs = Directions(tpr=tpr_dirs.to(device), linear=dhat.to(device), mu_mine=mu.to(device))
    readout = Readout(tpr=fit.tpr, linear=fit.linear.to(device))

    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device).eval()

    betas = [float(b) for b in args.betas.split(",")]
    conditions = [Condition("base", 0.0)]
    conditions += [Condition("tpr", float(s)) for s in args.scales.split(",")]
    conditions += [Condition("linear", b) for b in betas]
    conditions += [Condition("tpr_matched", b) for b in betas]

    population = pd.Series([r.population for r in rows])
    passes = {}
    for cond in conditions:
        done = run_pass(model, rows, cond, dirs, readout, device)
        passes[cond] = done
        hits = done["hit_tpr"].groupby(population).mean()
        print(f"{cond.method} scale={cond.scale}: "
              + ", ".join(f"{pop} hit_tpr={rate:.3f}" for pop, rate in hits.items()), flush=True)

    base = passes.pop(Condition("base", 0.0))
    table = per_row_table(rows, base, passes)
    rng = np.random.default_rng(SEED)
    summary = summarize(table, rng)

    out = repo_root() / "analysis" / "bracket_flip" / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / f"intervention_per_event{tag}.csv", index=False)
    summary.round(4).to_csv(out / f"intervention_summary{tag}.csv", index=False)
    (out / f"intervention_manifest{tag}.json").write_text(json.dumps({
        "model": MODEL_NAME, "hook": HOOK_POINT, "layer": LAYER,
        "n_events": len(events), "n_rows": len(rows),
        "rows_per_population": {k: int(v) for k, v in counts.items()},
        "skips": skips, "scales": args.scales, "betas": args.betas,
        "settled_min_age": SETTLED_MIN_AGE, "n_bootstrap": N_BOOT, "seed": SEED,
        "probe_corpus": args.corpus, **fit.provenance,
    }, indent=2))
    print(summary.to_string(index=False))
    print(f"wrote {out / f'intervention_per_event{tag}.csv'}")


if __name__ == "__main__":
    main()
