"""M5 and F6 of the violation-divergence study: does the divergence reach behaviour?

Three read-outs of the model's next-move distribution, event-locked to the branch
position t0 with the same kernels M1 uses, so the two are read on one x-axis:

  KL(p_branch || p_reference)  how far the branch's next-move distribution has
                               moved from the real game's, in nats
  legal mass                   probability on the moves that are legal on the
                               *reference* board -- does the model's notion of
                               legality follow the true board, or the corrupted
                               history it was fed?
  top-1 legality               how often the branch's argmax move is legal there,
                               against the reference branch's own rate as ceiling

`r` indexes the board position s = t0 + r whose logits row it is, the convention
`divergence.align` uses, and `logits[:, s]` is the distribution over the move at
s + 1. So r = 0 is the first prediction made after the branch token entered, and
KL is exactly zero for r < 0 -- asserted, not assumed.

Softmax in float32 over the whole 61-wide vocabulary, pass token (index 0)
included: the distribution stays the model's own, and `p_pass` is reported so a
reader can see how little it carries. `fp16_floor` measures what the fp16
storage of acts/logits.pt costs a KL, since a KL read off quantised logits has a
floor.

`--variant rollout` reads the tree where every branch, the reference included,
samples its own continuation. Two things change and the figure says both. Each
of the K rollouts of a game carries its own reference, so the pairing is per
(game, rollout) and the K rows of a branch are K samples of it -- the bootstrap
resamples source games, which holds them together, and the legend counts
branches. And the reference is now a *sampled* game, so its board exists only
while OthelloGPT keeps playing legally: the legal-set mask stops at the first
illegal move of the reference rollout and the positions after it are NaN, not
zero. KL needs no board and keeps its full n.

Usage: python analysis/violation_divergence/behavioral/make_figures.py [--smoke]
                                            [--variant rollout] [--camera-ready]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from spade.common.config import CTX_LEN
from spade.common.paths import results_dir
from spade.common.study import variant_dir
from spade.figures.paper_style import (
    CATEGORY_STYLE, GRID, INK, MUTED, NULL, REFERENCE, apply_style, save_figure)
from spade.othello.boards import regular_moves, replay, tokens_of
from spade.violation_divergence.branches import role
from spade.violation_divergence.compute_divergence import group_key, pairs
from spade.violation_divergence.divergence import PLATEAU_R, WINDOW, align, persistence

from circuits.othello_engine_utils import OthelloBoardState, to_string

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = results_dir("violation_divergence")

VOCAB = 61          # 60 board squares, plus the pass token at index 0
SQUARES = 60        # the chance model: uniform over the tokens that name a square
ROW_BATCH = 1024    # 1024 seqs x 59 positions x 61 tokens x 4 B = 15 MB per tensor
N_BOOT = 1000
FLOAT16_SAMPLE = 20000  # (row, position) pairs behind the quantisation floor

VARIANTS = ("spliced", "rollout")


def stem(name: str, variant: str) -> str:
    """Today's names stay with the spliced tree; a variant gets its own."""
    return name if variant == "spliced" else f"{name}_{variant}"


ILLEGAL = [c for c in CATEGORY_STYLE if c != "legal"]
CURVES = ["reference", *ILLEGAL, "control_legal"]  # baseline first, null last
BRANCH_STYLE = {
    "reference": {**REFERENCE, "label": "reference game (baseline)"},
    **{c: CATEGORY_STYLE[c] for c in ILLEGAL},
    "control_legal": {**NULL, "label": "control_legal (null)"},
}
PRE = np.asarray(WINDOW) < 0
PLATEAU = (np.asarray(WINDOW) >= PLATEAU_R[0]) & (np.asarray(WINDOW) <= PLATEAU_R[1])


# ---------------------------------------------------------------- measurement

def reference_rows(meta: dict[str, torch.Tensor]) -> torch.Tensor:
    """The reference row of each comparison group, in group order.

    A group is a source game, or in the rollout tree one (game, rollout) sample,
    since there each rollout carries its own reference.
    """
    rows = ((meta["category"] == 0) & (meta["is_control"] == 0)).nonzero().flatten()
    assert torch.equal(group_key(meta)[rows], torch.arange(len(rows))), \
        "reference rows are not one per group in group order"
    return rows


def legal_masks(meta: dict[str, torch.Tensor],
                sampled: bool) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Per group and board position, which tokens are legal on the reference board,
    and which colour moves next there.

    Only window positions are filled; everything else stays False and is never
    read. Where the nominal mover has no regular move the engine lets the
    opponent play (R4 forfeit) and that is what the next token is, so that is the
    set scored. An all-False window position is a finished game, and the legal
    mass there is undefined rather than zero.

    `sampled` says the reference's continuation came from the model rather than
    from the source game, so it can leave the legal game and stop having a board
    at all. Filling stops there and the rest of the window stays undefined; the
    count of references that do this is returned and reported on the figure.
    """
    rows = reference_rows(meta)
    mask = np.zeros((len(rows), CTX_LEN, VOCAB), dtype=bool)
    to_move = np.zeros((len(rows), CTX_LEN), dtype=np.int8)
    forfeits, left_legal = 0, 0
    for group, row in enumerate(rows.tolist()):
        tokens = meta["tokens"][row].long().tolist()
        window = {int(meta["t0"][row]) + r for r in WINDOW}
        board = OthelloBoardState()
        for s, token in enumerate(tokens):
            square = to_string(token)
            if sampled and square not in (regular_moves(board.state, board.next_hand_color)
                                          or regular_moves(board.state, -board.next_hand_color)):
                left_legal += s <= max(window)
                break
            board.umpire(square)
            if s not in window:
                continue
            mover = regular_moves(board.state, board.next_hand_color)
            forfeits += not mover
            legal = mover or regular_moves(board.state, -board.next_hand_color)
            mask[group, s, tokens_of(legal)] = True
            to_move[group, s] = board.next_hand_color if mover else -board.next_hand_color
    return mask, to_move, forfeits, left_legal


def branch_board_mass(logits: torch.Tensor, meta: dict[str, torch.Tensor],
                      rng: np.random.Generator) -> dict[str, float]:
    """`control_legal`'s mass at r = 0 on the legal set of the board it actually played.

    The illegal branches have no board -- updating one after a disc lands on an
    occupied square is exactly what the shared-suffix design avoids -- so this is
    the only branch that can say what "the model followed the history it was fed"
    looks like. It is the yardstick the reference-board masses are read against.
    """
    rows = (meta["is_control"] == 1).nonzero().flatten()
    mass = []
    for row in rows.tolist():
        t0 = int(meta["t0"][row])
        board = replay(meta["tokens"][row, :t0 + 1].long().tolist())
        legal = regular_moves(board.state, board.next_hand_color) \
            or regular_moves(board.state, -board.next_hand_color)
        mass.append(float(logits[row, t0].float().softmax(-1)[tokens_of(legal)].sum()))
    mass = np.asarray(mass)[:, None]
    lo, hi = game_ci(mass, np.arange(len(mass)), rng)
    return {"legal_mass_control_own_board_r0": float(mass.mean()),
            "legal_mass_control_own_board_r0_ci_lo": float(lo[0]),
            "legal_mass_control_own_board_r0_ci_hi": float(hi[0])}


def read_out(logits: torch.Tensor, meta: dict[str, torch.Tensor],
             mask: np.ndarray) -> dict[str, np.ndarray]:
    """Event-locked [n_row, len(WINDOW)] behaviour, one row per branch sequence.

    Reference rows are kept: their KL against themselves is the exactness check on
    the pairing, and their legal mass is the ceiling every other curve is read
    against.
    """
    branch, reference, _ = pairs(meta)
    own = reference_rows(meta)
    rows = torch.cat([branch, own])
    against = torch.cat([reference, own])
    # `group` indexes the legal-set mask (one reference per group); `game` is the
    # bootstrap's cluster, and in the rollout tree one game holds K groups.
    group, game = group_key(meta)[rows], meta["game_id"][rows]
    mask = torch.from_numpy(mask)

    per_position = {key: torch.empty(len(rows), CTX_LEN)
                    for key in ("kl", "legal_mass", "top1_legal", "p_pass", "n_legal_ref")}
    for i in range(0, len(rows), ROW_BATCH):
        sl = slice(i, i + ROW_BATCH)
        logp = logits[rows[sl]].float().log_softmax(-1)
        p = logp.exp()
        legal = mask[group[sl]]
        per_position["kl"][sl] = (p * (logp - logits[against[sl]].float().log_softmax(-1))).sum(-1)
        per_position["legal_mass"][sl] = (p * legal).sum(-1)
        per_position["top1_legal"][sl] = legal.gather(
            -1, p.argmax(-1, keepdim=True)).squeeze(-1).float()
        per_position["p_pass"][sl] = p[..., 0]
        per_position["n_legal_ref"][sl] = legal.sum(-1).float()

    t0 = meta["t0"][rows]
    out = {key: align(v, t0, WINDOW).numpy() for key, v in per_position.items()}
    finished = out["n_legal_ref"] == 0
    out["legal_mass"][finished] = np.nan
    out["top1_legal"][finished] = np.nan
    out["branch"] = np.array([role(int(c), int(k)) for c, k
                              in zip(meta["category"][rows], meta["is_control"][rows])])
    out["game"] = game.numpy()
    out["n_finished"] = finished.sum()
    return out


def turn_parity(to_move: np.ndarray, t0: np.ndarray) -> dict[str, float]:
    """Share of window positions where the reference game's next mover is the colour
    that played the branch move -- the sawtooth every curve carries, measured."""
    offsets = np.asarray(WINDOW)
    mover = np.take_along_axis(to_move, t0[:, None] + offsets, axis=1)
    same = mover == mover[:, [list(WINDOW).index(-1)]]  # r = -1 fixes the branch mover
    odd = offsets % 2 == 1
    return {"same_mover_as_branch_odd_r": float(same[:, odd].mean()),
            "same_mover_as_branch_even_r": float(same[:, ~odd].mean())}


def fp16_floor(logits: torch.Tensor, rng: np.random.Generator) -> dict[str, float]:
    """KL that one more fp16 rounding of the stored logits invents out of nothing.

    acts/logits.pt is fp16, so the measured KL carries the error of two roundings
    that already happened. Perturbing the stored values by their own half-ulp and
    re-measuring bounds what those roundings can be worth.
    """
    row = rng.integers(len(logits), size=FLOAT16_SAMPLE)
    position = rng.integers(1, CTX_LEN, size=FLOAT16_SAMPLE)  # position 0 is the sink
    x = logits[row, position].float()
    half_ulp = np.abs(np.spacing(logits[row, position].numpy())) / 2
    noise = torch.from_numpy(rng.uniform(-half_ulp, half_ulp).astype(np.float32))
    logp, logq = x.log_softmax(-1), (x + noise).log_softmax(-1)
    floor = (logp.exp() * (logp - logq)).sum(-1).numpy()
    return {"kl_fp16_floor_median": float(np.median(floor)),
            "kl_fp16_floor_p99": float(np.percentile(floor, 99))}


# ---------------------------------------------------------------- aggregation

def game_ci(values: np.ndarray, game: np.ndarray,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """95% percentile CI of each column's mean, resampling source games.

    Games are the unit of resampling because a game contributes at most one row
    per category and its branches share a prefix; rows of a resampled game move
    together. NaN columns (a finished game has no legal set) drop out of both
    numerator and denominator, so each column's mean is over its own live rows.
    """
    _, index = np.unique(game, return_inverse=True)
    counts = rng.multinomial(index.max() + 1, np.full(index.max() + 1, 1 / (index.max() + 1)),
                             size=N_BOOT)
    weight = counts[:, index].astype(np.float32)
    live = np.isfinite(values).astype(np.float32)
    means = (weight @ np.nan_to_num(values)) / (weight @ live)
    return np.percentile(means, [2.5, 97.5], axis=0)


def curves(data: dict[str, np.ndarray], rng: np.random.Generator) -> pd.DataFrame:
    """Per branch class, the three read-outs against r, with a game bootstrap.

    `n_rows` counts sequences, so in the rollout tree it is K per branch and the
    figure divides it down. The column means are over rows and a branch's K are
    balanced, so the estimate is a branch mean either way; the interval is a game
    bootstrap, which holds a branch's K rows together."""
    rows = []
    for name in [*CURVES, "illegal_pooled"]:
        sel = np.isin(data["branch"], ILLEGAL) if name == "illegal_pooled" \
            else data["branch"] == name
        game = data["game"][sel]
        frame = {"branch": name, "r": WINDOW, "n_rows": int(sel.sum()),
                 "n_games": len(np.unique(game))}
        for metric in ("kl", "legal_mass", "top1_legal", "p_pass"):
            values = data[metric][sel]
            lo, hi = game_ci(values, game, rng)
            frame[f"{metric}_mean"] = np.nanmean(values, axis=0)
            frame[f"{metric}_lo"], frame[f"{metric}_hi"] = lo, hi
        frame["kl_median"] = np.nanmedian(data["kl"][sel], axis=0)
        frame["legal_chance"] = np.nanmean(data["n_legal_ref"][sel], axis=0) / SQUARES
        rows.append(pd.DataFrame(frame))
    return pd.concat(rows, ignore_index=True)


def summary(data: dict[str, np.ndarray], curve: pd.DataFrame,
            extra: dict[str, float]) -> pd.DataFrame:
    """The contract keys, plus everything RESULTS.md quotes."""
    rows = []
    for name in [*CURVES, "illegal_pooled"]:
        c = curve[curve.branch == name].reset_index(drop=True)
        sel = np.isin(data["branch"], ILLEGAL) if name == "illegal_pooled" \
            else data["branch"] == name
        post = c.kl_mean.where(c.r >= 0)
        peak = int(post.idxmax())
        run = persistence(torch.from_numpy(data["kl"][sel]), WINDOW)
        low = int(c.legal_mass_mean.where(c.r >= 0).idxmin())
        values = {
            "kl_peak": c.kl_mean[peak], "kl_peak_r": c.r[peak],
            "kl_peak_ci_lo": c.kl_lo[peak], "kl_peak_ci_hi": c.kl_hi[peak],
            "kl_peak_r_per_branch": float(run["peak_r"].float().mean()),
            "kl_half_life_censored": float(torch.isinf(run["half_life"]).float().mean()),
            "kl_at_r0": float(c.kl_mean[list(WINDOW).index(0)]),
            "kl_plateau": float(np.nanmean(data["kl"][sel][:, PLATEAU])),
            "legal_mass_pre_t0": float(np.nanmean(data["legal_mass"][sel][:, PRE])),
            "legal_mass_r0": c.legal_mass_mean[list(WINDOW).index(0)],
            "legal_mass_min": c.legal_mass_mean[low], "legal_mass_min_r": c.r[low],
            "legal_mass_plateau": float(np.nanmean(data["legal_mass"][sel][:, PLATEAU])),
            "top1_legal_pre_t0": float(np.nanmean(data["top1_legal"][sel][:, PRE])),
            "top1_legal_r0": c.top1_legal_mean[list(WINDOW).index(0)],
            "top1_legal_plateau": float(np.nanmean(data["top1_legal"][sel][:, PLATEAU])),
            "p_pass_max": float(np.nanmax(c.p_pass_mean)),
        }
        rows += [{"key": k, "branch": name, "value": float(v),
                  "n_rows": int(c.n_rows[0]), "n_games": int(c.n_games[0])}
                 for k, v in values.items()]
    rows += [{"key": k, "branch": "all", "value": v, "n_rows": len(data["branch"]),
              "n_games": len(np.unique(data["game"]))} for k, v in extra.items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- F6

def panel_letter(ax, i: int) -> None:
    ax.text(0.0, 1.02, f"({chr(97 + i)})", va="bottom", transform=ax.transAxes,
            color=INK, fontsize=9)


def pre_event(ax) -> None:
    """The r < 0 baseline window, shaded, with the branch move at 0."""
    ax.axvspan(WINDOW[0], 0, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)


def curve(ax, c: pd.DataFrame, style: dict, label: str) -> None:
    ax.plot(c.r, c.estimate, ms=4.0, markeredgecolor="white", markeredgewidth=0.5,
            label=label, **{"lw": 1.6, "zorder": 3,
                            **{k: v for k, v in style.items() if k != "label"}})


def named(curve_data: pd.DataFrame, metric: str) -> pd.DataFrame:
    return curve_data.rename(columns={f"{metric}_mean": "estimate"})


KL_TITLE = {"spliced": "the break is immediate and never heals",
            "rollout": "two independent samples of the future, drifting apart"}
KL_PLATEAU_TAIL = {"spliced": "of the peak — no recovery",
                   "rollout": "of the peak — still climbing at the window edge"}
LEGAL_TITLE = {"spliced": "legality still tracks the true board",
               "rollout": "with a sampled reference board, no class tracks it"}
LEGAL_YLABEL = {
    "spliced": "probability on the reference board's legal moves\n"
               "higher = legality still tracks the true board",
    "rollout": "probability on the reference rollout's legal moves\n"
               "higher = the branch still predicts the reference's board",
}


def kl_panel(ax, curve_data: pd.DataFrame, k: int, variant: str) -> None:
    for name in [*ILLEGAL, "control_legal"]:
        c = named(curve_data[curve_data.branch == name], "kl")
        curve(ax, c, BRANCH_STYLE[name],
              f"{BRANCH_STYLE[name]['label']} (n={int(c.n_rows.iloc[0]) // k:,})")
    pooled = curve_data[curve_data.branch == "illegal_pooled"].reset_index(drop=True)
    peak = int(pooled.kl_mean.where(pooled.r >= 0).idxmax())
    plateau = float(pooled.kl_mean[PLATEAU].mean())
    ax.annotate(f"pooled illegal: {pooled.kl_mean[list(WINDOW).index(0)]:.2f} nats already at "
                f"$r$=0,\npeak {pooled.kl_mean[peak]:.2f} at $r$={pooled.r[peak]}",
                xy=(pooled.r[peak], pooled.kl_mean[peak]), xytext=(4.5, 0.955),
                textcoords=("data", "axes fraction"), va="top", fontsize=7.5, color=INK,
                arrowprops={"arrowstyle": "->", "lw": 0.8, "color": MUTED})
    ax.annotate(f"plateau over $r\\in[{PLATEAU_R[0]}, {PLATEAU_R[1]}]$: {plateau:.2f} nats,\n"
                f"{plateau / pooled.kl_mean[peak]:.0%} {KL_PLATEAU_TAIL[variant]}",
                xy=(22, plateau), xytext=(11.5, 0.30), textcoords=("data", "axes fraction"),
                va="top", fontsize=7.5, color=INK,
                arrowprops={"arrowstyle": "->", "lw": 0.8, "color": MUTED})
    ax.set_ylim(bottom=0)
    ax.set_ylabel("$KL(p_{\\mathrm{branch}} \\Vert p_{\\mathrm{reference}})$ (nats)\n"
                  "higher = the prediction has moved further")
    ax.set_title(KL_TITLE[variant], fontsize=9.5)


def legal_panel(ax, curve_data: pd.DataFrame, control_own_board: float, k: int,
                variant: str) -> None:
    for name in CURVES:
        c = named(curve_data[curve_data.branch == name], "legal_mass")
        curve(ax, c, BRANCH_STYLE[name],
              f"{BRANCH_STYLE[name]['label']} (n={int(c.n_rows.iloc[0]) // k:,})")
    reference = curve_data[curve_data.branch == "reference"].reset_index(drop=True)
    baseline = float(reference.legal_mass_mean[PRE].mean())
    ax.axhline(baseline, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.plot(reference.r, reference.legal_chance, color=MUTED, lw=0.8, ls=(0, (1, 2)),
            zorder=1, label=f"uniform over the {SQUARES} squares (chance)")
    pooled = curve_data[curve_data.branch == "illegal_pooled"].reset_index(drop=True)
    low = int(pooled.legal_mass_mean.where(pooled.r >= 0).idxmin())
    control = curve_data[curve_data.branch == "control_legal"].reset_index(drop=True)
    tail = (f"control_legal is lowest\nbecause it puts {control_own_board:.3f} on its "
            "*own* board's legal set" if variant == "spliced" else
            "the legal control falls\nwith it, to "
            f"{float(control.legal_mass_mean[PLATEAU].mean()):.2f} — this read-out no "
            "longer\nseparates legality from a different legal move")
    ax.annotate(f"pooled illegal falls to {pooled.legal_mass_mean[low]:.3f} at $r$={pooled.r[low]}\n"
                f"and settles near {float(pooled.legal_mass_mean[PLATEAU].mean()):.2f}. "
                + tail,
                xy=(pooled.r[low], pooled.legal_mass_mean[low]), xytext=(3.0, 0.52),
                textcoords=("data", "axes fraction"), va="top", fontsize=7.5, color=INK,
                arrowprops={"arrowstyle": "->", "lw": 0.8, "color": MUTED})
    ax.set_ylim(0, 1.04)
    ax.set_ylabel(LEGAL_YLABEL[variant])
    ax.set_title(LEGAL_TITLE[variant], fontsize=9.5)


F6_TITLE = {
    "spliced": "An illegal move changes what OthelloGPT predicts, immediately and for "
               "the rest of the game",
    "rollout": "Sampling every branch's future removes the shared suffix, and most of "
               "the read-out gap with it",
}


def f6(curve_data: pd.DataFrame, control_own_board: float, variant: str, k: int,
       left_legal: float, camera_ready: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    kl_panel(axes[0], curve_data, k, variant)
    legal_panel(axes[1], curve_data, control_own_board, k, variant)
    for i, ax in enumerate(axes):
        pre_event(ax)
        panel_letter(ax, i)
        ax.set_xlabel("move offset $r = s - t_0$ (plies since the branch move)")
        ax.set_xlim(WINDOW[0], WINDOW[-1] + 0.4)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=4,
               fontsize=8, title_fontsize=8,
               title="means over source games; n = branch sequences, one per game; 95% CIs "
                     "in behavioral_curves.csv. The reference is 0 nats by construction "
                     "and drawn only in (b), where it is the ceiling."
               if variant == "spliced" else
               f"means over source branches, a branch's K={k} rollouts held together; "
               "n = source branches; 95% CIs in behavioral_curves_rollout.csv.\n"
               "In (b) the sampled reference's board ends at its first illegal move in "
               f"{left_legal:.1%} of rollouts; those positions are dropped, not zeroed.")
    if not camera_ready:
        fig.suptitle(F6_TITLE[variant], y=1.0)
    fig.tight_layout()
    save_figure(fig, FIGURES / stem("f6_behavioral_divergence", variant), camera_ready)


# ---------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="the 50-game set under $DIR/smoke/")
    ap.add_argument("--variant", default="spliced", choices=VARIANTS,
                    help="rollout: every branch samples its own continuation "
                         "($DIR[/smoke]/rollout/), K per branch")
    ap.add_argument("--camera-ready", action="store_true",
                    help="vector PDF, no in-figure title")
    args = ap.parse_args()
    apply_style(camera_ready=args.camera_ready)
    rng = np.random.default_rng(0)
    variant = args.variant
    sampled = variant == "rollout"

    base = variant_dir("divergence", args.smoke, "" if variant == "spliced" else variant)
    meta = torch.load(base / "branches" / "branches.pt", weights_only=True)
    logits = torch.load(base / "acts" / "logits.pt", weights_only=True)
    assert logits.shape[1:] == (CTX_LEN, VOCAB), f"unexpected logits shape {logits.shape}"
    k = int(meta["rollout"].max()) + 1 if sampled else 1

    mask, to_move, forfeits, left_legal = legal_masks(meta, sampled)
    data = read_out(logits, meta, mask)
    assert (data["kl"][data["branch"] == "reference"] == 0).all(), \
        "a reference row diverges from itself"
    assert (data["kl"][:, PRE] == 0).all(), "KL nonzero before t0"

    # A spliced reference replays a real game, so it cannot leave the legal game and
    # the count is reported only where it can be nonzero.
    left_legal_frac = left_legal / len(reference_rows(meta))
    extra = {"n_forfeit_positions": forfeits, "n_finished_positions": int(data["n_finished"]),
             **({"n_reference_left_legal": left_legal,
                 "frac_reference_left_legal": left_legal_frac} if sampled else {}),
             "mean_reference_legal_moves": float(np.nanmean(data["n_legal_ref"])),
             **turn_parity(to_move, meta["t0"][reference_rows(meta)].numpy()),
             **branch_board_mass(logits, meta, rng), **fp16_floor(logits, rng)}
    curve_data = curves(data, rng)
    table = summary(data, curve_data, extra)

    RESULTS.mkdir(parents=True, exist_ok=True)
    curve_data.to_csv(RESULTS / f"{stem('behavioral_curves', variant)}.csv", index=False,
                      float_format="%.6g")
    table.to_csv(RESULTS / f"{stem('behavioral_summary', variant)}.csv", index=False,
                 float_format="%.6g")
    f6(curve_data, extra["legal_mass_control_own_board_r0"], variant, k,
       left_legal_frac, args.camera_ready)
    print(table[table.branch.isin(["illegal_pooled", "control_legal", "reference", "all"])]
          .to_string(index=False))


if __name__ == "__main__":
    main()
