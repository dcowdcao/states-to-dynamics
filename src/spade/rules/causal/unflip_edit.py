"""R5 primary causal test: the "un-flip" edit.

For a legal move m that flips square s, edit layer-5 resid_post at the move's
position along the board-state probe direction so s reads as its pre-flip
color, then measure how next-move probability mass moves between the real
board's legal set and the counterfactual (s un-flipped) board's legal set on
diagnostic squares (legal on exactly one of the two boards).

Populations compared at matched edit magnitude:
  flipped  — s was just flipped by m               (theirs -> mine edit)
  settled  — same-color disc untouched >= 10 plies (theirs -> mine edit)
  placed   — the disc m itself just placed          (theirs -> empty edit)
  random   — random direction, magnitude matched to the flipped edit

Edit magnitudes are calibrated on the canonical probe corpus's test games -- the
split the probe never saw: the edit moves the activation's projection onto the
class-difference direction to the target class's mean projection there (times
--beta overshoot).

Usage: python -m spade.rules.causal.unflip_edit [--n_items 2000] [--betas 1,2]
"""

import argparse
from random import Random

import pandas as pd
import torch

from spade.common.config import HF_REVISIONS, HOOK_POINT, LAYER, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.othello.boards import flip_set, regular_moves, replay, tokens_of
from spade.othello.taxonomy import LEGAL
from spade.probes import load as probes
from spade.probes.train import activations, corpus, game_split, rows as game_rows

force_ipv4()

from circuits.othello_engine_utils import to_string

EMPTY, THEIRS, MINE = 0, 1, 2
SETTLED_MIN_AGE = 10
BATCH_SIZE = 128


def probe_directions(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-square unit direction and per-(square, class) mean projection onto it.
    Returns (dhat [64, 2, 512], mu [64, 2, 3]) where dim 1 indexes the edit
    transition family: 0 = mine-vs-theirs, 1 = empty-vs-theirs.

    The probe is the canonical board probe (`spade.probes.load`). mu is the target
    the edit drives the activation to, so it is measured on the split that probe
    never trained on -- its own corpus's test games.
    """
    probe = probes.board(LAYER, flat=True).permute(1, 2, 0)  # [64, 3, 512]
    dhat = torch.stack([probe[:, MINE] - probe[:, THEIRS],
                        probe[:, EMPTY] - probe[:, THEIRS]], dim=1)
    dhat = dhat / dhat.norm(dim=-1, keepdim=True)

    y, games, n_games = corpus()
    held = game_rows(games, game_split(n_games, probes.card(LAYER)["seed"])[2])
    acts, labels = activations(LAYER)[held].float(), y[held]
    proj = torch.einsum("nd,sfd->nsf", acts, dhat)  # [N, 64, 2]
    mu = torch.stack([torch.stack([proj[labels[:, s] == cls, s, f].mean()
        for cls in (EMPTY, THEIRS, MINE)])
        for s in range(64) for f in (0, 1)]).reshape(64, 2, 3)
    return dhat.to(device), mu.to(device)


def build_rows(items: dict, n_items: int, rng: Random) -> list[dict]:
    """One row per (item, population) with board-derived scoring sets."""
    primary = ((items["category"] == LEGAL) & (items["is_control_legal"] == 0)) \
        .nonzero().flatten().tolist()[:n_items]
    rows = []
    for i in primary:
        length = int(items["length"][i])
        seq = items["tokens"][i, :length].long().tolist()
        board = replay(seq[:-1])
        mover = board.next_hand_color
        m_sq = to_string(seq[-1])
        flipped = flip_set(board.state, m_sq, mover)
        board.umpire(m_sq)
        real_legal = regular_moves(board.state, -mover)
        if not real_legal:
            continue  # forfeit after m: mover identity changes, skip

        s_flip = rng.choice(flipped)
        settled = [sq for sq in range(64)
            if board.state[sq // 8, sq % 8] == mover and board.age[sq // 8, sq % 8]
            >= SETTLED_MIN_AGE and sq not in flipped and sq != m_sq]
        targets = [("flipped", s_flip, 0, MINE), ("random", s_flip, 0, MINE)]
        if settled:
            targets.append(("settled", rng.choice(settled), 0, MINE))
        targets.append(("placed", m_sq, 1, EMPTY))

        for pop, s, family, target_cls in targets:
            state_cf = board.state.copy()
            state_cf[s // 8, s % 8] = -mover if target_cls == MINE else 0
            cf_legal = regular_moves(state_cf, -mover)
            if not cf_legal or real_legal == cf_legal:
                continue
            rows.append({
                "item": i, "seq": seq, "t": int(items["t"][i]),
                "population": pop, "square": s, "family": family,
                "target_cls": target_cls, "n_flipped": len(flipped),
                "real_only": tokens_of(real_legal - cf_legal),
                "cf_only": tokens_of(cf_legal - real_legal),
                "legal": tokens_of(real_legal),
            })
    return rows


def set_logmass(logprobs: torch.Tensor, token_sets: list[list[int]]) -> torch.Tensor:
    out = torch.full((len(token_sets),), float("nan"))
    for j, toks in enumerate(token_sets):
        if toks:
            out[j] = logprobs[j, toks].logsumexp(0)
    return out


def run_batches(model, rows: list[dict], device: str, dhat, mu, beta) -> pd.DataFrame:
    """beta=None -> base run (no edit). Returns per-row set log-masses."""
    records = []
    by_len: dict[int, list[int]] = {}
    for j, row in enumerate(rows):
        by_len.setdefault(len(row["seq"]), []).append(j)
    g = torch.Generator().manual_seed(13)
    for length, idxs in by_len.items():
        for start in range(0, len(idxs), BATCH_SIZE):
            batch = [rows[j] for j in idxs[start:start + BATCH_SIZE]]
            tokens = torch.tensor([r["seq"] for r in batch], device=device)
            hooks = []
            if beta is not None:
                d = torch.stack([dhat[r["square"], r["family"]] for r in batch])
                target = torch.stack([mu[r["square"], r["family"], r["target_cls"]]
                    for r in batch])
                rand = torch.randn(len(batch), 512, generator=g).to(device)
                is_rand = torch.tensor([r["population"] == "random" for r in batch],
                                       device=device)
                v = torch.where(is_rand.unsqueeze(1),
                                rand / rand.norm(dim=-1, keepdim=True), d)

                def edit(resid, hook, d=d, target=target, v=v):
                    proj = (resid[:, -1, :] * d).sum(-1)
                    resid[:, -1, :] += (beta * (target - proj)).unsqueeze(1) * v
                    return resid
                hooks = [(HOOK_POINT, edit)]
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            logprobs = logits[:, -1, :].log_softmax(-1).cpu()
            for k, r in enumerate(batch):
                lp = logprobs[k:k + 1]
                records.append({
                    "row": idxs[start + k],
                    "lp_real_only": float(set_logmass(lp, [r["real_only"]])),
                    "lp_cf_only": float(set_logmass(lp, [r["cf_only"]])),
                    "lp_legal": float(set_logmass(lp, [r["legal"]])),
                })
    return pd.DataFrame(records).set_index("row").sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_items", type=int, default=2000)
    parser.add_argument("--betas", default="1,2")
    args = parser.parse_args()
    betas = [float(b) for b in args.betas.split(",")]

    device = select_device()
    dhat, mu = probe_directions(device)
    items = torch.load(study_dir("rules") / "pairs" / "items.pt", weights_only=True)
    rows = build_rows(items, args.n_items, Random(7))
    counts = pd.Series([r["population"] for r in rows]).value_counts()
    print(f"{len(rows)} rows: {counts.to_dict()}")

    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device=device)
    model.eval()

    base = run_batches(model, rows, device, dhat, mu, beta=None)
    frames = []
    for beta in betas:
        edited = run_batches(model, rows, device, dhat, mu, beta=beta)
        f = pd.DataFrame({
            "item": [r["item"] for r in rows], "t": [r["t"] for r in rows],
            "population": [r["population"] for r in rows],
            "square": [r["square"] for r in rows],
            "n_flipped": [r["n_flipped"] for r in rows],
            "n_real_only": [len(r["real_only"]) for r in rows],
            "n_cf_only": [len(r["cf_only"]) for r in rows],
            "beta": beta,
            "d_lp_cf": edited["lp_cf_only"] - base["lp_cf_only"],
            "d_lp_real": edited["lp_real_only"] - base["lp_real_only"],
            "d_lp_legal": edited["lp_legal"] - base["lp_legal"],
            "base_lp_cf": base["lp_cf_only"], "base_lp_real": base["lp_real_only"],
        })
        frames.append(f)
        print(f"beta={beta}: " + ", ".join(
            f"{pop}: d_lp_cf={f[f.population == pop].d_lp_cf.mean():+.3f} "
            f"d_lp_real={f[f.population == pop].d_lp_real.mean():+.3f}"
            for pop in counts.index))

    out = repo_root() / "analysis" / "rules_causal" / "results"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(out / "unflip_edit.csv", index=False)
    print(f"wrote {out / 'unflip_edit.csv'}")


if __name__ == "__main__":
    main()
