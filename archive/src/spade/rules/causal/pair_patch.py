"""R5 localization: same-prefix pair patching, scored on flip-diagnostic squares.

Clean = prefix + m1, corrupted = prefix + m2 (both legal, different flip
sets). Each layer's attn_out and mlp_out at the injected position is patched
clean -> corrupted (single component, rest of the stream corrupted; see
patch_battery.py for why full-residual patches are tautological here), and
the score S = logmass(legal only after m1) - logmass(legal only after m2) is
tracked. Two diagnostic variants: `all` (every legality-differing square) and
`strict` (squares whose eight rays avoid both placements, so the legality
difference can only come through flipped discs).
Recovery = (S_patch - S_corr) / (S_clean - S_corr).

Usage: python -m spade.rules.causal.pair_patch [--n_pairs 4000]
"""

import argparse

import pandas as pd
import torch

from spade.common.config import HF_REVISIONS, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.othello.boards import rays_touch, regular_moves, replay, tokens_of
from spade.othello.taxonomy import LEGAL
from spade.rules.causal.patch_battery import PATCH_HOOKS, battery_logprobs

force_ipv4()

from circuits.othello_engine_utils import to_string

BATCH_SIZE = 128


def build_pairs(items: dict, n_pairs: int) -> list[dict]:
    primary = ((items["category"] == LEGAL) & (items["is_control_legal"] == 0)) \
        .nonzero().flatten().tolist()
    control_of = {int(items["prefix_id"][i]): i for i in
        (items["is_control_legal"] == 1).nonzero().flatten().tolist()}
    pairs = []
    for i1 in primary[:n_pairs]:
        i2 = control_of[int(items["prefix_id"][i1])]
        length = int(items["length"][i1])
        seq1 = items["tokens"][i1, :length].long().tolist()
        seq2 = items["tokens"][i2, :length].long().tolist()
        board = replay(seq1[:-1])
        mover = board.next_hand_color
        b1, b2 = replay(seq1), replay(seq2)
        l1 = regular_moves(b1.state, -mover)
        l2 = regular_moves(b2.state, -mover)
        if not l1 or not l2:
            continue  # forfeit on either branch: mover identity differs
        only1, only2 = l1 - l2, l2 - l1
        if not only1 or not only2:
            continue
        placements = {to_string(seq1[-1]), to_string(seq2[-1])}
        strict1 = {x for x in only1 if not rays_touch(x, placements)}
        strict2 = {x for x in only2 if not rays_touch(x, placements)}
        pairs.append({
            "pair": int(items["prefix_id"][i1]), "t": int(items["t"][i1]),
            "seq1": seq1, "seq2": seq2,
            "sets": {"all": (tokens_of(only1), tokens_of(only2)),
                     "strict": (tokens_of(strict1), tokens_of(strict2))
                         if strict1 and strict2 else None},
        })
    return pairs


def score(logprobs: torch.Tensor, sets) -> float:
    only1, only2 = sets
    return float(logprobs[only1].logsumexp(0) - logprobs[only2].logsumexp(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=4000)
    args = parser.parse_args()

    device = select_device()
    items = torch.load(study_dir("rules") / "pairs" / "items.pt", weights_only=True)
    pairs = build_pairs(items, args.n_pairs)
    n_strict = sum(p["sets"]["strict"] is not None for p in pairs)
    print(f"{len(pairs)} usable pairs ({n_strict} with strict diagnostics)")

    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device=device)
    model.eval()

    records = []
    by_len: dict[int, list[int]] = {}
    for j, p in enumerate(pairs):
        by_len.setdefault(len(p["seq1"]), []).append(j)
    for length, idxs in by_len.items():
        for start in range(0, len(idxs), BATCH_SIZE):
            batch = [pairs[j] for j in idxs[start:start + BATCH_SIZE]]
            tok1 = torch.tensor([p["seq1"] for p in batch], device=device)
            tok2 = torch.tensor([p["seq2"] for p in batch], device=device)
            logprobs = battery_logprobs(model, tok1, tok2)
            for k, p in enumerate(batch):
                for variant, sets in p["sets"].items():
                    if sets is None:
                        continue
                    s_clean = score(logprobs["clean"][k], sets)
                    s_corr = score(logprobs["corr"][k], sets)
                    for key in PATCH_HOOKS:
                        s_patch = score(logprobs[key][k], sets)
                        records.append({
                            "pair": p["pair"], "t": p["t"], "variant": variant,
                            "patch": key, "s_clean": s_clean, "s_corr": s_corr,
                            "s_patch": s_patch,
                            "recovery": (s_patch - s_corr) / (s_clean - s_corr)
                                if s_clean != s_corr else float("nan"),
                            "n_only1": len(sets[0]), "n_only2": len(sets[1]),
                        })
        print(f"length {length}: done")

    out = repo_root() / "analysis" / "rules_causal" / "results"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out / "pair_patch.csv", index=False)
    print(df.groupby(["variant", "patch"])["recovery"].mean().unstack(0))
    print(f"wrote {out / 'pair_patch.csv'}")


if __name__ == "__main__":
    main()
