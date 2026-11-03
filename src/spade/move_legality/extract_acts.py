"""Step 2: layer-5 resid_post at each fit row's read position.

A row's sequence is its source game with `token` spliced in at `position`; the read is at
that same position, which is the first one whose activation depends on the spliced token.
Positions after it are never read, so their tokens are the source game's and are there
only to keep every sequence 59 long.

One sequence per row rather than one per game: a game's legal rows could all be read off a
single pass, but then the legal class would come out of a different batch shape from the
illegal class and cuBLAS would reduce their shared prefix differently. A class difference
that is really a reduction-order difference is exactly the artifact this study cannot
afford, and the whole pass is a couple of minutes.

Usage: python -m spade.move_legality.extract_acts [--smoke]
"""

import argparse

import torch

from spade.common.config import CTX_LEN, D_MODEL, HF_REVISIONS, HOOK_POINT, LAYER, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.study import study_dir

force_ipv4()

BATCH_SIZE = 512
SMOKE_ROWS = 2048


def sequences(fit: dict, tokens: torch.Tensor) -> torch.Tensor:
    """[n_rows, 59] long: each row's source game with its token spliced in at its position."""
    seq = tokens[fit["game_id"].long()].long()
    seq[torch.arange(len(seq)), fit["position"].long()] = fit["token"].long()
    return seq


def extract(model, seq: torch.Tensor, position: torch.Tensor, device: str) -> torch.Tensor:
    out = torch.empty(len(seq), D_MODEL, dtype=torch.float16)
    for start in range(0, len(seq), BATCH_SIZE):
        batch = slice(start, start + BATCH_SIZE)
        _, cache = model.run_with_cache(seq[batch].to(device),
                                        names_filter=lambda name: name == HOOK_POINT)
        rows = cache[HOOK_POINT][torch.arange(len(cache[HOOK_POINT])),
                                 position[batch].long().to(device)]
        out[batch] = rows.cpu().half()
        if start % (20 * BATCH_SIZE) == 0:
            print(f"extracted {start + len(rows)}/{len(seq)}", flush=True)
    return out


def check_token_tracking(acts: torch.Tensor, fit: dict) -> None:
    """Within an item, two rows agree iff their spliced tokens do.

    `violation_divergence.extract_acts.check_prefix_invariance` over the fit set: the rows
    of an item share tokens[:t] exactly, so an activation that did not move with the token
    would mean the read position is wrong, and one that moved for equal tokens would mean
    the splice leaked.
    """
    key = fit["game_id"].long() * CTX_LEN + fit["position"].long()
    assert (key[1:] >= key[:-1]).all(), "rows are not grouped by item"
    bounds = [0] + (key[1:] != key[:-1]).nonzero().flatten().add(1).tolist() + [len(key)]
    for start, end in zip(bounds, bounds[1:]):
        rows, token = acts[start:end], fit["token"][start:end]
        same = (rows.unsqueeze(0) == rows.unsqueeze(1)).all(-1)
        assert torch.equal(same, token.unsqueeze(0) == token.unsqueeze(1)), \
            f"item {int(key[start])}: activations do not track the spliced token"
    print(f"token tracking ok on all {len(bounds) - 1} items", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help=f"first {SMOKE_ROWS} rows only")
    args = parser.parse_args()

    from transformer_lens import HookedTransformer

    fit_dir = study_dir("move_legality") / "fit_set"
    fit = {name: torch.load(fit_dir / f"{name}.pt", weights_only=True)
           for name in ["game_id", "position", "token"]}
    tokens = torch.load(fit_dir / "tokens.pt", weights_only=True)
    seq = sequences(fit, tokens)
    if args.smoke:
        seq, fit = seq[:SMOKE_ROWS], {k: v[:SMOKE_ROWS] for k, v in fit.items()}
    assert seq.shape[1] == CTX_LEN and (seq > 0).all(), "padding or pass token in a sequence"

    device = select_device()
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device=device).eval()
    acts = extract(model, seq, fit["position"], device)
    assert acts.isfinite().all(), "non-finite activation"
    check_token_tracking(acts, fit)

    out_dir = study_dir("move_legality") / ("acts_smoke" if args.smoke else "acts")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(acts, out_dir / f"resid_post_{LAYER}.pt")
    print(f"wrote {tuple(acts.shape)} to {out_dir}")


if __name__ == "__main__":
    main()
