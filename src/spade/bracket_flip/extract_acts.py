"""Step 2 of the bracket-flip study: OthelloGPT resid_post for the events and the probe set.

Per event two forward passes share the prefix: run(P + m) gives `before` at
position t-1 and `after` at position t, run(P + m') gives `control` at t.
`before` is read off the m-run rather than a third pass over P alone -- attention
is causal so the two agree exactly, which is checked on a sample.

Usage: python -m spade.bracket_flip.extract_acts
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from spade.bracket_flip.events import PROBE_T_MAX, PROBE_T_MIN
from spade.common.config import CTX_LEN, D_MODEL, HF_REVISIONS, MODEL_NAME, provenance
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.study import study_dir, variant_dir

force_ipv4()

N_LAYERS = 8
HOOKS = [f"blocks.{i}.hook_resid_post" for i in range(N_LAYERS)]
CONDITIONS = ("before", "after", "control")
EVENT_BATCH = 128
PROBE_BATCH = 50


def sequences(events: pd.DataFrame, column: str) -> list[list[int]]:
    return [[int(tok) for tok in prefix.split()] + [int(move)]
        for prefix, move in zip(events["prefix"], events[column])]


def extract_events(model, events: pd.DataFrame, device: str) -> dict[str, torch.Tensor]:
    out = {k: torch.empty(len(events), N_LAYERS, D_MODEL) for k in CONDITIONS}
    played, control = sequences(events, "move"), sequences(events, "control_move")
    ts = events["t"].to_numpy()
    for t in np.unique(ts):
        idx = np.flatnonzero(ts == t)
        for start in range(0, len(idx), EVENT_BATCH):
            batch = idx[start:start + EVENT_BATCH]
            tokens = torch.tensor([played[i] for i in batch] + [control[i] for i in batch],
                                  device=device)
            assert tokens.shape[1] == t + 1 and (tokens > 0).all()
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in HOOKS)
            acts = torch.stack([cache[h] for h in HOOKS], dim=1)  # [2b, layer, pos, d_model]
            b = len(batch)
            out["before"][batch] = acts[:b, :, -2].cpu()
            out["after"][batch] = acts[:b, :, -1].cpu()
            out["control"][batch] = acts[b:, :, -1].cpu()
        print(f"t={t}: {len(idx)} events")
    return out


def check_prefix_invariance(model, events: pd.DataFrame, before: torch.Tensor,
                            device: str, n_events: int = 8) -> None:
    """`before` must equal the last position of run(P), which is a separate forward pass."""
    worst = 0.0
    for i in range(n_events):
        prefix = [int(tok) for tok in events["prefix"].iloc[i].split()]
        assert len(prefix) == events["t"].iloc[i]
        _, cache = model.run_with_cache(torch.tensor([prefix], device=device),
                                        names_filter=lambda name: name in HOOKS)
        got = torch.stack([cache[h][0, -1] for h in HOOKS]).cpu()
        worst = max(worst, (got - before[i]).abs().max().item())
    assert worst < 1e-4, f"before != last position of run(P), max abs diff {worst:.2e}"
    print(f"prefix invariance ok over {n_events} events, max abs diff {worst:.2e}")


def extract_probe(model, tokens: torch.Tensor, device: str) -> torch.Tensor:
    n_games, n_pos = tokens.shape[0], PROBE_T_MAX - PROBE_T_MIN
    out = torch.empty(N_LAYERS, n_games, n_pos, D_MODEL)
    for start in range(0, n_games, PROBE_BATCH):
        batch = tokens[start:start + PROBE_BATCH].long().to(device)
        assert (batch > 0).all()
        _, cache = model.run_with_cache(batch, names_filter=lambda name: name in HOOKS)
        for layer, hook in enumerate(HOOKS):
            out[layer, start:start + PROBE_BATCH] = cache[hook][:, PROBE_T_MIN:PROBE_T_MAX].cpu()
        print(f"probe games {min(start + PROBE_BATCH, n_games)}/{n_games}")
    return out.reshape(N_LAYERS, n_games * n_pos, D_MODEL)


def main():
    from transformer_lens import HookedTransformer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="",
                        help="read $DIR/<variant>/events and write its acts there; the probe "
                             "activations are shared and are only written for the main run")
    args = parser.parse_args()

    device = select_device()
    event_dir = variant_dir("bracket", variant=args.variant) / "events"
    probe_dir = study_dir("bracket") / "probe"
    events = pd.read_csv(event_dir / "events.csv")
    assert events["t"].min() >= 5, "position 0 would be read as `before`"

    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device=device).eval()
    assert model.cfg.n_layers == N_LAYERS and model.cfg.d_model == D_MODEL

    acts = extract_events(model, events, device)
    check_prefix_invariance(model, events, acts["before"], device)
    for name, tensor in acts.items():
        assert tensor.shape == (len(events), N_LAYERS, D_MODEL) and tensor.isfinite().all()
        torch.save(tensor, event_dir / f"{name}.pt")
    (event_dir / "acts_manifest.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "hooks": HOOKS,
        "conditions": {"before": "position t-1 of run(P + m)",
                       "after": "position t of run(P + m)",
                       "control": "position t of run(P + m')"},
        "shape": [len(events), N_LAYERS, D_MODEL],
        "dtype": "float32",
        **provenance(),
    }, indent=2))
    print(f"wrote {len(CONDITIONS)} event tensors {tuple(acts['before'].shape)} to {event_dir}")
    if args.variant:
        return

    probe_tokens = torch.load(probe_dir / "tokens.pt", weights_only=True)
    assert probe_tokens.shape[1] == CTX_LEN
    probe_acts = extract_probe(model, probe_tokens, device)
    labels = torch.load(probe_dir / "board_states_relative.pt", weights_only=True)
    assert probe_acts.shape[1] == labels.shape[0], "probe activations and labels misaligned"
    torch.save(probe_acts, probe_dir / "all_layer_activations.pt")
    print(f"wrote probe activations {tuple(probe_acts.shape)} to {probe_dir}")


if __name__ == "__main__":
    main()
