"""Step 2 of the legality-probe study: layer-5 activations for the fit-set positions.

One batched run_with_cache pass at blocks.5.hook_resid_post, sliced to fit_set.POSITIONS
so rows line up with labels.pt: [n_positions, 512] fp32.

The pass also checks the alignment it depends on. The model's next-move prediction at
position p is scored against the label at that square; OthelloGPT plays legally almost
always, so an off-by-one between activations and labels collapses this number.

Usage: python -m spade.legality_probe.extract_acts
"""

import numpy as np
import torch

from spade.common.config import CTX_LEN, D_MODEL, HF_REVISIONS, HOOK_POINT, MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.study import study_dir
from spade.legality_probe.fit_set import POSITIONS

force_ipv4()

from circuits.othello_engine_utils import stoi_indices

BATCH_SIZE = 50
MIN_LEGAL_PREDICTIONS = 0.9


def main() -> None:
    from transformer_lens import HookedTransformer

    fit_dir = study_dir("legality") / "fit_set"
    tokens = torch.load(fit_dir / "tokens.pt", weights_only=True)
    labels = torch.load(fit_dir / "labels.pt", weights_only=True)
    n_games = tokens.shape[0]

    device = select_device()
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device=device).eval()

    acts = torch.empty(n_games, len(POSITIONS), D_MODEL)
    predicted = torch.empty(n_games, len(POSITIONS), dtype=torch.long)
    for start in range(0, n_games, BATCH_SIZE):
        batch = tokens[start:start + BATCH_SIZE].long().to(device)
        assert batch.shape[1] == CTX_LEN
        logits, cache = model.run_with_cache(batch, names_filter=HOOK_POINT)
        window = slice(POSITIONS.start, POSITIONS.stop)
        acts[start:start + BATCH_SIZE] = cache[HOOK_POINT][:, window].cpu()
        # vocab entry v+1 is board square stoi_indices[v]; the pass token cannot be a move
        predicted[start:start + BATCH_SIZE] = torch.tensor(
            np.asarray(stoi_indices))[logits[:, window, 1:].argmax(-1).cpu()]
        print(f"extracted {min(start + BATCH_SIZE, n_games)}/{n_games} games", flush=True)

    acts = acts.reshape(-1, D_MODEL).contiguous()
    legal = (labels.gather(1, predicted.reshape(-1, 1)) == 0).float().mean().item()
    assert legal > MIN_LEGAL_PREDICTIONS, \
        f"model's predicted move is legal at only {legal:.3f} of positions: rows are misaligned"

    out_dir = study_dir("legality") / "acts"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(acts, out_dir / "resid_post_5.pt")
    print(f"wrote {acts.shape[0]} rows to {out_dir / 'resid_post_5.pt'} "
          f"(model's predicted move legal at {legal:.4f} of positions)")


if __name__ == "__main__":
    main()
