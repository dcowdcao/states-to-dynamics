"""The detector you get without a probe: the model's own probability for the move played.

`s[i, t] = 1 - P_model(tokens[i, t] | tokens[i, :t])`, read off the next-move logits already
saved beside the branch activations. It needs no probe, no fit set and no labels, it is in
[0, 1] like the probe outputs, and it is rank-equivalent to the surprise -log P -- so the
same localization and detection statistics apply to it unchanged.

This is the control the probe has to beat. `spade.generation_legality` measures that
OthelloGPT puts 0.9982 of its next-move mass on legal squares, so a played illegal move is
already improbable under the model, and a probe that merely recovers that has read the
output distribution rather than the representation. What the score cannot separate is
"illegal" from "legal but unlikely" -- which is exactly why the legal control branch, a
*different* legal move at the same position, is the arm it is scored against.

Usage: python -m spade.move_legality.model_baseline
"""

import torch

from spade.common.config import CTX_LEN
from spade.common.study import study_dir

NAME = "model_prior"
BATCH = 2048


def score(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """[n, 59]: 1 - P(the token at t), predicted from the position before it.

    Position 0 has no predictor and is the attention sink, which nothing reads; it is
    filled with 0 rather than left undefined so the tensor has the read-out's shape.

    float64 throughout, and it matters: an illegal move's P is often below 1e-8, where
    float32 rounds 1 - P to exactly 1.0 and ties together most of the illegal arm. The
    probe outputs never come near that -- they are softmaxes of bounded logits -- so this
    is the one score in the study that needs the extra precision.
    """
    out = torch.zeros(len(tokens), CTX_LEN, dtype=torch.float64)
    for start in range(0, len(tokens), BATCH):
        rows = slice(start, start + BATCH)
        probability = logits[rows, :-1].double().softmax(-1)
        played = tokens[rows, 1:].long().unsqueeze(-1)
        out[rows, 1:] = 1.0 - probability.gather(2, played).squeeze(-1)
    return out


def main() -> None:
    divergence = study_dir("divergence")
    meta = torch.load(divergence / "branches" / "branches.pt", map_location="cpu",
                      weights_only=True)
    logits = torch.load(divergence / "acts" / "logits.pt", map_location="cpu",
                        weights_only=True)
    assert logits.shape[:2] == (len(meta["t0"]), CTX_LEN), logits.shape

    out = score(logits, meta["tokens"])
    assert out.isfinite().all() and out.min() >= 0 and out.max() <= 1, "score left [0, 1]"

    # The one check that says the tensor is aligned: an illegal move at t0 must be less
    # likely under the model than the legal move the game actually played there.
    reference = (meta["category"] == 0) & (meta["is_control"] == 0)
    at_t0 = out.gather(1, meta["t0"].unsqueeze(1)).squeeze(1)
    illegal, legal = at_t0[meta["category"] != 0].mean(), at_t0[reference].mean()
    assert illegal > legal, f"illegal {illegal:.4f} is not less likely than reference {legal:.4f}"

    path = study_dir("move_legality") / "readout" / f"p_illegal_{NAME}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"wrote {tuple(out.shape)} to {path}\n"
          f"mean 1 - P(played) at t0: illegal {illegal:.4f}, reference {legal:.4f}")


if __name__ == "__main__":
    main()
