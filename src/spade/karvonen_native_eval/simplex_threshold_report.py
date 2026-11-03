"""Does Karvonen's fraction-of-max binarization say anything about a simplex code?

His classifier stage binarizes feature j at `frac * max_j`, with max_j taken per feature
over the eval set. SpaDE codes are simplex-valued (f >= 0, sum_j f_j = 1), so a feature's
activation measures its *share* of a token's mass, not evidence for it in isolation. This
reports the quantities that decide whether the thresholds discriminate: the spread of
per-feature max, how much of the code mass each threshold retains, and how the support
shrinks as the fraction rises.

    python -m spade.karvonen_native_eval.simplex_threshold_report \
        --checkpoint_dir $SPADE_CHECKPOINTS/cosine_spade/trainer_0
"""

import argparse
import os

import torch as t

from spade.common.study import study_dir
from spade.sae_loader.load_sae import load_sae

THRESHOLD_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEFAULT_ACTS = study_dir("multiseed") / "data" / "activations"


def encode_all(ae, acts, device, batch=4096):
    out = []
    with t.no_grad():
        for i in range(0, acts.shape[0], batch):
            out.append(ae.encode(acts[i:i + batch].to(device)).cpu())
    return t.cat(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--activations", default=os.environ.get("SPADE_EVAL_ACTIVATIONS", DEFAULT_ACTS))
    p.add_argument("--n_samples", type=int, default=200_000)
    p.add_argument("--device", default="cuda" if t.cuda.is_available() else "cpu")
    args = p.parse_args()

    ae, meta = load_sae(args.checkpoint_dir, device=args.device)
    acts = t.load(os.path.join(args.activations, "test_activations.pt"),
                  map_location="cpu", weights_only=True, mmap=True)[:args.n_samples]
    f = encode_all(ae, acts, args.device)
    N, F = f.shape

    support = (f > 0)
    l0 = support.float().sum(-1)
    max_act = f.max(0).values
    alive = max_act > 0

    print(f"{meta.dict_class}  {args.checkpoint_dir}")
    print(f"N={N}  F={F}  L0={l0.mean():.1f} +- {l0.std():.1f}  (min {int(l0.min())}, max {int(l0.max())})")
    print(f"code mass sum: {f.sum(-1).mean():.6f}  (simplex => 1)")
    print(f"alive features: {int(alive.sum())}/{F} ({alive.float().mean():.1%}); "
          f"dead features get threshold 0 and predict nothing at every fraction")
    q = t.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    print(f"per-feature max_act quantiles (alive only) {q.tolist()}:")
    print("   " + "  ".join(f"{v:.4f}" for v in max_act[alive].quantile(q)))
    print(f"mean activation of an active feature = 1/L0 = {1 / l0.mean():.5f}")

    print("\nfrac   mean#active   frac of code mass kept   features ever above")
    n_mid = t.zeros(N)
    chunk = 8192
    for frac in THRESHOLD_FRACS:
        thresh = frac * max_act
        n_active, mass, ever = t.zeros(N), t.zeros(N), t.zeros(F, dtype=t.bool)
        for i in range(0, N, chunk):
            above = f[i:i + chunk] > thresh
            n_active[i:i + chunk] = above.sum(-1)
            mass[i:i + chunk] = (f[i:i + chunk] * above).sum(-1)
            ever |= above.any(0)
        if frac == 0.5:
            n_mid = n_active.clone()
        print(f"{frac:4.1f}   {n_active.mean():10.2f}   {mass.mean():22.4f}   {ever.float().mean():18.1%}")

    # A feature's activation is its share of the token, so the same evidence reads lower on a
    # contested token. A strong negative correlation means the threshold sweep is partly a
    # sweep over how many atoms compete, not over evidence for the atom.
    r = t.corrcoef(t.stack([l0, n_mid]))[0, 1]
    print(f"\ncorr(token L0, #features above 0.5*max on that token) = {r:+.3f}")


if __name__ == "__main__":
    main()
