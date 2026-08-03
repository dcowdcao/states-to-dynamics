"""Does the bracketing-flip signature survive an SAE bottleneck? (descriptive).

Fits the bilinear TPR of `tpr_probe` on each space's *latent* code rather than the
raw layer-5 residual, then re-reads the flip through it. Nine spaces: `raw`
(standardized residual, d=512) plus the eight seed-42 baseline SAE families from
`spade.spaces.representative_spaces()` (d=8192). No causal edits — every number
here is a read-out.

The fit path is `tpr_probe`'s, unchanged in behaviour: `fit_latent` mirrors the
committed `_fit` (same r=52, f=2, wd, epochs, seed) but infers the input dimension
from the latent code and keeps the probe on the encode device, so an 8192-wide
code trains without touching `tpr_probe`'s 512 default.

Three families of number per space, in rising order of accuracy-robustness:
  1. `acc_rel_L5`   — board decode, held-out 12k positions. Fully accuracy.
  2. `p_s_unchanged` / `_control` / gap — the double dissociation. p_s_unchanged
     is ~acc**2 (analysis/bracket_flip/reports/ROBUSTNESS.md), so it falls with a
     lossy SAE; the *gap* against the matched control is what carries the flip.
  3. `cos_flip` vs `cos_null` — displacement (after - control, strict) aligned
     with the TPR-predicted flip direction of s vs the unflipped-disc null.
     Scale-free, so this is the reading a SpaDE simplex code can still be judged
     on (analysis/bracket_flip/reports/SAE_ARM.md, analysis/rules_manifolds/README.md).

Usage: python -m spade.bracket_flip.sae_arm [--spaces raw,TopKTrainer-othello]
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot

from spade.bracket_flip import act_geometry as ag
from spade.bracket_flip import tpr_probe
from spade.common.device import select_device
from spade.common.config import LAYER
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.multiseed.aggregate_seeds import BASELINE_CONFIGS
from spade.spaces import batched_apply, load_space, representative_spaces

THEIRS, MINE = tpr_probe.CLASSES.index("theirs"), tpr_probe.CLASSES.index("mine")
KEEP = {"raw", *BASELINE_CONFIGS}  # raw + the 8 *Trainer-othello baselines; skip the controls and the SpaDE-lam sweep


def spaces() -> list[tuple]:
    return [(n, e) for n, e in representative_spaces() if n in KEEP]


def probe_set(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = study_dir("bracket") / "probe"
    acts = torch.load(d / "all_layer_activations.pt", map_location="cpu", weights_only=True)[LAYER]
    labels = torch.load(d / "board_states_relative.pt", map_location="cpu", weights_only=True).long()
    return acts.to(device), labels


def events() -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    d = study_dir("bracket") / "events"
    table = pd.read_csv(d / "events.csv")
    raw = {c: torch.load(d / f"{c}.pt", map_location="cpu", weights_only=True)[:, LAYER]
           for c in ("before", "after", "control")}
    codes = np.array([[int(ch) for ch in row] for row in table["control_squares"].astype(str)])
    assert (codes[np.arange(len(table)), table["square_s"]] == ag.CODE["flipped"]).all()
    return table, raw, codes, (table["control_kind"] == "strict").to_numpy()


def fit_latent(acts: torch.Tensor, labels: torch.Tensor, device: str) -> tpr_probe.TPRProbe:
    """`tpr_probe._fit` with the input dim inferred from the code and the probe on `device`."""
    tpr_probe.seed_all(tpr_probe.SEED)
    acts = acts.to(device)  # batched_apply returns on CPU; the batch indices below live on device
    module = tpr_probe.TensorProductBoardProbe(
        d_model=acts.shape[-1], role_dim=tpr_probe.ROLE_DIM, filler_dim=tpr_probe.FILLER_DIM).to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=tpr_probe.LR,
                                  betas=tpr_probe.BETAS, weight_decay=tpr_probe.WEIGHT_DECAY)
    targets = one_hot(labels.to(device), len(tpr_probe.CLASSES)).float().unsqueeze(1)
    mask = tpr_probe.build_square_selection_mask(exclude_center_squares=False, device=device)
    for _epoch in range(tpr_probe.EPOCHS):
        for batch in torch.randperm(acts.shape[0], device=device).split(tpr_probe.BATCH_SIZE):
            loss = tpr_probe.compute_tpr_probe_loss(
                module(acts[batch].unsqueeze(1)), targets[batch], square_mask=mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return tpr_probe.TPRProbe(module.requires_grad_(False).to("cpu"), LAYER)


def retain_directions(probe: tpr_probe.TPRProbe) -> np.ndarray:
    """[64, d] unit rows: the mine->theirs edit direction, one per square (act_geometry's TPR null)."""
    rows = np.stack([probe.edit_direction(q, MINE, THEIRS).numpy() for q in range(64)])
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def decode_class(probe: tpr_probe.TPRProbe, code: torch.Tensor, square_s: np.ndarray) -> np.ndarray:
    return probe.decode(code).argmax(-1).numpy()[np.arange(len(square_s)), square_s]


def measure(name: str, entry, device: str) -> dict:
    encode = load_space(name if entry is None else "sae", entry, device)
    latent = {c: batched_apply(encode, raw[c], device) for c in ("before", "after", "control")}

    acts, labels = PROBE
    train_code = batched_apply(encode, acts[:tpr_probe.N_TRAIN], device)
    probe = fit_latent(train_code, labels[:tpr_probe.N_TRAIN], device)
    z = batched_apply(encode, acts[tpr_probe.N_TRAIN:], device)
    acc = tpr_probe.accuracy(probe, z, labels[tpr_probe.N_TRAIN:])

    ss = TABLE["square_s"].to_numpy()
    before = decode_class(probe, latent["before"], ss)
    after = decode_class(probe, latent["after"], ss)
    control = decode_class(probe, latent["control"], ss)
    p_unchanged = float((before == after).mean())
    p_control = float((before == control).mean())

    dirs = retain_directions(probe)
    delta = (latent["after"] - latent["control"]).double().numpy()[STRICT]
    cos = ag.cos_retain(delta, dirs, CODES[STRICT])
    flip = cos[np.arange(STRICT.sum()), ss[STRICT]]
    null = cos[np.isin(CODES[STRICT], ag.NULL_CODES)]

    torch.save({**latent, "square_s": ss}, CACHE / f"events_latent_{name}.pt")
    torch.save(probe.module.state_dict(), CACHE / f"tpr_{name}.pt")
    return {
        "space": name, "is_spade": name in ("SpaDETrainer-othello", "CosineSpaDETrainer-othello"),
        "d_latent": latent["before"].shape[1], "acc_rel_L5": round(acc, 4),
        "acc_sq_floor": round(acc ** 2, 4),
        "p_s_unchanged": round(p_unchanged, 4), "p_s_unchanged_control": round(p_control, 4),
        "dissociation_gap": round(p_unchanged - p_control, 4),
        "cos_flip_median": round(float(np.median(flip)), 4),
        "cos_null_median": round(float(np.median(null)), 4),
        "cos_gap": round(float(np.median(flip) - np.median(null)), 4),
        "p_flip_positive": round(float((flip > 0).mean()), 4),
        "n_strict": int(STRICT.sum()), "n_null": int(len(null)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default="", help="comma-separated filter")
    args = parser.parse_args()

    device = select_device()
    global TABLE, raw, CODES, STRICT, PROBE, CACHE
    CACHE = study_dir("bracket") / "sae_arm"
    CACHE.mkdir(parents=True, exist_ok=True)
    TABLE, raw, CODES, STRICT = events()
    PROBE = probe_set(device)

    chosen = spaces()
    if args.spaces:
        keep = set(args.spaces.split(","))
        chosen = [s for s in chosen if s[0] in keep]

    rows = []
    for name, entry in chosen:
        row_path = CACHE / f"row_{name}.json"
        if row_path.exists():
            rows.append(json.loads(row_path.read_text()))
            print(f"skip {name} (cached)", flush=True)
            continue
        row = measure(name, entry, device)
        row_path.write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(f"{name:32} acc {row['acc_rel_L5']:.4f}  gap {row['dissociation_gap']:+.4f}  "
              f"cos_flip {row['cos_flip_median']:+.3f}  cos_null {row['cos_null_median']:+.3f}  "
              f"p+ {row['p_flip_positive']:.3f}", flush=True)

    out = repo_root() / "analysis" / "bracket_flip" / "results" / "sae_arm.csv"
    order = {n: i for i, n in enumerate(["raw", *sorted(BASELINE_CONFIGS)])}
    pd.DataFrame(rows).sort_values("space", key=lambda s: s.map(order)).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
