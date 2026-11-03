"""Settles what `circuits.evaluation.evaluate` actually measures.

`get_eval_results.get_evals` calls it with the default `n_batches=1`, so every
reconstruction column in the published CSV comes from ONE buffer batch of
`config.batch_size` activation vectors. This script rebuilds that buffer on CPU
for one released SAE and compares:

  published value  vs  whole-buffer value  vs  25-sample sampling spread

for both the centered (his) and uncentered (ours) variance formulas.

Run: python karvonen_repro/probe_eval_batch.py
"""

import argparse, json, sys
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer

from spade.common.config import HF_REVISIONS

REPO = "adamkarvonen/othello_saes"
MODEL = "Baidicoot/Othello-GPT-Transformer-Lens"
DATASET = "adamkarvonen/othello_45MB_games"

CTX_LEN = 59
BATCH_SIZE = 25  # Config.batch_size, == buffer refresh_batch_size and out_batch_size
N_CTXS = 512  # min(512, eval_results_n_inputs)


def buffer_activations(layer: int, device: str) -> torch.Tensor:
    """The activations NNsightActivationBuffer holds after its first refresh."""
    games = load_dataset(DATASET, split="train", revision=HF_REVISIONS[DATASET])
    model = HookedTransformer.from_pretrained(MODEL, revision=HF_REVISIONS[MODEL], device=device)
    target = N_CTXS * CTX_LEN

    chunks, filled, cursor = [], 0, 0
    while filled < target:
        toks = torch.tensor(
            [games[i]["tokens"][:CTX_LEN] for i in range(cursor, cursor + BATCH_SIZE)],
            device=device,
        )
        cursor += BATCH_SIZE
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=f"blocks.{layer}.hook_resid_post")
        acts = cache[f"blocks.{layer}.hook_resid_post"].reshape(-1, model.cfg.d_model)
        chunks.append(acts)
        filled += acts.shape[0]
    print(f"buffer: {filled} activations from {cursor} games", file=sys.stderr)
    return torch.cat(chunks)


def metrics(x: torch.Tensor, ae) -> dict[str, float]:
    x_hat, f = ae(x, output_features=True)
    total = torch.var(x, dim=0).sum()
    resid = torch.var(x - x_hat, dim=0).sum()
    ss_total = (x**2).sum()
    ss_resid = ((x - x_hat) ** 2).sum()
    return {
        "l0": (f != 0).float().sum(dim=-1).mean().item(),
        "l1_loss": f.norm(p=1, dim=-1).mean().item(),
        "l2_loss": torch.linalg.norm(x - x_hat, dim=-1).mean().item(),
        "frac_variance_explained": (1 - resid / total).item(),
        "fve_uncentered": (1 - ss_resid / ss_total).item(),
        "cossim": (
            torch.nn.functional.normalize(x, dim=-1) * torch.nn.functional.normalize(x_hat, dim=-1)
        )
        .sum(dim=-1)
        .mean()
        .item(),
        "l2_ratio": (torch.linalg.norm(x_hat, dim=-1) / torch.linalg.norm(x, dim=-1)).mean().item(),
        "frac_alive": (f.sum(dim=0) != 0).float().sum().item() / ae.dict_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainers", nargs="+", default=["trainer_0"])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external/Karvonen_Research"))
    from circuits.dictionary_learning.utils import load_dictionary
    import pandas as pd

    published = pd.read_csv(hf_hub_download(REPO, "matryoshka_othello/results.csv"))
    acts = buffer_activations(5, args.device)

    for trainer in args.trainers:
        sub = f"matryoshka_othello/{trainer}"
        ae_dir = Path(hf_hub_download(REPO, f"{sub}/ae.pt")).parent
        hf_hub_download(REPO, f"{sub}/config.json")
        cfg = json.loads((ae_dir / "config.json").read_text())
        assert cfg["trainer"]["layer"] == 5
        ae = load_dictionary(str(ae_dir), args.device)[0].to(args.device)

        with torch.no_grad():
            whole = metrics(acts, ae)
            draws = []
            for seed in range(args.draws):
                g = torch.Generator().manual_seed(seed)
                idx = torch.randperm(acts.shape[0], generator=g)[:BATCH_SIZE]
                draws.append(metrics(acts[idx], ae))

        row = published[published["autoencoder_path"].str.contains(f"/{trainer}/")].iloc[0]
        print(f"\n{sub}  k={cfg['trainer']['k']}  dict_size={cfg['trainer']['dict_size']}")
        print(f"{'metric':<26} {'published':>12} {'whole buffer':>14} {'25-draw mean':>14} {'25-draw sd':>12} {'25-draw range':>26}")
        for k in whole:
            vals = torch.tensor([d[k] for d in draws])
            pub = row[k] if k in row.index else float("nan")
            print(
                f"{k:<26} {pub:>12.6f} {whole[k]:>14.6f} {vals.mean():>14.6f} {vals.std():>12.6f}"
                f" {f'[{vals.min():.4f}, {vals.max():.4f}]':>26}"
            )


if __name__ == "__main__":
    main()
