"""R4: behavioral forfeit competence + turn-assignment patching.

Reads the mined minimal pairs (mine_forfeits.py). Behavioral: how much
probability mass the model puts on the true legal set at forfeit positions
(clean: e moves again) vs matched non-forfeit positions (corrupted: d moves).
Causal: patch each layer's attn_out / mlp_out at the final position clean ->
corrupted (single component — see patch_battery.py); the color score
S = logmass(E_diag) - logmass(D_diag) (both sets on the corrupted board,
legal for exactly one color) tracks whether the patched component carries
"d has no move — e moves again".

Usage: python -m spade.rules.causal.forfeit_patch
"""

import pandas as pd
import torch

from spade.common.config import MODEL_NAME
from spade.common.device import select_device
from spade.common.net import force_ipv4
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.rules.causal.patch_battery import PATCH_HOOKS, battery_logprobs

force_ipv4()

BATCH_SIZE = 128


def logmass(logprobs: torch.Tensor, tokens: list[int]) -> float:
    return float(logprobs[tokens].logsumexp(0))


def main():
    device = select_device()
    events = torch.load(study_dir("rules") / "causal" / "forfeit_pairs.pt",
                        weights_only=False)
    print(f"{len(events)} forfeit minimal pairs")

    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    model.eval()

    records = []
    by_len: dict[int, list[int]] = {}
    for j, e in enumerate(events):
        by_len.setdefault(len(e["clean_seq"]), []).append(j)
    for length, idxs in by_len.items():
        for start in range(0, len(idxs), BATCH_SIZE):
            batch = [events[j] for j in idxs[start:start + BATCH_SIZE]]
            tok1 = torch.tensor([e["clean_seq"] for e in batch], device=device)
            tok2 = torch.tensor([e["corr_seq"] for e in batch], device=device)
            logprobs = battery_logprobs(model, tok1, tok2)
            for k, e in enumerate(batch):
                clean_lp, corr_lp = logprobs["clean"][k], logprobs["corr"][k]
                base = {
                    "j": e["j"],
                    "clean_legal_mass": logmass(clean_lp, e["clean_legal"]),
                    "corr_legal_mass": logmass(corr_lp, e["corr_legal"]),
                    "clean_top1_legal": int(clean_lp.argmax()) in e["clean_legal"],
                    "s_clean": logmass(clean_lp, e["e_diag"])
                        - logmass(clean_lp, e["d_diag"]),
                    "s_corr": logmass(corr_lp, e["e_diag"])
                        - logmass(corr_lp, e["d_diag"]),
                    "n_e_diag": len(e["e_diag"]), "n_d_diag": len(e["d_diag"]),
                }
                for key in PATCH_HOOKS:
                    s_patch = logmass(logprobs[key][k], e["e_diag"]) \
                        - logmass(logprobs[key][k], e["d_diag"])
                    records.append({**base, "patch": key, "s_patch": s_patch,
                        "recovery": (s_patch - base["s_corr"])
                            / (base["s_clean"] - base["s_corr"])
                            if base["s_clean"] != base["s_corr"] else float("nan")})
        print(f"length {length}: done")

    out = repo_root() / "analysis" / "rules_causal" / "results"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out / "forfeit_patch.csv", index=False)
    per_event = df[df.patch == "resid0"]
    print(f"behavioral: clean legal mass (log) {per_event.clean_legal_mass.mean():.3f}, "
          f"top1-legal {per_event.clean_top1_legal.mean():.3f}, "
          f"corr legal mass {per_event.corr_legal_mass.mean():.3f}")
    print(df.groupby("patch")[["s_patch", "recovery"]].mean())
    print(f"wrote {out / 'forfeit_patch.csv'}")


if __name__ == "__main__":
    main()
