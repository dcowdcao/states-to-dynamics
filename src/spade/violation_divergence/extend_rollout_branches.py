"""Roll out the source games extend_branches appended, and append those rows too.

rollout_branches.py resamples the whole tree. Its uniforms are keyed on
(game_id, k) and would not move, but `generate` batches by t0 and a changed batch
composition changes the logits in their last bits, so rows that must stay fixed are
only probably fixed. This samples the appended source rows alone and concatenates,
so no existing row is recomputed and the per-(game_id, k) seeding gives the new
games streams that no existing row ever drew from.

`row = source_row * K + k` survives because the new source rows are at the end.

Usage: python -m spade.violation_divergence.extend_rollout_branches
"""

import json

import torch

from spade.common.config import HF_REVISIONS, MODEL_NAME
from spade.common.device import select_device
from spade.common.study import study_dir
from spade.violation_divergence.extend_branches import digest
from spade.violation_divergence.rollout_branches import (
    K_ROLLOUTS, ROLLOUT_DTYPES, SEED, generate, uniforms, verify)


def main():
    from transformer_lens import HookedTransformer

    base = study_dir("divergence")
    source = torch.load(base / "branches" / "branches.pt", weights_only=True)
    out_dir = base / "rollout" / "branches"
    saved = torch.load(out_dir / "branches.pt", weights_only=True)

    n_done = len(saved["t0"]) // K_ROLLOUTS
    assert torch.equal(saved["game_id"],
                       source["game_id"][:n_done].repeat_interleave(K_ROLLOUTS)), \
        "the rollout tree is not the head of branches.pt"
    new = {key: column[n_done:] for key, column in source.items()}
    assert len(new["t0"]), "every source branch already has its rollouts"
    print(f"{len(saved['t0'])} rollout rows on disk over {n_done} source branches; "
          f"sampling {len(new['t0'])} new branches x {K_ROLLOUTS}")
    before = digest(saved, len(saved["t0"]))

    items = {key: column.repeat_interleave(K_ROLLOUTS, 0) for key, column in new.items()}
    items["rollout"] = (torch.arange(len(items["t0"])) % K_ROLLOUTS).to(ROLLOUT_DTYPES["rollout"])

    device = select_device()
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, revision=HF_REVISIONS[MODEL_NAME], device=device).eval()
    u = uniforms(items["game_id"], items["rollout"], SEED)
    items["tokens"] = generate(model, items, u, device)

    full = {key: torch.cat([saved[key], items[key]]) for key in ROLLOUT_DTYPES}
    assert all(full[key].dtype == dtype for key, dtype in ROLLOUT_DTYPES.items()), \
        "a column changed dtype"
    assert digest(full, len(saved["t0"])) == before, "an existing rollout row changed"
    verify(full, source)

    path = out_dir / "branches.pt"
    torch.save(full, path)
    assert digest(torch.load(path, weights_only=True), len(saved["t0"])) == before, \
        "an existing rollout row changed on the round trip"
    print(f"existing {len(saved['t0'])} rows byte-identical, sha256 {before}")

    manifest = json.loads((out_dir / "manifest.json").read_text())
    manifest["n_source_branches"] = len(source["t0"])
    manifest["n_rows"] = len(full["t0"])
    manifest["appended"].append({
        "n_source_branches": len(new["t0"]),
        "n_rows": len(items["t0"]),
        "first_row": len(saved["t0"]),
        "reason": "the source games extend_branches appended; sampled on their own "
                  "(game_id, k) streams, with no existing row recomputed",
    })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(full['t0'])} rollout rows to {out_dir}")


if __name__ == "__main__":
    main()
