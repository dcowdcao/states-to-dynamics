"""Registry of every usable trained SAE checkpoint for the rules study.

65 entries. Seed 42 has weights for the 8 baseline configs plus the revived
lam1e-03-gam1e-04 sweep point; the other in-repo sweep dirs for the 14-config
set hold config.json without ae.pt (the two sweep dirs that do have weights
are lam1e-04 points outside the 14-config set). Seeds 43-46 have all 14
configs on netscratch, so the 6 sweep configs run with n_seeds=4.

Every entry is verified on construction: ae.pt exists and the checkpoint's
own wandb_name maps to the expected config via config_key().

The SpaDE gamma=1e-2 seed-42 entry ("SpaDETrainer-othello") is degenerate and is
excluded downstream in spade.spaces, not here — this registry lists what exists.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.multiseed.aggregate_seeds import BASELINE_CONFIGS, SWEEP_CONFIGS, config_key

NEW_SEEDS = [43, 44, 45, 46]

_SEED42_DIRS = {
    "PAnnealTrainer-othello": "checkpoints/main/trainer_0",
    "StandardTrainer-othello": "checkpoints/main/trainer_1",
    "GatedSAETrainer-othello": "checkpoints/main/trainer_2",
    "TopKTrainer-othello": "checkpoints/main/trainer_3",
    "GatedAnnealTrainer-othello": "checkpoints/main/trainer_4",
    "SpaDETrainer-othello": "checkpoints/main/trainer_5",
    "BatchTopKTrainer-othello": "checkpoints/batch_topk/trainer_0",
    "CosineSpaDETrainer-othello": "checkpoints/cosine_spade/trainer_0",
    "SpaDE-lam1e-03-gam1e-04": "checkpoints/spade_revival/trainer_0",
}


@dataclass(frozen=True)
class SAEEntry:
    config: str
    seed: int
    ckpt_dir: Path

    @property
    def name(self) -> str:
        return f"{self.config}_seed{self.seed}"

    @property
    def standardize(self) -> bool:
        """Every checkpoint here was trained on scale*(x - mean). The field exists so a
        roster can mix these with `l0_operating_point.checkpoints.Matched`, whose
        Karvonen picks were trained on the raw residual stream, and one encoder reads
        the rule off the entry instead of the caller knowing which family it holds."""
        return True


def _verify(entry: SAEEntry) -> SAEEntry:
    assert (entry.ckpt_dir / "ae.pt").exists(), f"missing ae.pt in {entry.ckpt_dir}"
    config = json.loads((entry.ckpt_dir / "config.json").read_text())
    wandb_name = config.get("trainer", config)["wandb_name"]
    assert config_key(wandb_name) == entry.config, \
        f"{entry.ckpt_dir}: wandb_name {wandb_name!r} != expected {entry.config!r}"
    return entry


def seed42_entries() -> list[SAEEntry]:
    """The 8 in-repo baseline checkpoints — no netscratch dependency."""
    root = repo_root()
    return [_verify(SAEEntry(cfg, 42, root / rel)) for cfg, rel in _SEED42_DIRS.items()]


def entries() -> list[SAEEntry]:
    multiseed_dir = study_dir("multiseed")
    out = [SAEEntry(cfg, seed,
                     multiseed_dir / f"seed_{seed}" / "autoencoders" / f"trainer_{i}")
        for seed in NEW_SEEDS
        for i, cfg in enumerate(BASELINE_CONFIGS + SWEEP_CONFIGS)]
    return seed42_entries() + [_verify(e) for e in out]


if __name__ == "__main__":
    found = entries()
    print(f"{len(found)} checkpoints verified")
    for e in found:
        print(f"  {e.name:45} {e.ckpt_dir}")
