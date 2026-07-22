"""Registry of trained SAE checkpoints across seeds (seed 42 in-repo + 43-46 on netscratch).

Reuses BASELINE_CONFIGS / SWEEP_CONFIGS / config_key from spade.multiseed.aggregate_seeds
so the checkpoint enumeration and the figure aggregation stay in lockstep. Verifies each
entry exists on construction (fail loudly).

Entrypoint (debug): `python -m spade.legality.sae_registry`  prints the registry.
"""

# @dataclass
# class SAEEntry: type: str; seed: int; checkpoint_dir: Path
# def entries() -> list[SAEEntry]: ...   # seed 42 -> checkpoints/*, seeds 43-46 -> $SPADE_MULTISEED_DIR
