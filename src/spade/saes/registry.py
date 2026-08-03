"""Registry of trained SAE checkpoints across seeds (seed 42 in-repo + 43-46 on netscratch).

Reuses BASELINE_CONFIGS / SWEEP_CONFIGS / config_key from spade.multiseed.aggregate_seeds
so the checkpoint enumeration and the figure aggregation stay in lockstep. Verifies each
entry exists on construction (fail loudly).

This is the most-depended-on module in the tree: the feature viewer, the rules program,
manifold-capture, bracket-flip, straightness and violation-divergence all resolve their
checkpoints through it. It lives under saes/ rather than inside a study for that reason.
`seed42_entries()` is the public in-repo-only subset — the viewer uses it so it never
touches netscratch.

Entrypoint (debug): `python -m spade.saes.registry` prints the registry.
"""

# @dataclass
# class SAEEntry: config: str; seed: int; checkpoint_dir: Path
# def entries() -> list[SAEEntry]: ...          # seed 42 -> checkpoints/*, 43-46 -> study_dir("multiseed")
# def seed42_entries() -> list[SAEEntry]: ...   # in-repo only; no netscratch dependency

# Checkpoint dirs are named by experiment (checkpoints/{main,batch_topk,cosine_spade,
# spade_sweep,spade_revival}), NOT by the source repo's autoencoders*/ names. When a dir
# moves, standardization_params.json moves with it at the save-dir level: the adapter
# searches the checkpoint dir and its parent, and the file lives at the parent.
