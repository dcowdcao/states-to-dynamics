"""Legality geometry: legal vs illegal separation in each SAE space; consumes spade.legality outputs.

Self-contained study — writes its PNGs + summary.json into this folder. Reads precomputed
eval tensors from results/ (and/or the linear probe) via spade.evaluation.karvonen_adapter.
Skeleton — port from the original analysis/geometry/legal_vs_illegal/.
"""

# def main(): ...   # build figures for this study; save alongside this script
