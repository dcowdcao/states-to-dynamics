"""3x3 board-F1 heatmaps of each SAE's sharpest / min-locality / max-locality feature.

Self-contained study — writes its PNGs + summary.json into this folder. Reads precomputed
eval tensors from results/ (and/or the linear probe) via spade.evaluation.karvonen_adapter.
Skeleton — port from the original analysis/geometry/feature_locality_heatmaps/.
"""

# def main(): ...   # build figures for this study; save alongside this script
