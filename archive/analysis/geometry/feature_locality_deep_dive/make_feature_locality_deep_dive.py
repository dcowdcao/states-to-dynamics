"""Per-feature 1x3 heatmaps for the TopK >=0.7 and SpaDE-1e-4 >=0.5 locality buckets.

Self-contained study — writes its PNGs + summary.json into this folder. Reads precomputed
eval tensors from results/ (and/or the linear probe) via spade.evaluation.karvonen_adapter.
Skeleton — port from the original analysis/geometry/feature_locality_deep_dive/.
"""

# def main(): ...   # build figures for this study; save alongside this script
