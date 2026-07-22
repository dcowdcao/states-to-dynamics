"""Per-feature locality histograms per SAE (bimodal baselines vs unimodal Standard/SpaDE).

Self-contained study — writes its PNGs + summary.json into this folder. Reads precomputed
eval tensors from results/ (and/or the linear probe) via spade.evaluation.karvonen_adapter.
Skeleton — port from the original analysis/geometry/feature_locality_distributions/.
"""

# def main(): ...   # build figures for this study; save alongside this script
