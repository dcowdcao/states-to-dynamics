"""2D PCA + UMAP of per-square (60 non-center) feature directions per reconstructor.

Self-contained study — writes its PNGs + summary.json into this folder. Reads precomputed
eval tensors from results/ (and/or the linear probe) via spade.evaluation.karvonen_adapter.
Skeleton — port from the original analysis/geometry/embedding_pca_umap/.
"""

# def main(): ...   # build figures for this study; save alongside this script
