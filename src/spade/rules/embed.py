"""Step 3: embed the legal/illegal activation set per rule, for the manifold figures.

2D PCA and Isomap over every representation space and view, in both raw and per-t-centered
variants. Centered is the main figure: t is the dominant confound, and an uncentered plot
mostly shows move number. Isomap runs on a shared category-balanced subsample after a
PCA-50 pre-reduction, so panels stay comparable across spaces.

The generic half of what this file used to be — load_space, representative_spaces,
batched_apply, center_by_t — now lives in spade.spaces, because manifold_capture and
bracket_flip import it too. What stays here is the rules-specific run.

Coordinates are committed as float16 to analysis/rules_manifolds/embeddings/.
"""

# def main(): ...   # for each space x view x {raw, centered}: fit, save coords + evr
