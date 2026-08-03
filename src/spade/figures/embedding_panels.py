"""Shared renderer for PCA/Isomap class clouds (FIGURE_GUIDELINES §4.6).

Draws the cloud once in shuffled order — per-class overplotting encodes draw order as if
it were density — then a shape-coded centroid and 50% probability contour per class, with
the per-component variance ratio on each axis. A nuisance-variable panel colored by move
number ships with every embedding figure even when it shows nothing, because that is the
confound a reader will otherwise suspect.

Panels from separate fits are not comparable point-for-point; the caption says so.
"""

# def class_panel(ax, coords, labels, style): ...
# def confound_panel(ax, coords, t): ...
