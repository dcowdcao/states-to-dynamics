"""Shared trajectory panels: depth profiles, event-locked curves, KDE densities.

Event-locked panels put a thin dashed rule at the event position behind the data, shade
and name the baseline window numerically, and shade any region that is zero by
construction rather than leaving a reader to infer it. Distributions over three or more
classes are KDE curves, not step histograms, with bootstrap bands computed at a fixed
full-sample bandwidth.
"""

# def profile_panel(ax, depth, curves, ci): ...
# def event_locked_panel(ax, offsets, curves, baseline_window): ...
# def kde_panel(ax, samples_by_class, bandwidth): ...
