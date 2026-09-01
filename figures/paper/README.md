# The paper's main-text figures, under the names `paper.tex` uses

Staged copies, not sources. Edit a figure in the study that owns it, re-render there, then
run `python figures/paper/stage.py`. Anything edited here is overwritten on the next run.

| paper | file here | owned by |
|---|---|---|
| Fig. 1 | `rules_and_generation` | `src/spade/figures/figure_rules_and_generation.py` |
| Fig. 2 | `flanking` | `analysis/bracket_flip/make_figures.py` |
| Fig. 3 | `probe_geometry_and_detection` | `figures/probe_directions_and_detection/make_figure.py` |
| Fig. 4 | `detector_example` | `analysis/violation_divergence/illegal_move_detector/figure_overview.py` |
| Fig. 5 | `detector_SAE` | `analysis/violation_divergence/illegal_move_detector/figure_sae_overview.py` |
| Fig. 7 | `karvonen_sae_sweep` | `src/spade/l0_operating_point/combine_figure.py` |

Each is a `.png` and a `.pdf`; `paper.tex` includes the `.png` today.

Every source name differs from the name the paper uses, and three of the six differ enough
that a pattern match gets them wrong -- `fig1_flip_and_basis` is Fig. 2, and
`figure4_othello_with_spade` is Fig. 7. That is what `stage.py`'s `SOURCES` is for.

**The appendix is not staged here.** Figs. 6 and 8-16 are still referenced straight out of
their studies, and all eight of the appendix's grid figures are the `_l0_matched` twins,
which the sources' filenames record and their captions do not.
