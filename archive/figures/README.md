# figures/

Cross-study and headline figures only. Per-study outputs live with their study under
`analysis/<study>/figures/`.

Subdirectories are by experiment (`cosine_spade_sweep/`, `spade_revival/`), not a flat
namespace: the source repo has eight distinct files named `Figure_4_karvonen_extension*.png`
in six directories, all different. Flattening loses five of them.

Two subdirectories are not experiments. `othello_rules/` draws the R1-R3 constraint rules of
[docs/othello_rules.md](../docs/othello_rules.md) on a single board, and it lives here because
every rules study cites that taxonomy and none of them owns it. `rules_and_generation/` puts
that board beside `analysis/generation_legality/`'s accumulated by-rule mass — the rules, and
what the model does with them — so it belongs to neither of the two studies it draws.

How a figure gets made is not a per-script choice. [FIGURE_GUIDELINES.md](../FIGURE_GUIDELINES.md)
is normative and plotting code cites it by section; `spade.figures.paper_style` is the only
module where a color literal may appear, and shared panel renderers live beside it. The
short version: encode by color *and* shape so it survives grayscale, put uncertainty on
every claim-supporting curve and state its semantics and n, style the reference class as a
reference and nulls as gray/dashed/last, and draw at final printed width rather than
rescaling in LaTeX.
