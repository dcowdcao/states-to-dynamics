# Analysis

Downstream studies on the trained SAEs. Code and outputs stay co-located: each study is a
self-contained folder you can rerun in place. Nothing here retrains — analyses read
precomputed eval tensors under `results/`, or the study intermediates on netscratch.

The division of labour with `src/spade/`: the heavy, GPU-bound, resumable pipeline steps
live in the package; the folder here holds `make_figures.py`, which only draws. A study
that recomputes inside its figure script has put its expensive half in the wrong place.

- `spade_vs_baselines.md` — the headline comparison writeup. Its pre-2026-07 sections are
  about distance-SpaDE only; see `docs/claims_status.md`.
- `linear_probe/` — the gold-standard board-state probe (0.9919 at layer 5) reused as ground
  truth by the rules, bracket-flip and violation-divergence studies. Its board labels used to
  carry 53 mislabeled rows in 10,000, because the mover was taken from move index rather than
  from engine replay; the labels and every artifact under `activations/` have been rebuilt
  (`docs/finish_notes/probe-parity-fix.md`), and anything computed before that used the old
  ones.
- `geometry/` — 13 studies of SAE feature geometry; an older generation, built on a 7-SAE
  set. See `geometry/SUMMARY.md`, and read `docs/claims_status.md` before quoting it.
- `rules_manifolds/`, `rules_causal/` — the rules-of-Othello program.
- `bracket_flip/`, `violation_divergence/`, `straightness/`, `manifold_capture/` — the
  four studies built on top of it.
- `generation_legality/` — the prior question those four skip: does the model emit illegal
  moves unprompted (0.9982 of next-move mass is legal; 11.5% of self-generated games are
  not), and which rule the residual breaks.
- `move_legality/` — a linear probe on the residual at the position of a *played* move,
  read out over a whole game to localize a single spliced illegal move. Its own control,
  the model's next-move probability for that move, beats it outright; read the results
  section before quoting a probe number.
- `cosine_spade_gamma_sweep/` — the γ-sweep writeup; its data lives in `results/multiseed/`.

Each study's README is canonical prose and is ported from the source repo rather than
rewritten — several of them lead with a retraction, and that structure is deliberate.
