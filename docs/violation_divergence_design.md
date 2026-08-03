# Divergence after an illegal move: two futures from one prefix

Design doc for the `violation_divergence` study. Subagents implement against
this; the rule set it sits in is `docs/othello_rules.md`.

> ## Correction — this design's central premise did not hold
>
> The study ran, and the hypothesis it was built to test was **inverted in the
> data**. This document is kept as the record of what was intended and how the
> branch set was constructed, not as a statement of results.
>
> - **"An illegal branch diverges further / persists longer than a legal
>   control" is false.** The illegal/control ratio at layer 5 is 0.9793
>   [0.9719, 0.9854] paired — illegal branches diverge *less*. The honest
>   statement of the effect is: *the representational effect of an impossible
>   move is 92–103% of an ordinary legal one.*
> - **The fairness argument below — "the null carries the same handicap" — is
>   false**, and the confound runs *against* the hypothesis. The paragraph is
>   corrected in place in "The branch construction" section; read that before
>   reusing this design.
> - **"The divergence peak migrates with depth" is false.** `peak` and `peak_r`
>   are max-of-noise: the persistence kernel averages per-row maxima, i.e.
>   E[max] rather than max E. A within-row permutation null reproduces the peak
>   to 0.4% at every layer and 79–83% of the depth drift, and the population
>   mean curve peaks at offset 0 in every layer. `peak_layer` as defined here is
>   separately vacuous, because the metric normalizes within each space so a
>   cross-space "excess over the input floor" has no meaning.
> - **R3's directional prediction is falsified** — the opponent-only category
>   shows no parity signature (agreement 0.9930, mid-pack).
> - **`kl_peak_L5` is misnamed** in the contract below: it is a logit quantity,
>   not a layer-5 one.
>
> What survives is in `docs/claims_status.md`, and it is worth stating because it
> is the study's real result: the probe decodes a disc on the illegally played
> square in 98.5–99.9% of events, there is no surprise signature, and a
> Standard-SAE feature fires on 99–100% of empty-square violations against 25–26%
> on both legal branches. *The legality signal is computed and linearly
> available, and the model does not act on it.* There is no illegal-specific
> direction — the legal-vs-legal control reproduces the same cosine band.
>
> One open defect: three sub-studies were computed on the superseded one-control
> branch set and never re-derived against two controls.

## The question

A game is legal up to position *t0*. At *t0* two futures branch: the move that
was actually played, and an illegal one. Every move after *t0* is the real
game's own — legal, ordinary Othello. How far apart do OthelloGPT's activation
trajectories travel, in which direction, and does the gap close? Then the same
question in each SAE latent space: does the dictionary make the violation
sparse and localized, or smear it across the code?

## The branch construction, which is what the study rests on

Source game *g* (59 tokens, HF synthetic corpus = the model's training
distribution). Branch position *t0*. Every branch shares the prefix `g[:t0]`
**and** the suffix `g[t0+1:]`:

| branch | token at *t0* | role |
|---|---|---|
| `reference` | `g[t0]` | the real game; every distance is measured against it |
| `control_legal` | another legal move ≠ `g[t0]` | the null — legal, but still changes the board |
| six illegal | one token per available category of `spade.othello.taxonomy` | the conditions: R1 ×2, R2 ×3, R3 ×1 |

Two branches differ in **exactly one token**. A divergence at position *t0+r* is
therefore attributable to that token and to nothing else. This is why the suffix
is shared rather than re-derived per branch: two independently continued games
differ in every token after *t0*, and their distance measures move content, not
the violation.

Stated plainly, because it is the design's one real cost: after *t0* the suffix
is legal on the reference board and off-distribution for every other branch,
`control_legal` included. That is the object of study — how the model carries an
impossible history.

> **This paragraph used to end "and the null carries the same handicap, so the
> illegal-vs-control contrast is fair." That was wrong, and the study measured
> it.** The null carries the *larger* handicap. A legal alternative genuinely
> updates the board and passes the turn, so the shared suffix goes wrong for the
> control: `g[t0+1]` stays legal on the control's board in only 76.8% of games
> and a median of 2 suffix moves survive, while an illegal move is absorbed as a
> near-no-op and its suffix stays consistent with the model's belief state. The
> confound therefore runs *against* the hypothesis, and `ratio_illegal_control`
> is not a clean effect size. See `analysis/violation_divergence/RED_TEAM.md` §F1.
> Anyone reusing this branch design needs a second legal control per game, whose
> divergence against the first is the null this one lacks.

**What this buys over the straightness study's violation class.** That class had
to update the board after the illegal move in order to generate a continuation.
The update is ill-defined for a disc placed on an occupied square, so the class
was restricted to the four empty-square categories and needed a
place-with-no-flips-and-pass convention. Here the suffix comes from the source
game, no board update after *t0* is ever needed, and **all six R1–R3 categories
are in scope**, occupied squares included.

**And the occupied-square categories are the cleanest contrast available.**
Every other branch adds a disc, and `control_legal` also flips **1.93** discs on
average (measured on this study's own branch set; `flip_labels.py`'s 2.28 is
over t ∈ [5, 57] and must not be quoted here — the t0 ∈ [10, 34] window is
earlier and less flippy). `occupied_own` adds nothing at all — the
mover's disc is already there, and the only consistent reading of the move is a
no-op. So `occupied_own vs control_legal` separates *the model noticed something
impossible* from *the board changed*, which no other pair in this design does.

`occupied_opp` is the interesting sibling rather than a second copy: placing on
the opponent's square admits two readings, no-op or recolor, and they are not
the same board. **Which one the model behaves as if it took is a result, not an
assumption** — it is read off `occupied_own vs occupied_opp` in M1 and M5, and
nothing in the pipeline should presume either.

## Rules in scope

R1, R2, R3 — exactly the constraint rules, exactly the six categories of
`spade.othello.taxonomy`. R4 (forfeit) and R5 (flipping) are dynamics
rules whose violations cannot appear as a token at all
(`docs/othello_rules.md`), so "R1, R2, R3, …" stops at R3 by construction rather
than by choice.

## Window, and why there is no attrition

*t0* is assigned round-robin over **[10, 34]**, and the analysis window is
**r ∈ [−8, +24]**. At *t0* = 34 the offset r = 24 lands on position 58, the last
one; at *t0* = 10 the offset r = −8 lands on position 2. **Every source game
therefore contributes to every offset in the window**, so n is constant across
the x-axis and the usual ragged-n caveat of event-locked panels does not apply
here. Say so in the captions — it is a property worth having.

## Spaces

- Raw `resid_post` layers 0–7, plus the token embedding as the input anchor
  (`FIGURE_GUIDELINES.md` §4.1). Layer 5 is the headline; the depth sweep is
  what separates *a different token went in* from *the model computed a
  consequence*.
- Eight SAE latent spaces: the seed-42 registry minus the SpaDE γ=1e-2
  checkpoint, whose provenance is not trustworthy. That exclusion is no longer
  made here — it is made once in `spade.spaces` and applies to every study and
  to the viewer, with the SpaDE family represented by the default λ=1e-3, γ=1e-4
  point. This is the same set as the straightness study, so the two trajectory
  studies compare directly.

**Cross-space comparability.** Latent norms differ by two orders of magnitude
across families, so every divergence is reported *relative*: as a fraction of
the reference trajectory's own norm, as a cosine, or as a ratio to that space's
own `control_legal` divergence. Never as a bare distance across families
(the percent-of-level rule).

## Measurements

`z^b_s` is branch *b*'s activation at position *s* in a given space;
`r = s − t0`; `Δ_b(r) = z^b − z^ref` at position *t0+r*.

- **M1 — divergence magnitude.** `δ̂_b(r) = ‖Δ_b(r)‖ / ‖z^ref‖`. Causal attention
  makes `δ̂ ≡ 0` for `r < 0`; that is asserted, and the region is shaded and
  annotated in every panel (§4.2).
- **M2 — direction.** `cos(Δ_b(r), Δ_b(0))`: does the offset introduced at *t0*
  persist as one direction, or rotate away? And `cos(Δ_b(r), Δ_control(r))`: do
  illegal and legal branches leave along the same direction or different ones?
- **M3 — persistence.** From `δ̂_b(r)` over `r ≥ 0`: peak value, the offset it
  peaks at, half-life (first `r` past the peak with `δ̂ ≤ ½ peak`), and plateau
  (mean over `r ∈ [20, 24]`). The question is whether the model recovers from
  the violation or carries it to the end of the game. **The peak and its offset
  did not survive red-teaming** (see the correction header); the plateau and the
  never-halves fraction did, because they are means rather than
  averaged per-row maxima.
- **M4 — sparsity of the divergence (SAE only).** Support Jaccard
  `|S_b ∩ S_ref| / |S_b ∪ S_ref|`, `ΔL0`, the share of `‖Δ‖` carried by the 10
  largest coordinates, and the participation ratio of `Δ²`
  (`(Σ Δ_j²)² / Σ Δ_j⁴` — an effective coordinate count, comparable across
  spaces of different width in a way a top-10 share is not). Plus the
  cross-event recurrence of the top-|Δ| features per category: a violation
  feature that recurs is worth naming. This is the measurement raw space cannot
  make, and it is what a dictionary is supposed to buy.
- **M5 — behavioral divergence.** `KL(p^b_s ‖ p^ref_s)` over the next-move
  distribution at every `s ≥ t0`, and the probability mass each branch puts on
  the reference board's legal set. Free with the forward pass, and it grounds
  M1: representational divergence that never reaches behavior is a weaker claim
  than one that does.
- **M6 — what the divergence encodes.** The committed layer-5 linear probe
  (`analysis/linear_probe/activations/linear_probe.pt`, `[512, 8, 8, 3]`,
  accuracy 0.9896, already the instrument behind
  `spade.rules.causal.unflip_edit`) read out on both branches at every
  position: does the illegal branch's decoded board contain the illegally
  placed disc, and if so for how many plies? It is one einsum against
  activations that are extracted anyway, so this is not a cost question.

  **The frame is the subtlety.** The probe is mover-relative — {empty, theirs,
  mine} with "mine" = the player to move next — and after an illegal move whose
  turn it is is exactly what is in dispute; R3 violations *are* turn confusion.
  The shared suffix is what makes this tractable: the reference branch fixes the
  frame, both branches are scored against the reference board's labels in it,
  and the ambiguity stops being a blocker and becomes a measurement. Report by
  square class (the illegally played square, the reference move's square,
  untouched discs, empty squares), and separately whether the illegal branch's
  decoded parity tracks the reference's — R3 turn confusion read straight off
  the representation. That parity readout is the one that came back null.

  This study needs **no tensor-product probe**. The TPR factorizes a square's
  role⊗filler binding, which is the bracket-flip study's question; nothing here
  reads one square's binding structure.

## Confounds, and how each is handled

- **Token identity.** Branches play different squares, so some divergence is
  just *a different token went in*. The token-embedding space is the floor:
  whatever `δ̂` it shows is input-side, and every depth claim is about the excess
  over it. (The excess is not comparable *across* spaces — see the correction
  header on `peak_layer`.)
- **Board change.** Handled three ways: the `control_legal` null; the
  occupied-square subset, where the board does not change at all; and reporting
  `control_legal`'s divergence stratified on how many discs it flips
  (`spade.rules.flip_labels.flip_stats`). The null itself is not clean — see the
  corrected paragraph above.
- **Attrition.** Eliminated by the window above rather than corrected for.
- **t0 distribution.** Round-robin over [10, 34], so *t0* is uniform by
  construction and identical across branch types within a game.
- **Category availability.** Not every category exists at every prefix, so
  branch counts are unequal — report the per-category n, and never pool
  categories without saying so. Measured on the 2000-game set: `occupied_own`,
  `occupied_opp` 2000; `opponent_only` 1998; `empty_detached` 1956 (absent only
  at t0 ≥ 26, sharply — the adjacency corollary biting as the board fills);
  `empty_no_bracket` 1941; **`empty_own_only` 1636**, absent in 18.2% of games
  and spread across all t0. That last one is the category to watch: it is the
  only one whose missingness is large, and it is not a t0 effect, so whatever
  makes a prefix offer no own-only square is a board-shape confound that the
  per-category n does not capture on its own.
- **Position 0** is an attention sink (norm ~300 vs ~22) and is never used.

## Layout

```
docs/violation_divergence_design.md                     this doc
src/spade/violation_divergence/{branches,extract_acts,divergence,compute_divergence}.py
analysis/violation_divergence/{make_figures.py,figures/,README.md}
results/violation_divergence/*.csv                      committed, small enough to read as text
scripts/slurm/studies/violation_divergence/
```

Large artifacts on `$SPADE_VIOLATION_DIVERGENCE_DIR` (default netscratch
`spade_violation_divergence`), resolved by `spade.common.study.study_dir()` and
shared across worktrees so extraction runs once.

## Interfaces the agents must honor

Exact, so wave 2 can be written before wave 1's data lands.

```python
# spade.common.study
study_dir("violation_divergence") -> Path

# spade.violation_divergence.branches  ->  $DIR/branches/branches.pt
# torch.save of aligned tensors, one row per branch sequence:
#   tokens        int16  [n, 59]  full sequence; no padding, no pass tokens
#   game_id       long   [n]      source game; rows sharing it share prefix and suffix
#   t0            long   [n]      0-based position of the branch token
#   category      int8   [n]      spade.othello.taxonomy index; 0 = legal
#   is_control    int8   [n]      1 for control_legal, 0 otherwise
#   branch_token  long   [n]      the token at t0
#   n_flipped     long   [n]      discs the branch token flips; -1 for illegal branches
#   n_legal       long   [n]      legal moves available at t0
# Exactly one reference row (category == 0 and is_control == 0) per game_id.
# Also $DIR/branches/manifest.json: counts per category, skips, t0 histogram,
# seed, dataset, datasets version.

# spade.violation_divergence.extract_acts  ->  $DIR/acts/
#   resid_post_{0..7}.pt   fp16 [n, 59, 512]
#   token_embed.pt         fp16 [n, 59, 512]
#   logits.pt              fp16 [n, 59, 61]   next-move distribution at every position
# Rows aligned with branches.pt.

# spade.violation_divergence.divergence    pure tensor kernels, no I/O
relative_divergence(Zb, Zref)      -> Tensor  # [n, 59]
direction_cosine(Zb, Zref, t0)     -> Tensor  # [n, 59]  cos(Δ(r), Δ(0))
support_jaccard(Fb, Fref)          -> Tensor  # [n, 59]
topk_share(Fb, Fref, k=10)         -> Tensor  # [n, 59]
participation_ratio(Fb, Fref)      -> Tensor  # [n, 59]
align(values, t0, window)          -> Tensor  # [n, len(window)]  event-locked
persistence(delta_hat, window)     -> dict    # peak, peak_r, half_life, plateau

# spade.violation_divergence.compute_divergence -> $DIR/divergence/<space>.npz
# Per space, the event-locked arrays every figure and CSV reads. Resumable:
# a space whose npz exists is skipped.
```

A 50-game smoke set mirrors the same tree one level down —
`$DIR/smoke/{branches,acts,divergence}/` — so every module takes one
`--smoke` flag and nothing else changes. It exists so the analysis wave is
never blocked on the GPU queue.

Two integer conventions, as everywhere else in this repo: **tokens are 1–60**,
**engine squares are 0–63**, and `to_string(token)` converts. `category` indexes
`spade.othello.taxonomy.CATEGORIES`.

## Acceptance criteria

- **Prefix invariance.** For a sample of games, every branch's activations agree
  exactly at all positions `< t0`, and `δ̂(r < 0)` is exactly zero.
- **Engine re-derivation.** The reference game replays legally end to end;
  `classify(board_at_t0, branch_square) == category` for every branch; the
  `control_legal` token is legal at *t0* and differs from the reference token.
- **Suffix identity.** Rows sharing a `game_id` agree on `tokens[:t0]` and on
  `tokens[t0+1:]`, and differ at `t0`.
- **No pass tokens, no padding.** Every sequence is 59 nonzero tokens.
- The two analysis agents independently report the same n and the same sign and
  rough magnitude for the headline effect; the red team arbitrates.

## Cross-validation contract

Wave-2 agents report these keys, under these names, in a `RESULTS.md` table.
Comparing them is mechanical, which is the point — agreement is checkable
rather than asserted.

| key | definition | reported by |
|---|---|---|
| `n_games`, `n_branches` | source games; branch sequences | A3, A4, A5 |
| `delta_peak_L5[cat]` | peak relative divergence at layer 5, per category — **max-of-noise, retired** | A3, A5 |
| `delta_peak_r_L5[cat]` | offset *r* of that peak — **max-of-noise, retired** | A3, A5 |
| `delta_plateau_L5[cat]` | mean `δ̂` over r ∈ [20, 24] | A3, A5 |
| `ratio_illegal_control_L5` | plateau `δ̂`, illegal (pooled) ÷ `control_legal` — **not a clean effect size; the null carries the larger handicap** | A3, A4, A5 |
| `delta_peak_WE` | the same peak in token-embedding space — the input floor | A3, A5 |
| `peak_layer` | layer maximizing `δ̂` excess over the token-embedding floor — **vacuous as defined; the metric normalizes within each space** | A3, A5 |
| `jaccard_min[space]` | minimum support Jaccard over *r*, per SAE space | A4, A5 |
| `topk10_share[space]` | share of `‖Δ‖` in the 10 largest coordinates at r = 0 | A4, A5 |
| `kl_peak_L5` | peak KL of the next-move distribution — **misnamed: a logit quantity, not a layer-5 one** | A3, A5 |
| `p_illegal_disc_decoded[cat]` | events whose illegal branch decodes a disc on the illegally played square, at r = 0 | A3, A5 |
| `parity_agree_r0[cat]` | events whose illegal branch decodes the same mover as the reference, at r = 0 | A3, A5 |

Agreement: counts exact, `delta_peak_r_L5` and `peak_layer` exact, proportions
within 2 points, ratios within 5%. Any mismatch is A5's to arbitrate and the
integrator's to resolve before merging — explained, never averaged away.

## Figures

Per `FIGURE_GUIDELINES.md`; colors and markers come from
`spade.figures.paper_style`, which only the integrator extends.
`CATEGORY_STYLE` already covers the six illegal categories and `legal`;
`control_legal` takes `NULL` styling — gray, dashed, last in the legend —
because that is what it is. Every figure prints the rule key
(`paper_style.rule_key`).

The figure set was pruned after the results landed: what shipped as seven
figures over ~30 panels is six figures over 11, because most of the panels were
showing the same null in a different basis. What each survivor earns its place
for is below; the cut ones are named with their reason.

- **F1** `figures/divergence_layer5` — event-locked `δ̂(r)` at layer 5 per
  category, and the paired excess over `control_legal`. Two panels: the two
  cosine panels it carried said "no illegal-specific direction", which is a null
  and is now a sentence. `cos_t0` / `cos_control` stay in
  `divergence_curves.csv`.
- **F3** `figures/depth_sweep` — plateau `δ̂` and the never-halves fraction vs
  layer, `token_embed` anchored as its own tick behind a rule. The `peak` and
  `peak_r` panels are **removed as unsound, not merely as clutter**: both average
  a per-row argmax over 33 offsets, estimating E[max] rather than max E, and an
  offset permutation reproduces them (RED_TEAM.md F2). The two survivors are
  means.
- **F6** `behavioral/f6_behavioral_divergence` — KL and reference-board legal
  mass vs *r*.
- **F7** `probe_readout/f7_probe_readout` — does the illegal branch write the
  disc, and does an occupied square read as no-op or recolor. The parity panel
  is gone: the two mover conventions disagree on 0.20% of cells and the probe is
  at chance on exactly those, so the curve was flat at ~1.
- **`sae_spaces/sae_feature_recurrence`** — the one place a dictionary separates
  from the residual stream: features that fire on a violation and almost never
  on either legal branch. **F2** (δ̂ over nine spaces, 3×3), **F4** (Jaccard /
  top-10 / PR) and **F5** (space × category plateau matrix) are cut — every
  space lands in the same 0.91–1.03 band, and 21 panels is an expensive way to
  print a null that the CSVs already carry.
- **The fork figures** `game_trajectory/{game_0_fork_pca, space_fork_grid_pca}`
  — one game's layer-5 trajectory forking at *t0*, move index on its own axis
  over a PCA of the *displacement*. Illustration of the design, not
  claim-supporting. Isomap variants and the six-game grid are cut; PCA answers
  the question the figure is for.

Uncertainty on every claim-supporting curve: 95% CI over source games, n stated,
empirical nulls preferred over analytic chance lines.

## Running things on this cluster

`python` is not on `PATH` and `module` is unavailable in non-interactive shells;
use `/n/home10/dcao/.conda/envs/spade_env/bin/python` directly. GPU work goes
through `scripts/slurm/studies/violation_divergence/` (account `kempner_ba_lab`,
partition `kempner_h100`). Extraction is ~15k sequences × 59 tokens in one
batched pass — minutes, not hours. `HF_USE_SOFTFILELOCK=1` does nothing in this
datasets version; point `HF_DATASETS_CACHE` at node-local `$TMPDIR` instead and
leave `HF_HOME` on NFS.

## Agent plan

Two waves of worktree-isolated agents, each merged only after review.

**Wave 1 — foundation, parallel, disjoint files.**

- **A1 data.** `branches.py`, `extract_acts.py`, the SLURM script,
  `DATA_REPORT.md`. Everything that touches the engine or the model.
  Also writes a 50-game smoke set so wave 2 is not blocked on the queue.
- **A2 metrics.** `divergence.py` and `compute_divergence.py`: pure kernels with
  a synthetic selftest that passes before any real data exists, written against
  the interface above.

The split keeps A2 off the GPU and off A1's critical path.

**Wave 2 — analysis, parallel, deliberately overlapping.**

- **A3 raw activation space.** M1, M2, M3, M5, M6 across `resid_post` 0–7 and
  the token embedding → F1, F3, F6, F7.
- **A4 SAE latent spaces.** M1, M3, M4 across the eight dictionaries → F2, F4,
  F5.
- **A5 red team.** Re-derives the contract's headline keys by its own code path
  from `$DIR/acts/`, attacks the confound handling (token identity, board
  change, category availability, the shared-suffix decision, prefix invariance,
  the sink, SAE standardization), and writes a verdict. No figures.

A5 is why the correction header at the top of this document exists, and is the
part of the plan that earned its cost.

**Rules for every agent.** Work only inside your assigned paths. Never edit
`spade/figures/paper_style.py`, `FIGURE_GUIDELINES.md`, `PROCESS.md`, `CLAUDE.md`,
or any existing study — request changes from the integrator. Read `CLAUDE.md`
and `FIGURE_GUIDELINES.md` first. Reuse `spade.othello.taxonomy`,
`spade.othello.boards`, `spade.othello.games.stream_games`,
`spade.rules.flip_labels.flip_stats`, `spade.saes.registry` and
`spade.evaluation.karvonen_adapter`; do not reimplement the Othello engine or SAE
loading. Report numbers, not adjectives.

**Integration (not delegated).** Cross-check A3/A4/A5 against the contract,
resolve disagreements, merge worktrees, unify figures and `paper_style` entries,
write `analysis/violation_divergence/README.md` and the PROCESS.md entry, commit
once.
