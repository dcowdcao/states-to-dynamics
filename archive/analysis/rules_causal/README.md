# Rules causal experiments

Clean/corrupted causal tests for the dynamics rules — the rules every move
exercises, so they have no violation class to embed. Rule definitions and
study logic: `docs/othello_rules.md`.

- **R5 un-flip edit** (`unflip_edit.png`) — for a move that flips square s,
  the layer-5 residual at the move's position is edited along the board-state
  probe direction so s reads as its pre-flip color; probability mass should
  move onto the counterfactual board's legal set and off the real board's
  (diagnostic squares only — legal on exactly one board). Populations at
  matched edit magnitude: just-flipped vs long-settled vs freshly-placed
  squares, plus a random-direction null. The probe is the canonical board probe
  (`spade.probes.load`); edit magnitudes are calibrated to class-mean probe
  projections on that corpus's **test** games, the split the probe never saw.
- **R5 pair patching** (`pair_patch.png`) — same prefix, two legal moves;
  per-layer attn_out / mlp_out component patches at the injected position
  (full-residual patches are tautological when the runs share a prefix —
  see `src/spade/rules/causal/patch_battery.py`); score recovery on
  legality-differing squares. `strict` restricts to squares whose eight rays
  avoid both placements, so the difference can only arrive through flipped
  discs — the flip-attributable curve. Localizes where flip consequences
  are written into the stream.
- **R4 forfeit** (`forfeit.png`) — natural forfeit minimal pairs (clean:
  move a after which the opponent has no move anywhere and the same color
  moves again; corrupted: alternative move b restoring alternation).
  Behavioral panel: legal-set mass at forfeit vs matched non-forfeit
  positions. Causal panel: per-layer patching of the color score
  (e-only vs d-only legal squares on the corrupted board) — does the patch
  carry "the opponent has no move"?

Pipeline (one SLURM job — `scripts/slurm/studies/rules/submit_rules_causal.sh`):
`spade.rules.causal.mine_forfeits` (CPU stream; skipped if mined) →
`unflip_edit` → `pair_patch` → `forfeit_patch` → this folder's
`make_figures.py` (`--camera-ready` for vector PDFs). Per-item CSVs in
`results/`, mined pairs on `$SPADE_RULES_DIR/causal/`.

Error bars are 95% CIs of the mean over items (edits), pairs
(patching) or events (forfeit), with n on every panel. Layers are transformer
blocks 0-7, 0 = first block. The `random` edit direction and the unpatched
corrupted run are nulls and are drawn as such — gray, dashed, last in the
legend; a full-residual patch recovers 1.0 by construction, so it is the labeled
ceiling rather than a result.

Reading guide: a real flip representation shows the `flipped` population with
Δmass onto counterfactual-only squares well above the `random` null, and the
pair-patch `strict` curve rising at the layers where flips are computed. In
the forfeit panel, patched color score above the corrupted baseline at
pre-final layers means turn assignment transfers as a computed feature, not
just as final-logit content.

## Figure notes

The rule each figure tests, which used to be printed in a grey key under the
panels:

- `unflip_edit.png` / `unflip_edit.pdf` — R5 flipping — placing a disc flips every bracketed opponent disc, in all directions at once.
- `pair_patch.png` / `pair_patch.pdf` — R5 flipping — placing a disc flips every bracketed opponent disc, in all directions at once.
- `forfeit.png` / `forfeit.pdf` — R4 forfeit — a player with no legal move anywhere is skipped, and the opponent moves again.

## Headline results (jobs 34239305 for the patching arms, 40534718 for the edit)

- **Un-flip edit**: flipped-square edits move +3.51 log-mass onto the
  counterfactual legal set at β=1 (+7.06 at β=2) vs +0.22 (+0.84) for the
  random-direction null — ~16x, with the correct asymmetry (real-only mass
  drops 1.51 / 4.99). `settled` ≈ `flipped` (+3.39 vs +3.51): a flip is
  integrated into the causally-used world model immediately — no "just flipped"
  fragility. `placed` (theirs→empty) is still the strongest edit (+8.68),
  sensible since un-placing removes a whole disc from every bracket
  computation. These numbers are ~3x the ones this section used to carry
  (+1.07 / +0.10): the edit target `mu` is now read off the canonical probe's
  test games rather than the retired ARENA activation set, and the direction is
  the canonical probe's. The ordering and the asymmetry are unchanged.
- **Pair patching** (probe-independent; re-ran byte-identical against the
  canonical probe, which is the check that it is): flip consequences are
  written by MLPs — recovery rises
  mlp4 (0.13) → mlp5 (0.38) → mlp6 (0.57) and collapses at mlp7; attn_out
  ≈ 0 through layer 6 with a small attn7 effect. `strict` and `all` curves
  overlap, so the transfer is genuinely flip-borne, and mlp0 (~0.2-0.26)
  shows an immediate partial write at the injection layer. Full-resid sanity
  = 1.00 exactly.
- **Forfeit** (probe-independent; also byte-identical): behaviorally solved — 99.8% top-1-legal at natural forfeit
  positions, legal-set mass ≈ non-forfeit controls. Causally, the color
  score climbs from the corrupted baseline (−7.1) to −4.5 (mlp5) and −3.0
  (mlp6), with attn7 the largest single component (−2.4 vs clean −0.4):
  the "opponent has no move" conclusion is computed in the mid-late MLPs and
  routed by final-layer attention.
