# Rules of Othello as representation targets in OthelloGPT

OthelloGPT predicts the next move in a legal game. Every rule of Othello is
therefore something the model must compute (or shortcut) internally. This doc
defines the full rule set and how each rule is operationalized as a probing /
geometry study. It is the reference the `spade.rules` studies implement against.

## Two classes of rules, two kinds of evidence

- **Constraint rules** restrict *which* moves may be played. Violations exist
  as concrete counterfactual tokens, so they are studied *observationally*:
  injected-move matched pairs (shared legal prefix, one legal vs
  category-stratified illegal continuations) embedded in 2D PCA and Isomap —
  do the rules' activations cluster relative to each other, and does each
  rule's violation class form its own manifold? See
  `analysis/rules_manifolds/`.
- **Dynamics rules** dictate how the board state *evolves* after a move. A
  game record cannot contain a violation of them — the state update is
  implied, never written — so every move exercises them and there is no
  violation class to embed. They are studied *causally*: clean/corrupted
  interventions measuring the effect on the next-move distribution, with
  isolation coming from diagnostic move sets (squares whose legality depends
  only on the intervened property — engine ground truth) or from
  representation-level edits along board-state probe directions, never from a
  whole-activation patch alone. Patched/edited activations join the same
  PCA/Isomap panels, so a successful intervention is visible as migration
  between manifolds.

Engine conventions (`external/Karvonen_Research/circuits/othello_engine_utils.py`):
black = 1, white = −1, black moves first; board is row-major 8×8; the four
center squares start occupied — black (3,4),(4,3), white (3,3),(4,4) — and are
excluded from the 60-token move vocabulary because they are never played.
Forfeits are unmarked in the move stream: the same color simply moves twice,
and the engine's `umpire` reinterprets a token that flips nothing for the
mover as the opponent's move.

## The rule set

**R1 — Placement-on-empty** *(constraint)*. A move places one disc of the
mover's color on an empty square.
Violations: `occupied_own`, `occupied_opp`. **Implemented** (manifold views).

**R2 — Bracketing** *(constraint)*. The placed disc must flank at least one
contiguous line of opponent discs terminated at the far end by the mover's own
disc, in any of the eight directions, with no empty gaps.
Violations, ordered by the board computation needed to detect them:
`empty_detached`, `empty_own_only`, `empty_no_bracket`. **Implemented.**
The per-rule view asks whether geometric distance from the legal manifold
tracks that computation-depth ordering.

**R3 — Turn order** *(constraint)*. Black moves first; colors alternate;
legality is evaluated for the mover, not the opponent.
Violation: `opponent_only` (legal for the opponent only — turn confusion).
**Constraint side implemented.** The representation side — probing whose turn
it is at every position — is defined but deferred.

The three constraint rules are drawn on one board in
`build/othello_rules/rules_legal_and_illegal.png` (committed copy archived)
(`src/spade/figures/figure_othello_rules.py`). One panel carries the move a corpus
game actually played, ticked, and one violation of each rule at the *same position*,
each crossed — an occupied square, a line of opponent discs that nothing of the mover's
closes, and a square that brackets only for the player who is not to move. One game,
one position, one board, so the three violations are comparable against the legal move
and against each other with nothing changing between them.

**R4 — Forfeit** *(dynamics)*. A player with no legal move anywhere is
skipped; the opponent moves again. Detecting a forfeit requires concluding
"no square is legal for the opponent" — a universal quantification over all
60 squares, the deepest rule computation in the game. In the data it is
unmarked, so tracking parity through a forfeit is the model's only evidence
of it. **Implemented** (`analysis/rules_causal/`).
Operationalization: clean = natural forfeit position, corrupted
= t-matched non-forfeit position; patch the forfeit position's activations
into the corrupted run, sweeping layers/positions. The vocabulary is
colorless, so "whose turn does the model think it is" is scored as
probability mass on squares legal for exactly one color.

**R5 — Flipping** *(dynamics)*. Placing a disc flips **all** bracketed
opponent discs in **all** directions. Flips are mandatory, simultaneous, and
non-recursive (flipped discs do not trigger further flips).
Sub-studies (signed off):
- *"Un-flip" edit (primary causal test)*: run `prefix + m` where m flips
  square s; edit the residual along board-state probe directions so s reads
  as its pre-flip color; score the next-move distribution against the real
  vs counterfactual board's legal sets on s-diagnostic squares. Built-in
  comparison: edit efficacy on just-flipped vs long-settled vs freshly-placed
  squares. **Implemented** (`analysis/rules_causal/`).
- *Pair-patching (localization)*: same prefix, two legal moves with different
  flip sets; patch layer-by-layer, position-by-position; score only on
  squares in the flip-set symmetric difference, away from both placements.
  **Implemented** (`analysis/rules_causal/`).
- *Flip-extent manifold view*: legal-move activations colored by
  discs-flipped and flip directions — is the legal manifold organized by
  flip extent? **Implemented** (`r5_flips` view in
  `analysis/rules_manifolds/`).

**R6 — Permanence** *(dynamics)*. Discs are never removed or moved; only
flips recolor them; occupancy grows by exactly one disc per move, so
empty→occupied is irreversible. Defined but deferred.
Operationalization when taken up: probe fidelity of a square's state as a
function of time-since-placement and number of times flipped.

**R7 — Initial setup** *(static)*. Fixed center-four configuration, given
above. A constant admits no counterfactual contrast — sanity checks only,
no study.

**R8 — Termination and scoring**. The game ends when neither player can move;
most discs wins. Out of scope: a next-move model has no prediction target
after the game ends, and the "no one can move" computation is R4's
quantifier.

Derived invariants — disc count = t + 4, every legal move is adjacent to an
existing disc, parity ↔ move index absent forfeits — are consequences of the
rules above, not independent rules. They get no studies of their own
(`empty_detached` already probes the adjacency corollary).

## Status and mapping

| Rule | Class | Status |
|------|-------|--------|
| R1 placement-on-empty | constraint | manifold views implemented |
| R2 bracketing | constraint | manifold views implemented |
| R3 turn order | constraint | manifold views implemented; parity probe deferred |
| R4 forfeit | dynamics | implemented (behavioral + causal) |
| R5 flipping | dynamics | manifold view + causal tests implemented |
| R6 permanence | dynamics | defined, deferred |
| R7 initial setup | static | sanity only |
| R8 termination | — | out of scope |

The pipeline lives in `src/spade/rules/` (pairs → activations → flip labels →
embeddings; causal experiments in `src/spade/rules/causal/`) with figures in
`analysis/rules_manifolds/` and `analysis/rules_causal/`. The six illegal
categories are exactly the violations of R1, R2, and R3, and they are defined
once in `spade.othello.taxonomy`, which four studies import. An earlier
metrics-heavy legality analysis (probes, distance AUCs, matched contrasts) was
retired in favor of this visualization-first design; it survives only in the
source repo's git history under `src/legality/`.

One result here is retired and must not be restated: **R3's directional
prediction is falsified** — the `opponent_only` category shows no parity
signature (agreement 0.9930, mid-pack). See `archive/docs/claims_status.md`.
