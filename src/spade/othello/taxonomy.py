"""7-category move-legality taxonomy, matched to the Othello engine's get_valid_moves().

Classifies an illegal candidate move into a category (occupied, no-flip, off-board, ...),
ordered by the board computation depth the category requires. The categories map onto the
formal rules in docs/othello_rules.md: R1 placement, R2 bracketing, R3 turn order. Forfeit
positions are excluded by an explicit contract, not by accident.

Shared by the rules, straightness and violation-divergence pipelines and by their figure
scripts. Uses circuits.othello_engine_utils.
"""

# CATEGORIES = [...]   # ordered by required board-computation depth
# LEGAL = ...
# def flips(board, move): ...
# def classify(board, move) -> str: ...
# def classify_all(board) -> dict: ...
