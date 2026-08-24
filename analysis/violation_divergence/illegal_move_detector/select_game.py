"""Pick the one game the board panel and the trace both draw, and write it down.

Two panels of one game only work if the game is honest about the claim. The criteria, in the
order they bind:

  * all three rules have a branch at this game's t0, so one board shows R1, R2 and R3,
    each drawn from the member category that is actually legible as its rule (PREFERRED)
  * t0 - 5 .. t0 + 5 all sit inside the scanned band, so the trace needs no truncation
  * raw's score peaks at t0 on all three rule arms, so the figure is not drawing a miss
  * the matched legal control does NOT peak at t0, so the peak reads as the violation
  * the detector is right about at least MIN_CORRECT of the five plies board_states draws:
    all three rule branches clear tau at t0, and the reference game -- legal at every one of
    those plies -- stays under tau at at least MIN_CORRECT - 1 of the other four
  * among those, the largest margin between the rule arms' t0 score and their own runner-up

The last two criteria are selections over games and the figure caption has to say so: this is
an illustrative game chosen to be legible, not a typical one. Figures 3 and 4 are the
population, where 28% of legal positions sit above tau. MIN_CORRECT is 4, not 5, so the
criterion does not hunt for a window with no false alarm in it; the current pick happens to
have none.

Writes figure_game.json.

Usage: PYTHONPATH=src python analysis/violation_divergence/illegal_move_detector/select_game.py
"""

import json
from pathlib import Path

import numpy as np

import detector
from detector import CAND_OFFSET, RULES
from spade.othello.taxonomy import CATEGORIES

HERE = Path(__file__).resolve().parent
TRACE_HALF = 5
PANELS = (-2, -1, 1, 2)   # the plies board_states draws either side of t0
MIN_CORRECT = 4           # of its five panels, t0 included

# Which member category to draw for each rule, and why -- `figure_othello_rules.py` settled
# this and the panels must agree with it. R2's `empty_detached` touches no opponent disc at
# all, so it draws an empty square with nothing to see; `empty_no_bracket` has a real line of
# opponent discs that simply is not closed. R1's `occupied_opp` is also the cell the supervised
# probe reads worst, so it is the one worth showing. Fall back in order if a game lacks one.
PREFERRED = {"R1": ["occupied_opp", "occupied_own"],
             "R2": ["empty_no_bracket", "empty_own_only", "empty_detached"],
             "R3": ["opponent_only"]}


def main() -> None:
    data = detector.load()
    cand = detector.band()
    s = detector.score(data, cand)
    sel = detector.arms(data)
    tau = detector.operating_point(s, cand, data, sel)
    move = cand + CAND_OFFSET

    t0_of = data["t0"]
    games = data["game_id"][sel["reference"]]
    peak = move[s.argmax(1)]
    rule = detector.rule_of_row(data["category"])

    category = np.array([CATEGORIES[c] for c in data["category"]])
    rows_by_game = {}
    for r in np.arange(len(data["game_id"])):
        rows_by_game.setdefault(int(data["game_id"][r]), []).append(r)

    best = None
    for g, rows in rows_by_game.items():
        t0 = int(t0_of[rows[0]])
        if not (move[0] <= t0 - TRACE_HALF and t0 + TRACE_HALF <= move[-1]):
            continue
        by_rule = {u: [r for r in rows if rule[r] == u] for u in RULES}
        if not all(by_rule.values()):
            continue
        rank = lambda r: PREFERRED[rule[r]].index(category[r])
        picked = {u: min(v, key=rank) for u, v in by_rule.items()}
        if not all(peak[r] == t0 for r in picked.values()):
            continue
        control = [r for r in rows if data["is_control"][r] == 1]
        if not control or peak[control[0]] == t0:
            continue
        reference = int([r for r in rows
                         if data["is_control"][r] == 0 and rule[r] == ""][0])
        j = int(np.searchsorted(move, t0))
        if not all(s[r, j] > tau for r in picked.values()):
            continue
        quiet = sum(s[reference, int(np.searchsorted(move, t0 + o))] <= tau for o in PANELS)
        if 1 + quiet < MIN_CORRECT:
            continue
        margin = min(s[r, j] - np.delete(s[r], j).max() for r in picked.values())
        if best is None or margin > best["margin"]:
            best = {"game_id": g, "t0": t0, "margin": float(margin),
                    "panels_correct": int(1 + quiet),
                    "rows": {u: int(v) for u, v in picked.items()},
                    "categories": {u: str(category[v]) for u, v in picked.items()},
                    "control_row": int(control[0]), "reference_row": reference}
    assert best, "no game satisfies the panel criteria"
    (HERE / "figure_game.json").write_text(json.dumps(best, indent=2) + "\n")
    print(json.dumps(best, indent=2))
    print(f"\nwrote {HERE / 'figure_game.json'}")


if __name__ == "__main__":
    main()
