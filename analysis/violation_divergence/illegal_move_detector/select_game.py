"""Pick the one game the board panel and the trace both draw, and write it down.

Two panels of one game only work if the game is honest about the claim. The criteria, in the
order they bind:

  * all three rules have a branch at this game's t0, so one board shows R1, R2 and R3
  * t0 - 5 .. t0 + 5 all sit inside the scanned band, so the trace needs no truncation
  * raw's score peaks at t0 on all three rule arms, so the figure is not drawing a miss
  * the matched legal control does NOT peak at t0, so the peak reads as the violation
  * among those, the largest margin between the rule arms' t0 score and their own runner-up

The last criterion is a selection over games and the figure caption has to say so: this is an
illustrative game chosen to be legible, not a typical one. Figures 3 and 4 are the population.

Writes figure_game.json.

Usage: PYTHONPATH=src python analysis/violation_divergence/illegal_move_detector/select_game.py
"""

import json
from pathlib import Path

import numpy as np

import detector
from detector import CAND_OFFSET, RULES

HERE = Path(__file__).resolve().parent
TRACE_HALF = 5


def main() -> None:
    data = detector.load()
    cand = detector.band()
    s = detector.score(data, cand)
    sel = detector.arms(data)
    move = cand + CAND_OFFSET

    t0_of = data["t0"]
    games = data["game_id"][sel["reference"]]
    peak = move[s.argmax(1)]
    rule = detector.rule_of_row(data["category"])

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
        picked = {u: v[0] for u, v in by_rule.items()}
        if not all(peak[r] == t0 for r in picked.values()):
            continue
        control = [r for r in rows if data["is_control"][r] == 1]
        if not control or peak[control[0]] == t0:
            continue
        j = int(np.searchsorted(move, t0))
        margin = min(s[r, j] - np.delete(s[r], j).max() for r in picked.values())
        if best is None or margin > best["margin"]:
            best = {"game_id": g, "t0": t0, "margin": float(margin),
                    "rows": {u: int(v) for u, v in picked.items()},
                    "control_row": int(control[0]),
                    "reference_row": int([r for r in rows
                                          if data["is_control"][r] == 0
                                          and rule[r] == ""][0])}
    assert best, "no game satisfies the panel criteria"
    (HERE / "figure_game.json").write_text(json.dumps(best, indent=2) + "\n")
    print(json.dumps(best, indent=2))
    print(f"\nwrote {HERE / 'figure_game.json'}")


if __name__ == "__main__":
    main()
