"""Step 2: results/eval_games.csv, the 500 t0-stratified games every detector figure scores.

`detector.stratified` over the eligible tree. This is the file's one writer; the figures here,
`changepoint_localization` and `move_legality.detection_auroc` read it, and the first two
assert their own draw against it (`detector.assert_eval_set`).

Usage: PYTHONPATH=src python analysis/violation_divergence/illegal_move_detector/eval_games.py
"""

import csv

import detector


def main() -> None:
    data = detector.load()
    sel = detector.arms(data)
    reference = sel["reference"][detector.stratified(data, sel)]
    detector.RESULTS.mkdir(parents=True, exist_ok=True)
    out = detector.RESULTS / "eval_games.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["game_id", "t0"])
        writer.writerows(sorted(zip(data["game_id"][reference].tolist(),
                                    data["t0"][reference].tolist())))
    print(out)


if __name__ == "__main__":
    main()
