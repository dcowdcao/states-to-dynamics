"""Do the SAE dictionaries keep what the residual stream has, and does any of them sharpen it?

One table, `results/localization_by_space.csv`: for each of the 11 representation spaces and
for the no-probe model baseline, how often the probe's argmax over t <= t0 lands on the
spliced move, where t0 ranks, and how far the illegal move outranks the same game's legal
control at t0. The README's per-space table cites these rows; `localization.py` draws the
event-locked curve behind them for one space at a time.

The rows are not equally informative by construction. `random_proj` is a ReLU of a dense
512 -> 8192 projection, near information-preserving, so it bounds what "a probe reads this
space" can show rather than nulling it; `random_topk` at L0 80 is the sparsity-matched null
for the k=80 dictionaries. They are not one gray dashed pair. The `model_baseline` row is not
a space at all: it is the model's own 1 - P(the move played), which needs no probe, and it is
the number the eleven have to beat.

Usage: python analysis/move_legality/space_comparison.py
"""

import pandas as pd

from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.figures.paper_style import SPACES
from spade.move_legality import localize
from spade.move_legality.model_baseline import NAME as BASELINE

RESULTS = repo_root() / "analysis" / "move_legality" / "results"
RULES = localize.RULES
SEED = 42
BASELINE_SCORE = "1-P(played)"


def available() -> list[str]:
    """SPACES in paper_style order -- reference first, untrained controls last."""
    have = {p.stem.removeprefix("p_illegal_")
            for p in (study_dir("move_legality") / "readout").glob("p_illegal_*.pt")}
    missing = [s for s in SPACES if s not in have]
    assert not missing, f"no read-out for {missing}; run spade.move_legality.readout"
    return list(SPACES)


def main() -> None:
    meta = localize.branches()
    selection = localize.arms(meta, SEED)

    rows = []
    for space in [*available(), BASELINE]:
        p2 = localize.read(space, "p_illegal")
        rows += localize.scalar_rows(space, BASELINE_SCORE if space == BASELINE else "P(illegal)",
                                     p2, meta, selection, ["control", "illegal", *RULES], SEED)
        if space == BASELINE:
            print(f"scored {space} (no probe, one score and no rule heads)", flush=True)
            continue
        p4 = localize.read(space, "proba4")
        for i, rule in enumerate(RULES, start=1):
            rows += localize.scalar_rows(space, f"P({rule})", p4[..., i], meta, selection,
                                         [rule], SEED)
        rows += localize.scalar_rows(space, "1-P(legal)", 1.0 - p4[..., 0], meta, selection,
                                     ["control", *RULES], SEED)
        print(f"scored {space}", flush=True)

    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv = RESULTS / "localization_by_space.csv"
    table.to_csv(csv, index=False)
    print(f"wrote {csv}")
    print(table[table["arm"] != "control"].to_string(index=False))


if __name__ == "__main__":
    main()
