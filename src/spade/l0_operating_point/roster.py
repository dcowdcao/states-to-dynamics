"""The eight L0-matched picks as a figure roster, and how they are styled.

A by-space figure needs a name, a family to take its colour from, and the L0 the checkpoint
was seated at -- all three in `picks.csv`. It does not need weights, so this never calls
`checkpoints.matched()`, which needs them on disk.

The canonical seed-42 roster sits wherever each family's default hyperparameters put it on the
sparsity axis, so a difference between two of those spaces confounds architecture with L0.
This roster is one checkpoint per family, each the nearest-L0 member of its own grid to the
shared target, which is what removes that confound.
"""

from dataclasses import dataclass

import pandas as pd

from spade.figures.paper_style import SPACE_STYLE
from spade.l0_operating_point.checkpoints import PICKS, space_name

# A pick is the same architecture as a canonical space, so it keeps that space's hue, marker
# and hatch (FIGURE_GUIDELINES 1.5) and only the L0 differs. `picks.csv` names the trainer
# class; `SPACE_STYLE` is keyed by the seed-42 config, and the two notations do not meet.
FAMILY = {
    "StandardTrainer": "StandardTrainer-othello",
    "PAnnealTrainer": "PAnnealTrainer-othello",
    "GatedSAETrainer": "GatedSAETrainer-othello",
    "GatedAnnealTrainer": "GatedAnnealTrainer-othello",
    "TrainerTopK": "TopKTrainer-othello",
    "BatchTopKTrainer": "BatchTopKTrainer-othello",
    "SpaDETrainer": "SpaDE-lam1e-03-gam1e-04",
    "CosineSpaDETrainer": "CosineSpaDETrainer-othello",
}


@dataclass(frozen=True)
class Space:
    name: str        # the npz stem these studies read, `l0_<config>`
    family: str      # the SPACE_STYLE key it borrows its identity from
    config: str
    l0: float


def _read() -> list[Space]:
    picks = pd.read_csv(PICKS)
    return [Space(space_name(r.config), FAMILY[r.trainer_class], r.config, float(r.l0))
            for _, r in picks.iterrows()]


ROSTER = _read()
SPACES = [s.name for s in ROSTER]
L0 = {s.name: s.l0 for s in ROSTER}


def styles() -> dict[str, dict]:
    """`SPACE_STYLE` plus one entry per L0-matched space.

    The tick label is the family name alone. Each pick's L0 is in `L0` here and in every
    figure's CSV, which is where a reader checks how well "matched" holds -- on the axis it
    is nine numbers competing with the nine the figure is actually about.
    """
    matched = {s.name: dict(SPACE_STYLE[s.family]) for s in ROSTER}
    return SPACE_STYLE | matched
