"""Where the weights are for each row of `picks.csv`, and how each one has to be fed.

`pick.py` chooses a config per SAE type at the shared L0; this maps that choice onto a
checkpoint directory. The mapping is not one lookup because the eight picks come from three
places and two naming conventions:

  * Karvonen's five are published checkpoints, not in the git tree. `download_saes.sh` fetches
    them; they live on netscratch (see KARVONEN_ROOT). `picks.csv` names them by the
    `autoencoder_path` of the results table -- `gated/trainer32` is stored as `trainer32-v2`.
  * Our three come from the L0 grids on netscratch, named by `wandb_name` minus its
    `_trainer_N` suffix.
  * CosineSpaDE's picks read `beta1-...` because the eval ran after `karvonen/cosine-beta`
    added an inverse temperature. `DEFAULT_COSINE_BETA = 1.0` is plain cosine and reproduces
    every pre-beta checkpoint bitwise, so `CosineSpaDE-beta1-gamG` IS the pre-beta run at
    gamma G, whose own config.json has no `beta` key at all.

`standardize` is the load-bearing field. Ours were trained on `scale*(x - mean)` activations
and must be wrapped in `StandardizedDictionary`; Karvonen's were trained on the raw residual
stream and must not be. Feeding either the other's convention is silently wrong rather than
merely inaccurate (`spade.spaces`), so each entry carries its own.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from spade.common.paths import repo_root

KARVONEN_ROOT = Path("/n/netscratch/ba_lab/Lab/dcao/karvonen_saes")
GRID_ROOT = Path("/n/netscratch/ba_lab/Lab/dcao")
PICKS = repo_root() / "results" / "l0_operating_point" / "picks.csv"

# Karvonen's results table records `gated/trainer32`; the archive stores `trainer32-v2`.
KARVONEN_DIRS = {
    "gated/trainer32": "othello-trained_model-layer_5-2024-05-23/"
                       "othello-trained_model-layer_5-gated/trainer32-v2",
    "gated_anneal/trainer25": "othello-trained_model-layer_5-2024-05-23/"
                              "othello-trained_model-layer_5-gated_anneal/trainer25",
    "p_anneal/trainer22": "othello-trained_model-layer_5-2024-05-23/"
                          "othello-trained_model-layer_5-p_anneal/trainer22",
    "standard/trainer25-v2": "othello-trained_model-layer_5-2024-05-23/"
                             "othello-trained_model-layer_5-standard/trainer25-v2",
    "top_k/trainer_61": "othello-trained_model-layer_5-2024-07-10_topk/"
                        "othello-trained_model-layer_5-top_k/trainer_61",
}


@dataclass(frozen=True)
class Matched:
    """One SAE type at the shared L0."""
    trainer_class: str
    config: str
    l0: float
    ckpt_dir: Path
    standardize: bool

    @property
    def space(self) -> str:
        """A filename-safe space name. Karvonen configs carry a `/` (`gated/trainer32`),
        which silently becomes a directory component in any path built from it."""
        return "l0_" + self.config.replace("/", "-")


def _params(config: str) -> dict:
    """The numeric hyperparameters a pick's config string names.

    Matched on values, not on the string: `picks.csv` writes `lam0.001-gam0.005` where the
    checkpoint's own `wandb_name` writes `lam1e-03-gam5e-03`. Same run, two notations, and
    string equality silently finds neither.
    """
    keys = {"k": "k", "lam": "lambda_init", "gam": "gamma_reg", "beta": "beta"}
    out = {}
    for prefix, field in keys.items():
        m = re.search(rf"-{prefix}([\d.eE+-]+?)(?=-|$)", config)
        if m:
            out[field] = float(m.group(1))
    return out


def _ours(config: str) -> Path:
    """The grid checkpoint whose trainer hyperparameters equal the pick's."""
    trainer_class = config.split("-")[0] + "Trainer"
    want = _params(config)
    assert want, f"no hyperparameters parsed out of {config}"
    for cfg in GRID_ROOT.rglob("config.json"):
        if not (cfg.parent / "ae.pt").exists():
            continue
        try:
            trainer = json.loads(cfg.read_text()).get("trainer", {})
        except json.JSONDecodeError:
            continue
        if trainer.get("trainer_class") != trainer_class:
            continue
        # beta needs its own rule: a pre-beta checkpoint has no `beta` key, and beta1 IS that
        # checkpoint (DEFAULT_COSINE_BETA = 1.0 is plain cosine). Matching gamma alone would
        # accept the beta=5 run at the same gamma, which is a different model.
        beta = want.get("beta")
        have = trainer.get("beta")
        if beta is not None and not (float(have) == beta if have is not None else beta == 1.0):
            continue
        rest = {f: v for f, v in want.items() if f != "beta"}
        if all(trainer.get(f) is not None and float(trainer[f]) == v for f, v in rest.items()):
            return cfg.parent
    raise FileNotFoundError(f"no {trainer_class} checkpoint with {want}")


def matched() -> list[Matched]:
    picks = pd.read_csv(PICKS)
    out = []
    for _, r in picks.iterrows():
        karvonen = r.config in KARVONEN_DIRS
        path = KARVONEN_ROOT / KARVONEN_DIRS[r.config] if karvonen else _ours(r.config)
        assert (path / "ae.pt").exists(), f"{r.config}: no ae.pt at {path}"
        out.append(Matched(r.trainer_class, r.config, float(r.l0), path, not karvonen))
    return out
