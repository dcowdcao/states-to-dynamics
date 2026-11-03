"""Where the weights are for each row of `picks.csv`, and how each one has to be fed.

`pick.py` chooses a config per SAE type at the shared L0; this maps that choice onto a
checkpoint directory. The mapping is not one lookup because the eight picks come from three
places and two naming conventions:

  * Karvonen's five are published checkpoints, not in the git tree. `download_saes.sh` fetches
    them; they live on netscratch (see KARVONEN_ROOT). `picks.csv` names them by the
    `autoencoder_path` of the results table, and `matched()` checks each one's sparsity knob.
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
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from spade.common.paths import repo_root
from spade.common.study import data_root, study_dir
from spade.evaluation.karvonen_adapter import (
    StandardizedDictionary, load_karvonen_sae, load_standardization_params)

KARVONEN_ROOT = data_root() / "karvonen_saes"
PICKS = repo_root() / "results" / "l0_operating_point" / "picks.csv"
SWEEP_FULL = repo_root() / "karvonen_othello_sweep" / "sweep_full.csv"

KARVONEN_DIRS = {
    "gated/trainer32": "othello-trained_model-layer_5-2024-05-23/"
                       "othello-trained_model-layer_5-gated/trainer32",
    "gated_anneal/trainer25": "othello-trained_model-layer_5-2024-05-23/"
                              "othello-trained_model-layer_5-gated_anneal/trainer25",
    "p_anneal/trainer22": "othello-trained_model-layer_5-2024-05-23/"
                          "othello-trained_model-layer_5-p_anneal/trainer22",
    "standard/trainer25-v2": "othello-trained_model-layer_5-2024-05-23/"
                             "othello-trained_model-layer_5-standard/trainer25-v2",
    "top_k/trainer_61": "othello-trained_model-layer_5-2024-07-10_topk/"
                        "othello-trained_model-layer_5-top_k/trainer_61",
}

# Our three, seated at one seed under study_dir("multiseed"). picks.csv's CosineSpaDE metrics
# are its 5-seed mean; the weights are this seed's.
SEED = 42
OUR_DIRS = {
    "BatchTopK-k220": "autoencoders_batch_topk_l0_grid/trainer_10",
    "CosineSpaDE-beta1-gam0.05": "autoencoders_cosine_spade_sweep/trainer_0",
    "SpaDE-lam0.001-gam0.005": "autoencoders_l0_coverage/trainer_5",
}


def space_name(config: str) -> str:
    """A filename-safe space name for a pick. Karvonen configs carry a `/` (`gated/trainer32`),
    which silently becomes a directory component in any path built from it."""
    return "l0_" + config.replace("/", "-")


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
        return space_name(self.config)

    def encode(self, device: str):
        """This pick's encode function, standardized only if that is how it was trained.

        The rule lives with the field that decides it: every caller that feeds a pick
        activations has to get this right, and getting it wrong is silently wrong rather
        than merely inaccurate (`spade.spaces`).
        """
        ae, _ = load_karvonen_sae(str(self.ckpt_dir), device=device)
        if not self.standardize:
            return ae.to(device).encode
        params = load_standardization_params(str(self.ckpt_dir), device=device)
        return StandardizedDictionary(
            ae, torch.tensor(params["mean"], device=device), params["scale"]).to(device).encode


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
    """The pick's pinned checkpoint, asserted to be the run its config string names."""
    ckpt = study_dir("multiseed") / f"seed_{SEED}" / OUR_DIRS[config]
    trainer = json.loads((ckpt / "config.json").read_text())["trainer"]
    want = _params(config)
    # A pre-beta checkpoint has no `beta` key, and beta1 IS that checkpoint (DEFAULT_COSINE_BETA
    # = 1.0 is plain cosine); the beta=5 run at the same gamma is a different model.
    beta = want.pop("beta", None)
    assert (trainer["trainer_class"] == config.split("-")[0] + "Trainer"
            and trainer["seed"] == SEED
            and (beta is None or float(trainer.get("beta", 1.0)) == beta)
            and all(float(trainer[f]) == v for f, v in want.items())), \
        f"{ckpt} is not {config} at seed {SEED}: {trainer}"
    return ckpt


def _karvonen(config: str) -> Path:
    """His checkpoint for a pick, held to the sparsity knob of the sweep row it was picked from.

    A directory name is not an identity: `gated/` holds `trainer32` and `trainer32-v2` under
    one wandb_name, at L0 220 and 74.
    """
    path = KARVONEN_ROOT / KARVONEN_DIRS[config]
    sweep = pd.read_csv(SWEEP_FULL)
    row = sweep.set_index(sweep.group_dir.str.removeprefix("othello-trained_model-layer_5-")
                          + "/" + sweep.trainer_dir).loc[config]
    knob = json.loads((path / "config.json").read_text())["trainer"][row.hyperparameter]
    assert math.isclose(knob, row.hyperparameter_value, rel_tol=1e-6), \
        f"{config}: {path} has {row.hyperparameter} {knob}, its sweep row {row.hyperparameter_value}"
    return path


def matched() -> list[Matched]:
    picks = pd.read_csv(PICKS)
    out = []
    for _, r in picks.iterrows():
        karvonen = r.config in KARVONEN_DIRS
        path = _karvonen(r.config) if karvonen else _ours(r.config)
        assert (path / "ae.pt").exists(), f"{r.config}: no ae.pt at {path}"
        out.append(Matched(r.trainer_class, r.config, float(r.l0), path, not karvonen))
    return out
