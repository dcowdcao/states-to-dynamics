"""Regenerate the complete record of Karvonen's OthelloGPT SAE sweep from his released CSVs.

Hyperparameters are read from each checkpoint's config.json (and, where the config omits it,
from the ae.pt encoder shape), not inferred from the trainer index. The index-decoding rule the
earlier pass derived is kept only as a cross-check and its verdict goes into the report.

Emits sweep_full.csv (one row per SAE), best_per_type.csv (every selection criterion),
controls.csv (random-model and raw-MLP context rows), and REPORT.md. Every number in REPORT.md
is interpolated from these frames; none is typed by hand.
"""

import argparse
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ACT_DIM = 512
EXPANSIONS = (8, 16)

TRAINED = "06-10-othello-trained-results_with_07-10_topk.csv"
TRAINED_SUBSET = "othello-6-10-trained-f1_results.csv"
RANDOM_MODEL = "othello-6-10-random-f1_results.csv"
MLP_NEURONS = "othello-6-10-mlp-acts-f1_results.csv"

DEFAULT_CONFIG_ROOT = "/n/netscratch/ba_lab/Lab/dcao/karvonen_released"
# group dir (as it appears in the results CSV) -> its parent release dir on disk
CONFIG_GROUPS = {
    "othello-trained_model-layer_5-standard": "othello-trained_model-layer_5-2024-05-23",
    "othello-trained_model-layer_5-gated": "othello-trained_model-layer_5-2024-05-23",
    "othello-trained_model-layer_5-p_anneal": "othello-trained_model-layer_5-2024-05-23",
    "othello-trained_model-layer_5-gated_anneal": "othello-trained_model-layer_5-2024-05-23",
    "othello-trained_model-layer_5-top_k": "othello-trained_model-layer_5-2024-07-10_topk",
}
SPARSITY_FIELDS = ("l1_penalty", "sparsity_penalty", "k")


def _grid(lo, hi, n):
    """circuits/sae_training/train_saes_parallel.py builds grids with t.linspace().tolist()."""
    return np.linspace(lo, hi, n, dtype=np.float32).astype(float)


# The rule the earlier pass inferred, kept as a cross-check: (trainer_class, sweep) -> grid
# indexed by trainer_index % SPARSITY_STEPS.
INFERRED_GRIDS = {
    ("StandardTrainer", "orig"): _grid(0.035, 0.9, 20),
    ("StandardTrainer", "v2"): _grid(0.035, 0.08, 20),
    ("GatedSAETrainer", "orig"): _grid(0.2, 1.0, 20),
    ("GatedSAETrainer", "v2"): _grid(1.0, 2.0, 20),
    ("PAnnealTrainer", "orig"): _grid(0.025, 0.05, 20),
    ("GatedAnnealTrainer", "orig"): _grid(0.15, 1.0, 20),
    ("TrainerTopK", "orig"): np.arange(10, 410, 10, dtype=float),
}
SPARSITY_STEPS = {"TrainerTopK": 40}
GRID_TOLERANCE = 1e-6  # config.json stores the float32 grid value widened to float64

TYPE_LABEL = {
    "StandardTrainer": "Standard",
    "PAnnealTrainer": "Standard + p-annealing",
    "GatedSAETrainer": "Gated",
    "GatedAnnealTrainer": "Gated + p-annealing",
    "TrainerTopK": "TopK",
}
TYPE_ORDER = list(TYPE_LABEL)

MINE_YOURS = "games_batch_to_state_stack_mine_yours_BLRRC_"
BLANK_MASK = "games_batch_to_state_stack_mine_yours_blank_mask_BLRRC_"
VALID_MOVES = "games_batch_to_valid_moves_BLRRC_"

METRIC_SOURCE = {
    "recon_f1": MINE_YOURS + "best_f1_score_per_class",
    "coverage": MINE_YOURS + "best_average_f1",
    "recon_f1_bm": BLANK_MASK + "best_f1_score_per_class",
    "coverage_bm": BLANK_MASK + "best_average_f1",
    "recon_f1_moves": VALID_MOVES + "best_f1_score_per_class",
    "coverage_moves": VALID_MOVES + "best_average_f1",
}
# Blank-masked coverage still divides by all 192 (square, class) entries while the 64 blank-class
# entries are identically zero, so it is 128/192 of the honest mine/yours mean.
BLANK_MASK_RESCALE = 192 / 128

CRITERIA = {
    "coverage": "Coverage, mine/yours (all 3 classes)",
    "coverage_bm": "Coverage, blank-masked (mine/yours only)",
    "recon_f1": "Board reconstruction F1, mine/yours",
    "recon_f1_bm": "Board reconstruction F1, blank-masked",
}
# Populated at both dict sizes by all five types; brackets the SpaDE study's L0 = 80 TopK point.
MATCHED_BAND = (60.0, 200.0)
MATCHED_DICT_SIZE = 8192
L0_BANDS = [(10, 40), (40, 60), (60, 100), (100, 160), (160, 260), (260, 400), (400, 600)]

# Every collapsed row sits far below this; every surviving row sits far above it.
ALIVE_FVE = 0.2


def read_configs(config_root: Path) -> pd.DataFrame:
    rows = []
    for group, parent in CONFIG_GROUPS.items():
        group_dir = config_root / parent / group
        if not group_dir.is_dir():
            raise FileNotFoundError(f"released checkpoints not found at {group_dir}")
        for d in sorted(group_dir.glob("trainer*")):
            cfg = json.loads((d / "config.json").read_text())["trainer"]
            field = next(f for f in SPARSITY_FIELDS if f in cfg)
            rows.append({
                "group_dir": group, "trainer_dir": d.name,
                "trainer_class": cfg["trainer_class"], "hyperparameter": field,
                "hyperparameter_value": float(cfg[field]),
                "dict_size": cfg.get("dict_size") or _dict_size_from_weights(d),
                "activation_dim": cfg.get("activation_dim") or ACT_DIM,
                "lr": cfg["lr"], "seed": cfg.get("seed"), "steps": cfg.get("steps"),
            })
    return pd.DataFrame(rows)


def _dict_size_from_weights(trainer_dir: Path) -> int:
    """The released standard configs omit dict_size; the encoder shape is the direct measurement."""
    weights = torch.load(trainer_dir / "ae.pt", map_location="cpu", weights_only=True)
    return int(weights["encoder.weight"].shape[0])


def load(data_dir: Path, cfg: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(data_dir / TRAINED)
    d = pd.DataFrame(index=raw.index)
    d["trainer_class"] = raw.trainer_class
    d["type"] = raw.trainer_class.map(TYPE_LABEL)
    d["group_dir"] = raw.autoencoder_group_path.str.rstrip("/").str.split("/").str[-1]
    d["trainer_dir"] = raw.autoencoder_path.str.rstrip("/").str.split("/").str[-1]
    d["sweep"] = np.where(d.trainer_dir.str.endswith("-v2"), "v2", "orig")
    d["trainer_idx"] = d.trainer_dir.str.extract(r"trainer_?(\d+)")[0].astype(int)

    d = d.merge(cfg.drop(columns="trainer_class"), on=["group_dir", "trainer_dir"], how="left",
                validate="one_to_one")
    if d.hyperparameter_value.isna().any():
        missing = d[d.hyperparameter_value.isna()].trainer_dir.tolist()
        raise ValueError(f"no released config for {missing}")
    d["expansion"] = d.dict_size // d.activation_dim

    for c in ("l0", "num_alive_features", "frac_variance_explained", "frac_recovered",
              "frac_alive", "l2_ratio", "cossim"):
        d[c] = raw[c]
    for short, col in METRIC_SOURCE.items():
        d[short] = raw[col]
    d["coverage_bm_rescaled"] = d.coverage_bm * BLANK_MASK_RESCALE
    d["status"] = np.where(d.frac_variance_explained > ALIVE_FVE, "alive", "collapsed")
    return d.sort_values(["trainer_class", "sweep", "dict_size", "hyperparameter_value"])


def decoding_check(cfg: pd.DataFrame) -> pd.DataFrame:
    """Does `trainer index = expansion_index * S + sparsity_index` reproduce the config values?"""
    c = cfg.copy()
    c["sweep"] = np.where(c.trainer_dir.str.endswith("-v2"), "v2", "orig")
    idx = c.trainer_dir.str.extract(r"trainer_?(\d+)")[0].astype(int)
    steps = c.trainer_class.map(SPARSITY_STEPS).fillna(20).astype(int)
    c["expansion_from_idx"] = [EXPANSIONS[i] for i in idx // steps]
    c["value_from_idx"] = [INFERRED_GRIDS[(t, s)][i]
                           for t, s, i in zip(c.trainer_class, c.sweep, idx % steps)]
    c["value_gap"] = (c.value_from_idx - c.hyperparameter_value).abs()
    c["dict_matches"] = c.expansion_from_idx * c.activation_dim == c.dict_size
    return c.groupby(["trainer_class", "sweep"]).agg(
        n=("value_gap", "size"), max_value_gap=("value_gap", "max"),
        dict_size_mismatches=("dict_matches", lambda s: int((~s).sum())))


def superset_check(data_dir: Path, n_rows: int) -> dict:
    """The released 240-row CSV must be a value-identical subset of the 280-row one."""
    sub = pd.read_csv(data_dir / TRAINED_SUBSET)
    full = pd.read_csv(data_dir / TRAINED)
    key_full = full.autoencoder_path.str.rstrip("/")
    key_sub = sub.autoencoder_path.str.rstrip("/")
    shared_rows = sorted(set(key_full) & set(key_sub))
    a, b = full.set_index(key_full), sub.set_index(key_sub)
    shared_cols = [c for c in b.columns if c in a.columns and not c.startswith("Unnamed")]
    conflicts = 0
    for c in shared_cols:
        x, y = a.loc[shared_rows, c], b.loc[shared_rows, c]
        if x.dtype.kind in "fi":
            same = np.isclose(x.astype(float), y.astype(float), rtol=1e-9) | (x.isna() & y.isna())
        else:
            same = x.astype(str) == y.astype(str)
        conflicts += int((~same).sum())
    return {"subset_rows": len(sub), "full_rows": len(full), "shared_rows": len(shared_rows),
            "shared_cols": len(shared_cols), "conflicts": conflicts,
            "extra_rows": n_rows - len(shared_rows)}


def controls(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path, kind in [(RANDOM_MODEL, "random-weight model"), (MLP_NEURONS, "raw MLP neurons")]:
        raw = pd.read_csv(data_dir / path)
        c = pd.DataFrame(index=raw.index)
        c["control"] = kind
        c["name"] = raw.autoencoder_path.str.rstrip("/").str.split("/").str[-1]
        c["trainer_class"] = raw.trainer_class
        for col in ("l0", "num_alive_features", "frac_variance_explained", "frac_recovered"):
            c[col] = raw[col]
        for short, col in METRIC_SOURCE.items():
            c[short] = raw[col]
        c["coverage_bm_rescaled"] = c.coverage_bm * BLANK_MASK_RESCALE
        frames.append(c)
    return pd.concat(frames, ignore_index=True)


def selections(alive: pd.DataFrame) -> pd.DataFrame:
    out = []
    for metric in CRITERIA:
        out.append(alive.loc[alive.groupby("trainer_class")[metric].idxmax()]
                   .assign(criterion=f"max {metric}", scope="whole sweep"))
    lo, hi = MATCHED_BAND
    band = alive[(alive.dict_size == MATCHED_DICT_SIZE) & alive.l0.between(lo, hi)]
    for metric in CRITERIA:
        out.append(band.loc[band.groupby("trainer_class")[metric].idxmax()]
                   .assign(criterion=f"max {metric}",
                           scope=f"dict {MATCHED_DICT_SIZE}, L0 {lo:.0f}-{hi:.0f}"))
    for blo, bhi in L0_BANDS:
        b = alive[alive.l0.between(blo, bhi, inclusive="left")]
        if b.empty:
            continue
        for metric in CRITERIA:
            out.append(b.loc[b.groupby("trainer_class")[metric].idxmax()]
                       .assign(criterion=f"max {metric}", scope=f"L0 {blo}-{bhi}"))
    cols = ["criterion", "scope", "type", "sweep", "trainer_dir", "expansion", "dict_size",
            "hyperparameter", "hyperparameter_value", "l0", "num_alive_features",
            "frac_variance_explained", "frac_recovered", "recon_f1", "coverage", "recon_f1_bm",
            "coverage_bm", "coverage_bm_rescaled", "recon_f1_moves", "coverage_moves"]
    return pd.concat(out, ignore_index=True)[cols]


def q(v, digits=4):
    if pd.isna(v):
        return "--"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{float(v):.{digits}g}"


def md_table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(header)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|",
                      *(line(r) for r in rows)])


SWEEP_HEADER = ["trainer dir", "sweep", "exp", "dict", "hyperparam", "value", "L0", "alive feat",
                "FVE", "recon F1", "coverage", "recon F1 bm", "cov bm", "cov bm x1.5", "status"]


def sweep_row(r) -> list[str]:
    return [r.trainer_dir, r.sweep, str(r.expansion), str(r.dict_size), r.hyperparameter,
            q(r.hyperparameter_value), q(r.l0), str(r.num_alive_features),
            q(r.frac_variance_explained), q(r.recon_f1), q(r.coverage), q(r.recon_f1_bm),
            q(r.coverage_bm), q(r.coverage_bm_rescaled),
            "collapsed" if r.status == "collapsed" else ""]


PICK_HEADER = ["type", "trainer dir", "sweep", "hyperparam", "value", "dict", "L0", "alive feat",
               "coverage", "cov bm", "cov bm x1.5", "recon F1", "recon F1 bm"]


def pick_row(r) -> list[str]:
    return [r.type, r.trainer_dir, r.sweep, r.hyperparameter, q(r.hyperparameter_value),
            str(r.dict_size), q(r.l0), str(r.num_alive_features), q(r.coverage), q(r.coverage_bm),
            q(r.coverage_bm_rescaled), q(r.recon_f1), q(r.recon_f1_bm)]


def picks_table(frame: pd.DataFrame, metric: str) -> str:
    best = frame.loc[frame.groupby("trainer_class")[metric].idxmax()].sort_values(
        metric, ascending=False)
    return md_table([pick_row(r) for _, r in best.iterrows()], PICK_HEADER)


def reflow(text: str, width: int = 96) -> str:
    """Rewrap prose paragraphs; leave tables, headings and blank lines alone.

    Interpolated numbers change line lengths, so the template's own line breaks land wherever.
    """
    atomic = lambda s: not s.strip() or s.startswith(("|", "#"))
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        if atomic(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        bullet = lines[i].startswith("- ")
        para = [lines[i]]
        i += 1
        while (i < len(lines) and not atomic(lines[i]) and not lines[i].startswith("- ")
               and lines[i].startswith("  ") == bullet):
            para.append(lines[i])
            i += 1
        out.append(textwrap.fill(" ".join(s.strip() for s in para), width,
                                 subsequent_indent="  " if bullet else "",
                                 break_long_words=False, break_on_hyphens=False))
    return "\n".join(out)


def band_matrix(alive: pd.DataFrame) -> str:
    header = ["L0 band", "metric"] + [TYPE_LABEL[t] for t in TYPE_ORDER]
    rows = []
    for blo, bhi in L0_BANDS:
        b = alive[alive.l0.between(blo, bhi, inclusive="left")]
        if b.empty:
            continue
        for metric in CRITERIA:
            cells = []
            for t in TYPE_ORDER:
                g = b[b.trainer_class == t]
                cells.append("--" if g.empty else
                             f"{q(g.loc[g[metric].idxmax()][metric])} ({g.loc[g[metric].idxmax()].trainer_dir})")
            rows.append([f"{blo}-{bhi}", metric, *cells])
    return md_table(rows, header)


def topk_gap(cfg: pd.DataFrame, d: pd.DataFrame) -> str:
    tk = cfg[cfg.trainer_class == "TrainerTopK"].copy()
    tk["evaluated"] = tk.trainer_dir.isin(d.trainer_dir)
    rows = []
    for evaluated in (True, False):
        for dict_size in sorted(tk.dict_size.unique()):
            s = tk[(tk.evaluated == evaluated) & (tk.dict_size == dict_size)]
            ks = ", ".join(str(int(k)) for k in sorted(s.hyperparameter_value))
            rows.append(["yes" if evaluated else "no", str(int(dict_size)), str(len(s)), ks])
    return md_table(rows, ["in published CSV", "dict", "n", "k"])


def report(d, ctl, cfg, prov, check) -> str:
    alive = d[d.status == "alive"]
    dead = d[d.status == "collapsed"]
    lo, hi = MATCHED_BAND
    band = alive[(alive.dict_size == MATCHED_DICT_SIZE) & alive.l0.between(lo, hi)]
    rec = band.loc[band.groupby("trainer_class").coverage_bm.idxmax()].sort_values(
        "coverage_bm", ascending=False)
    order = lambda frame, metric: list(
        frame.loc[frame.groupby("trainer_class")[metric].idxmax()]
        .sort_values(metric, ascending=False).trainer_class)
    if order(band, "coverage_bm") != order(alive, "coverage_bm"):
        raise ValueError("matched-band and whole-sweep type orderings diverged; rewrite the claim")
    rank = {t: i for i, t in enumerate(order(alive, "coverage_bm"))}
    flip = pd.Series([rank[t] for t in order(alive, "recon_f1")]).corr(
        pd.Series(range(len(rank))), method="spearman")

    mlp5 = ctl[(ctl.control == "raw MLP neurons") & (ctl.name == "layer_5")].iloc[0]
    mlp7 = ctl[(ctl.control == "raw MLP neurons") & (ctl.name == "layer_7")].iloc[0]
    rand = ctl[ctl.control == "random-weight model"]
    rand_best = rand.loc[rand.coverage.idxmax()]
    rescale_gap = float((alive.coverage_bm_rescaled - alive.coverage).abs().mean())

    top_cov = alive.loc[alive.coverage.idxmax()]
    top_covbm = alive.loc[alive.coverage_bm.idxmax()]
    top_f1 = alive.loc[alive.recon_f1.idxmax()]
    fr = alive.frac_recovered
    std = alive[alive.trainer_class == "StandardTrainer"]
    topk80 = alive[(alive.trainer_class == "TrainerTopK") & (alive.dict_size == MATCHED_DICT_SIZE)
                   & (alive.l0 == 80)].iloc[0]
    seam = lambda t, s, ds, i: d[(d.trainer_class == t) & (d.sweep == s) & (d.dict_size == ds)
                                 & (d.trainer_idx % 20 == i)].iloc[0]

    parts = [f"""# Karvonen's OthelloGPT SAE sweep: the complete record

280 evaluated SAEs, five trainer types, layer 5 of OthelloGPT. Metrics from
`external/Karvonen_Research/autoencoders/saved_data/{TRAINED}`; hyperparameters read directly
from the {len(cfg)} released checkpoints' `config.json`. Regenerated by `build_report.py`; no
number below was typed by hand.

## The answer: one config per type

Criterion: **blank-masked coverage** (`..._blank_mask_BLRRC_best_average_f1`), at matched
dictionary size {MATCHED_DICT_SIZE} and matched sparsity L0 in [{lo:.0f}, {hi:.0f}].
Reasoning in *Which criterion, and why*. Karvonen defines no composite selection score of his
own -- confirmed by reading his code -- so the criterion is ours, and it is stated rather than
buried in a weighted sum.

{md_table([pick_row(r) for _, r in rec.iterrows()], PICK_HEADER)}

`cov bm x1.5` puts the blank-masked column back on the same 0-1 scale as the unmasked one; see
trap 3. All four board metrics run the same way: higher is better.

**The ordering is not an artifact of the band.** Ranking each type by its best blank-masked
coverage anywhere in the sweep, at either dictionary size, gives the same five-way order
(`build_report.py` raises if that ever stops being true). It *is* an artifact of the metric: rank
the same five types by reconstruction F1 and the order all but inverts, Spearman {q(flip, 2)}.

## What the four metrics are

Karvonen scores each SAE against three board-state functions. Two are used here; the third,
`games_batch_to_valid_moves`, is in the CSV and summarised at the end.

- `mine_yours` -- each square carries one of 3 classes (blank / mine / yours): 64 x 3 = 192
  board-state properties.
- `mine_yours_blank_mask` -- the same, scored only on squares that actually hold a disc.

For each function, two numbers, both taken after an internal sweep over 11 activation thresholds:

- **coverage** (`best_average_f1`, `circuits/f1_analysis.py:14-28`): for each of the 192
  properties, take the F1 of the single best feature for it; average over properties. "Does the
  dictionary contain a dedicated feature for each thing the board can be doing?" Karvonen's own
  plot axis calls this *Coverage*.
- **board reconstruction F1** (`best_f1_score_per_class`,
  `circuits/eval_board_reconstruction.py:307-331`): OR together the labels of every
  simultaneously-active feature to rebuild the board, then micro-F1 over all (board, square,
  class) cells. "Can you read the board off the dictionary?"

They measure different things and they do not agree: across the {len(alive)} surviving SAEs their
Spearman correlation is {q(alive.coverage.corr(alive.recon_f1, method='spearman'))}.

## Read this before you rank anything

**1. {len(dead)} of the 280 rows are collapsed, not merely sparse.** All of them are
`StandardTrainer`, all from the original sweep, all at l1_penalty >= {q(dead.hyperparameter_value.min())}.
The original othello L1 grid ran to 0.9 while the corrected one stops at 0.08, so from the fourth
grid point on the SAE trains to nothing. {int((d.l0 == 0).sum())} have L0 and
`frac_variance_explained` exactly 0. They are flagged `collapsed` in `sweep_full.csv` and excluded
from every selection here. Note their `num_alive_features` is still
{int(dead.num_alive_features.min())}-{int(dead.num_alive_features.max())}: that column does not go
to zero on a dead SAE.

**2. `-v2` rows are a second sweep, and the two grids overlap at the seams.** On 2024-05-26 the
standard and gated grids were re-run: standard `linspace(0.035, 0.9, 20)` ->
`linspace(0.035, 0.08, 20)`, gated `linspace(0.2, 1.0, 20)` -> `linspace(1.0, 2.0, 20)`. p_anneal
and gated_anneal never were, which is why only those two types carry `-v2` rows. The released
configs confirm this directly -- `trainer19-v2` records l1_penalty
{q(seam('StandardTrainer', 'v2', 4096, 19).hyperparameter_value)} against `trainer1`'s
{q(seam('StandardTrainer', 'orig', 4096, 1).hyperparameter_value)} -- and the grids share
endpoints, so a few configs appear twice at different seeds:

{md_table([
    ["standard, l1 = 0.035", "trainer0", q(seam('StandardTrainer', 'orig', 4096, 0).l0),
     "trainer0-v2", q(seam('StandardTrainer', 'v2', 4096, 0).l0)],
    ["standard, l1 ~ 0.08", "trainer1", q(seam('StandardTrainer', 'orig', 4096, 1).l0),
     "trainer19-v2", q(seam('StandardTrainer', 'v2', 4096, 19).l0)],
    ["gated, l1 = 1.0", "trainer19", q(seam('GatedSAETrainer', 'orig', 4096, 19).l0),
     "trainer0-v2", q(seam('GatedSAETrainer', 'v2', 4096, 0).l0)],
], ["overlapping config (dict 4096)", "orig dir", "L0", "-v2 dir", "L0"])}

Treat those as replicate pairs; do not concatenate the two sweeps into one grid.

**3. The blank-masked coverage column sits on a different scale from the unmasked one.**
`circuits/othello_utils.py:119` zeroes the *blank class channel* of the labels, but
`othello_mine_yours_blank_mask_config` (`circuits/chess_utils.py:772-778`) leaves
`one_hot_mask_idx` at `None`, and that field is the only thing that would shrink the denominator
in `f1_analysis.py:21-24`. So all 64 blank-class F1 entries are identically zero while the mean is
still taken over 192 entries. The column is mechanically capped at 128/192 = 0.6667 and its
absolute value is **not** comparable to the unmasked column. (The chess masked configs *do* set
`one_hot_mask_idx=6` and are unaffected.) Two consequences: the cap is the same constant on every
row, so **ranking within the column is untouched**; and multiplying by {BLANK_MASK_RESCALE:.1f}
restores the unmasked scale, which is the `cov bm x1.5` column -- it lands within
{q(rescale_gap, 2)} of unmasked coverage on average across surviving rows, the check that the
constant is right.

**4. `frac_recovered` is saturated and ranks nothing.** Across the {len(alive)} surviving SAEs it
spans {q(fr.min(), 6)} to {q(fr.max(), 6)}, with {int(((fr - 1).abs() < 1e-3).sum())} of them
within 0.001 of 1.000. It is also computed from a single batch of 25 sequences against a
zero-ablation baseline. Do not put it in a selection rule, and do not report it as evidence of
quality.

**5. `frac_alive` is not the fraction of live features.** On every non-TopK row it equals exactly
`l0 / dict_size` -- the mean fraction of the dictionary active *per token*, mislabelled. On every
TopK row it equals exactly `1 / dict_size`, a constant, because a later version of the accumulator
flattened both batch dims before `.any()`. The merged CSV therefore mixes two incompatible
definitions in one column. Use `num_alive_features` (fired at least once in ~29.5k token
activations); it is the only honest aliveness statistic in the file.

## Context: the floor and the neuron bar

{md_table([
    ["random-weight OthelloGPT, best of 40 SAEs", q(rand_best.l0), str(int(rand_best.num_alive_features)),
     q(rand_best.coverage), q(rand_best.coverage_bm), q(rand_best.coverage_bm_rescaled),
     q(rand_best.recon_f1), q(rand_best.recon_f1_bm)],
    ["raw MLP neurons, layer 5 (2048, no SAE)", q(mlp5.l0), str(int(mlp5.num_alive_features)),
     q(mlp5.coverage), q(mlp5.coverage_bm), q(mlp5.coverage_bm_rescaled),
     q(mlp5.recon_f1), q(mlp5.recon_f1_bm)],
    ["best trained SAE, by coverage", q(top_cov.l0), str(int(top_cov.num_alive_features)),
     q(top_cov.coverage), q(top_cov.coverage_bm), q(top_cov.coverage_bm_rescaled),
     q(top_cov.recon_f1), q(top_cov.recon_f1_bm)],
    ["best trained SAE, by blank-masked coverage", q(top_covbm.l0), str(int(top_covbm.num_alive_features)),
     q(top_covbm.coverage), q(top_covbm.coverage_bm), q(top_covbm.coverage_bm_rescaled),
     q(top_covbm.recon_f1), q(top_covbm.recon_f1_bm)],
    ["best trained SAE, by reconstruction F1", q(top_f1.l0), str(int(top_f1.num_alive_features)),
     q(top_f1.coverage), q(top_f1.coverage_bm), q(top_f1.coverage_bm_rescaled),
     q(top_f1.recon_f1), q(top_f1.recon_f1_bm)],
], ["row", "L0", "alive feat", "coverage", "cov bm", "cov bm x1.5", "recon F1", "recon F1 bm"])}

*The floor is high.* SAEs trained on a **randomly initialised** OthelloGPT reach coverage
{q(rand_best.coverage)} and reconstruction F1 {q(rand_best.recon_f1)}. Any claim about board
representations has to clear that, not zero. (Karvonen's random-model sweep is standard-trainer
only and inherits the same broken L1 grid, so {int((rand.l0 == 0).sum())} of its 40 rows are dead;
the row above is the best survivor.)

*No SAE in the sweep beats raw neurons on coverage.* Layer-5 MLP neurons -- 2048 of them, read
directly, L0 {q(mlp5.l0)} -- reach coverage {q(mlp5.coverage)}, above the best SAE
({q(top_cov.coverage)}). On reconstruction F1 the best SAE ({q(top_f1.recon_f1)}) does beat the
neurons ({q(mlp5.recon_f1)}). Caveat: the neurons are the MLP hidden layer while the SAEs sit on
the residual stream, so this is a bar rather than a controlled ablation -- but it is the bar that
matters when the pitch is "the dictionary is more interpretable than the neurons".

## Best per type, one criterion at a time

Nothing is aggregated across these four tables. Each is the argmax of one metric over surviving
rows, both dictionary sizes, both sweeps.
"""]

    for metric, label in CRITERIA.items():
        parts.append(f"\n### {label} -- `{metric}`\n\n{picks_table(alive, metric)}\n")

    parts.append(f"""
The tables disagree, and systematically:

{md_table([[TYPE_LABEL[t],
            q(alive[alive.trainer_class == t].l0.corr(alive[alive.trainer_class == t].coverage, method='spearman')),
            q(alive[alive.trainer_class == t].l0.corr(alive[alive.trainer_class == t].recon_f1, method='spearman')),
            q(alive[alive.trainer_class == t].coverage.corr(alive[alive.trainer_class == t].recon_f1, method='spearman')),
            str(len(alive[alive.trainer_class == t]))]
           for t in TYPE_ORDER],
          ["type", "rho(L0, coverage)", "rho(L0, recon F1)", "rho(coverage, recon F1)", "n alive"])}

(Spearman over surviving rows of that type, both dictionary sizes pooled; n in the last column.)

Coverage and reconstruction F1 peak at different sparsities, and for p-annealed standard SAEs they
move in opposite directions with L0 entirely. Reconstruction F1 rewards density: the sweep-wide
maximum ({q(top_f1.recon_f1)}) belongs to an L0-{q(top_f1.l0)} standard SAE whose coverage,
{q(top_f1.coverage)}, is barely above the random-model floor of {q(rand_best.coverage)}. That is
the clearest single reason not to rank SAE families on reconstruction F1.

Masked and unmasked also disagree at the top, and only there: unmasked coverage puts
{TYPE_LABEL[top_cov.trainer_class]} first, blank-masked coverage puts
{TYPE_LABEL[top_covbm.trainer_class]} first. The flip is exactly the blank class being counted.

## Best per type at matched L0

Same argmax, one L0 band at a time, both dictionary sizes pooled. `--` means that type has no
surviving config in the band -- itself a finding, since it means the families never covered the
same sparsity range and any unconstrained comparison is partly a comparison of where each grid
happened to land.

{band_matrix(alive)}

## Which criterion, and why

**Karvonen has no composite selection score.** In his repo every `best_*` is an argmax *within*
one SAE (over the 11 activation thresholds, or over features); the only cross-config ranking
anywhere is a `print_top_n_values(df, key, n)` in `circuits/f1_analysis.ipynb` that sorts by one
metric column at a time to quote maxima in the paper's results text. Every figure is a scatter of
one metric against L0, coloured by trainer class, with the trade-off left to the reader's eye.
Inventing a weighted score would be putting our thumb on his scale, so we do not.

That leaves choosing a criterion and defending it. The recommendation above uses **blank-masked
coverage at matched dictionary size and matched L0**:

*Coverage over reconstruction F1*, because reconstruction F1 is a decodability test a dense
uninterpretable basis passes. It is maximised at the densest end of the grid
(rho(L0, recon F1) = {q(std.l0.corr(std.recon_f1, method='spearman'))} for standard SAEs) and its
sweep-wide winner has near-floor coverage. Coverage asks the question a dictionary exists to
answer: is there one feature per board-state property.

*Blank-masked over unmasked*, because "this square is empty" is the cheap third of the unmasked
label set -- early-game boards are mostly empty and blankness is close to a function of move
count. Removing it is the entire point of the masked config. The masked column's denominator bug
(trap 3) rescales every row by one constant and so cannot distort a ranking, and the switch does
change the answer at the top.

*Matched dict {MATCHED_DICT_SIZE} and matched L0 {lo:.0f}-{hi:.0f}*, because the SpaDE study
trains every architecture at dict 8192 and evaluates its TopK and BatchTopK baselines near L0 80,
and because the families' grids do not span the same sparsity range -- an unconstrained per-type
maximum partly rewards whichever grid happened to reach the most favourable L0.

**What this does not license.** These are Karvonen's published numbers from his evaluation code.
The SpaDE study re-implements coverage and board reconstruction with a train/test split for the
feature-to-property assignment and a 60-square (non-middle) mask, and retrains its own baselines.
At a nominally identical config -- TopK, dict {MATCHED_DICT_SIZE}, L0 80 -- this CSV gives
coverage {q(topk80.coverage)} and reconstruction F1 {q(topk80.recon_f1)}, which the study's own
run of the same trainer does not reproduce. Use this sweep to pick *hyperparameters to retrain and
re-evaluate in house*; do not paste these numbers into a table beside SpaDE's.

## Checkpoints released but never evaluated

The TopK group ships {int((cfg.trainer_class == 'TrainerTopK').sum())} checkpoints; the published
CSV evaluates {int((d.trainer_class == 'TrainerTopK').sum())} of them. The unevaluated half has
weights on disk and no metrics, so it is free headroom if a k between the published points matters:

{topk_gap(cfg, d)}

## The full sweep

One row per evaluated SAE, all {len(d)}. Sorted by dictionary size then by the sparsity
hyperparameter, which is read from `config.json` (not decoded from the directory index).
`FVE` is `frac_variance_explained`, `bm` is blank-masked, `cov bm x1.5` is the trap-3 rescale.
""")

    for t in TYPE_ORDER:
        g = d[d.trainer_class == t]
        parts.append(f"\n### {TYPE_LABEL[t]} (`{t}`) -- {len(g)} SAEs\n")
        for sweep in ("orig", "v2"):
            s = g[g.sweep == sweep]
            if s.empty:
                continue
            tag = ("`-v2` re-sweep (2024-05-26)" if sweep == "v2"
                   else f"original sweep -- `{s.group_dir.iloc[0]}`")
            n_dead = int((s.status == "collapsed").sum())
            note = f" -- {n_dead} collapsed rows, kept and flagged" if n_dead else ""
            parts.append(f"\n**{tag}**{note}\n\n")
            body = s.sort_values(["dict_size", "hyperparameter_value"])
            parts.append(md_table([sweep_row(r) for _, r in body.iterrows()], SWEEP_HEADER) + "\n")

    moves = alive.loc[alive.groupby("trainer_class").coverage_moves.idxmax()].sort_values(
        "coverage_moves", ascending=False)
    parts.append(f"""
## The third board function

`games_batch_to_valid_moves` (is this square a legal move?) is in the CSV and used in no selection
above. Best per type by its coverage, for reference:

{md_table([[r.type, r.trainer_dir, str(r.dict_size), q(r.l0), q(r.coverage_moves), q(r.recon_f1_moves)]
           for _, r in moves.iterrows()],
          ["type", "trainer dir", "dict", "L0", "coverage (moves)", "recon F1 (moves)"])}

Raw layer-5 MLP neurons score {q(mlp5.coverage_moves)} / {q(mlp5.recon_f1_moves)} here, and layer
7 neurons reach {q(mlp7.coverage_moves)} / {q(mlp7.recon_f1_moves)} -- legal-move information is
concentrated late in the model, unlike board state. Its denominator is 64 (one class), so it is on
yet another scale again.

## Provenance

The results CSV carries no hyperparameter columns, only `autoencoder_path` (e.g. `trainer23-v2`).
Every hyperparameter in this report is **read from the released checkpoint's `config.json`**, one
file per row, joined on (group dir, trainer dir); `build_report.py` raises if any evaluated row
lacks a config. The released standard configs omit `activation_dim`, `dict_size` and `dict_class`,
so for those {int((cfg.trainer_class == 'StandardTrainer').sum())} checkpoints the dictionary size
is measured from `ae.pt`'s `encoder.weight` shape -- still a direct measurement, not an inference.
The recorded field is `l1_penalty` for standard and gated, `sparsity_penalty` for the two
p-annealed types (the sweep script calls the same variable `initial_sparsity_penalty_`), and `k`
for TopK.

**Cross-check of the index-decoding rule.** The earlier pass inferred
`trainer index = expansion_index * S + sparsity_index` with S = 20 for the four penalty trainers
and S = 40 for top_k, expansion factors {list(EXPANSIONS)} at activation dim {ACT_DIM}, over the
grids listed in trap 2. Against the config files that rule is **correct everywhere**: no
dictionary-size mismatch on any of the {len(cfg)} checkpoints, and the largest disagreement in a
sparsity value is {q(check.max_value_gap.max(), 2)}, which is float32 grid rounding.

{md_table([[TYPE_LABEL[t], s, str(int(r.n)), q(r.max_value_gap, 2), str(int(r.dict_size_mismatches))]
           for (t, s), r in check.iterrows()],
          ["type", "sweep", "n checkpoints", "max value gap", "dict size mismatches"])}

**Superset.** `{TRAINED}` ({prov['full_rows']} rows) contains all {prov['subset_rows']} rows of
`{TRAINED_SUBSET}` with {prov['conflicts']} conflicting values across {prov['shared_cols']} shared
columns on the {prov['shared_rows']} shared rows. The {prov['extra_rows']} extra rows are the
2024-07-10 TopK group.

Regenerate with `python karvonen_othello_sweep/build_report.py`
(`--config-root` defaults to `$KARVONEN_RELEASED`, then to `{DEFAULT_CONFIG_ROOT}`).
""")
    return "".join(parts)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path,
                    default=here.parent / "external/Karvonen_Research/autoencoders/saved_data")
    ap.add_argument("--config-root", type=Path,
                    default=Path(os.environ.get("KARVONEN_RELEASED", DEFAULT_CONFIG_ROOT)))
    ap.add_argument("--out-dir", type=Path, default=here)
    args = ap.parse_args()

    cfg = read_configs(args.config_root)
    d = load(args.data_dir, cfg)
    ctl = controls(args.data_dir)
    check = decoding_check(cfg)
    prov = superset_check(args.data_dir, len(d))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out_dir / "sweep_full.csv", index=False)
    ctl.to_csv(args.out_dir / "controls.csv", index=False)
    selections(d[d.status == "alive"]).to_csv(args.out_dir / "best_per_type.csv", index=False)
    (args.out_dir / "REPORT.md").write_text(reflow(report(d, ctl, cfg, prov, check)))
    print(f"{len(d)} SAEs ({int((d.status == 'collapsed').sum())} collapsed) from {len(cfg)} "
          f"released configs -> {args.out_dir}")


if __name__ == "__main__":
    main()
