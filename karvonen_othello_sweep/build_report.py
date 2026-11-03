"""Regenerate the complete record of Karvonen's OthelloGPT SAE sweep.

Primary source is the 320 released checkpoints: each one's own `results.csv` for the metrics, its
`config.json` (and `ae.pt` encoder shape where the config omits dict_size) for the
hyperparameters, and `karvonen_coverage_recomputed.csv` for coverage, which the per-checkpoint
results.csv does not carry. Karvonen's published 280-row aggregate is used only to cross-check
and to mark which rows he actually published.

Emits karvonen_sweep_full.csv (one row per SAE; it and the coverage table are the paper's, in
results/shared/l0_roster/), and into --out-dir best_per_type.csv (every selection criterion),
controls.csv (random-model and raw-MLP context rows), and REPORT.md. Every number in REPORT.md is
interpolated from these frames; none is typed by hand.
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

AGGREGATE = "06-10-othello-trained-results_with_07-10_topk.csv"
AGGREGATE_SUBSET = "othello-6-10-trained-f1_results.csv"
RANDOM_MODEL = "othello-6-10-random-f1_results.csv"
MLP_NEURONS = "othello-6-10-mlp-acts-f1_results.csv"

DEFAULT_CONFIG_ROOT = "/n/netscratch/ba_lab/Lab/dcao/karvonen_released"
# group dir (as it appears in the results CSVs) -> its parent release dir on disk
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


# The rule the first pass inferred, kept as a cross-check now that configs are read directly.
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

TYPE_LABEL = {
    "StandardTrainer": "Standard",
    "PAnnealTrainer": "Standard + p-annealing",
    "GatedSAETrainer": "Gated",
    "GatedAnnealTrainer": "Gated + p-annealing",
    "TrainerTopK": "TopK",
}
TYPE_ORDER = list(TYPE_LABEL)

FUNCS = {"mine_yours": "games_batch_to_state_stack_mine_yours_BLRRC",
         "bm": "games_batch_to_state_stack_mine_yours_blank_mask_BLRRC",
         "moves": "games_batch_to_valid_moves_BLRRC"}
# short name -> (per-checkpoint column suffix) for the reconstruction metrics
RECON = {"recon_f1": "best_f1_score_per_class", "recon_f1_sq": "best_f1_score_per_square"}
COVERAGE = {"mine_yours": "coverage", "bm": "coverage_bm", "moves": "coverage_moves"}

SAE_QUALITY = ["l0", "l1_loss", "l2_loss", "frac_alive", "frac_variance_explained", "cossim",
               "l2_ratio", "frac_recovered", "num_alive_features"]

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

# Collapsed rows top out at L0 2.75; the next surviving row is at 19.2. Thresholding on L0 rather
# than FVE because the two evaluation runs disagree on FVE by up to 0.76 on exactly these rows.
ALIVE_L0 = 5.0
# Median relative disagreement between Karvonen's two evaluation runs on l0 (see the join section).
L0_EDGE_MARGIN = 0.10


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


def read_per_checkpoint(config_root: Path) -> pd.DataFrame:
    frames = []
    for group, parent in CONFIG_GROUPS.items():
        for d in sorted((config_root / parent / group).glob("trainer*")):
            row = pd.read_csv(d / "results.csv")
            frames.append(row.assign(group_dir=group, trainer_dir=d.name))
    return pd.concat(frames, ignore_index=True)


def read_aggregate(data_dir: Path) -> pd.DataFrame:
    agg = pd.read_csv(data_dir / AGGREGATE)
    agg["group_dir"] = agg.autoencoder_group_path.str.rstrip("/").str.split("/").str[-1]
    agg["trainer_dir"] = agg.autoencoder_path.str.rstrip("/").str.split("/").str[-1]
    return agg


def read_coverage(path: Path) -> pd.DataFrame:
    """karvonen_coverage_recomputed.csv, long -> one column per board function."""
    cov = pd.read_csv(path)
    wide = cov.pivot(index=["group_dir", "trainer_dir"], columns="func",
                     values="best_average_f1").reset_index()
    return wide.rename(columns={FUNCS[k]: v for k, v in COVERAGE.items()})


def assemble(cfg, per, cov, agg) -> pd.DataFrame:
    d = cfg.merge(per, on=["group_dir", "trainer_dir"], suffixes=("", "_per"),
                  how="left", validate="one_to_one")
    d = d.merge(cov, on=["group_dir", "trainer_dir"], how="left", validate="one_to_one")
    if d[list(COVERAGE.values())].isna().any().any():
        missing = d[d.coverage.isna()].trainer_dir.tolist()
        raise ValueError(f"no recomputed coverage for {missing}; run recompute_coverage.py")

    d["type"] = d.trainer_class.map(TYPE_LABEL)
    d["sweep"] = np.where(d.trainer_dir.str.endswith("-v2"), "v2", "orig")
    d["trainer_idx"] = d.trainer_dir.str.extract(r"trainer_?(\d+)")[0].astype(int)
    d["expansion"] = d.dict_size // d.activation_dim
    published = set(zip(agg.group_dir, agg.trainer_dir))
    d["provenance"] = np.where([k in published for k in zip(d.group_dir, d.trainer_dir)],
                               "published", "unpublished")
    for short, suffix in RECON.items():
        for key, func in FUNCS.items():
            name = short if key == "mine_yours" else f"{short}_{key}"
            d[name] = d[f"{func}_{suffix}"]
    d["coverage_bm_rescaled"] = d.coverage_bm * BLANK_MASK_RESCALE
    d["status"] = np.where(d.l0 >= ALIVE_L0, "alive", "collapsed")

    keyed = agg.set_index(["group_dir", "trainer_dir"])
    idx = pd.MultiIndex.from_arrays([d.group_dir, d.trainer_dir])
    d["l0_published"] = keyed.l0.reindex(idx).values
    d["coverage_published"] = keyed[f"{FUNCS['mine_yours']}_best_average_f1"].reindex(idx).values
    d["recon_f1_published"] = keyed[f"{FUNCS['mine_yours']}_best_f1_score_per_class"].reindex(idx).values
    d["frac_variance_explained_published"] = keyed.frac_variance_explained.reindex(idx).values

    cols = ["trainer_class", "type", "group_dir", "trainer_dir", "sweep", "provenance",
            "trainer_idx", "hyperparameter", "hyperparameter_value", "dict_size", "activation_dim",
            "expansion", "lr", "seed", "steps", *SAE_QUALITY,
            *[f"{s}{'' if k == 'mine_yours' else '_' + k}" for s in RECON for k in FUNCS],
            *COVERAGE.values(), "coverage_bm_rescaled",
            "l0_published", "coverage_published", "recon_f1_published",
            "frac_variance_explained_published", "status"]
    return d[cols].sort_values(["trainer_class", "sweep", "dict_size", "hyperparameter_value"])


def published_frame(agg: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """The same schema, but every metric taken from Karvonen's published aggregate.

    Only used to answer "did switching source or adding the 40 unpublished rows move a winner".
    """
    keyed = agg.set_index(["group_dir", "trainer_dir"])
    p = d[d.provenance == "published"].copy()
    idx = pd.MultiIndex.from_arrays([p.group_dir, p.trainer_dir])
    for c in SAE_QUALITY:
        p[c] = keyed[c].reindex(idx).values
    for key, func in FUNCS.items():
        name = "recon_f1" if key == "mine_yours" else f"recon_f1_{key}"
        p[name] = keyed[f"{func}_best_f1_score_per_class"].reindex(idx).values
        p[COVERAGE[key]] = keyed[f"{func}_best_average_f1"].reindex(idx).values
    p["coverage_bm_rescaled"] = p.coverage_bm * BLANK_MASK_RESCALE
    p["status"] = np.where(p.l0 >= ALIVE_L0, "alive", "collapsed")
    return p


def agreement(per: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Column-by-column comparison of the two evaluation runs on the rows they share."""
    P = per.set_index(["group_dir", "trainer_dir"])
    A = agg.set_index(["group_dir", "trainer_dir"])
    shared = sorted(set(P.index) & set(A.index))
    numeric = sorted(c for c in set(P.columns) & set(A.columns)
                     if P[c].dtype.kind in "fi" and A[c].dtype.kind in "fi"
                     and not c.startswith("Unnamed"))
    rows = []
    for c in numeric:
        x, y = P.loc[shared, c].astype(float).values, A.loc[shared, c].astype(float).values
        both = ~(np.isnan(x) | np.isnan(y))
        gap = np.abs(x[both] - y[both])
        rel = gap / np.maximum(np.abs(y[both]), 1e-12)
        is_topk = np.array([g.endswith("top_k") for g, _ in shared])[both]
        rows.append({"column": c, "n": int(both.sum()), "median_abs": float(np.median(gap)),
                     "max_abs": float(gap.max()), "median_rel": float(np.median(rel)),
                     "n_identical": int((gap == 0).sum()),
                     "topk_max_abs": float(gap[is_topk].max()),
                     "kind": "board" if "BLRRC" in c else "sae quality"})
    return pd.DataFrame(rows)


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


def superset_check(data_dir: Path) -> dict:
    """The released 240-row CSV must be a value-identical subset of the 280-row one."""
    sub = pd.read_csv(data_dir / AGGREGATE_SUBSET)
    full = pd.read_csv(data_dir / AGGREGATE)
    key_full = full.autoencoder_path.str.rstrip("/")
    key_sub = sub.autoencoder_path.str.rstrip("/")
    shared = sorted(set(key_full) & set(key_sub))
    a, b = full.set_index(key_full), sub.set_index(key_sub)
    cols = [c for c in b.columns if c in a.columns and not c.startswith("Unnamed")]
    conflicts = 0
    for c in cols:
        x, y = a.loc[shared, c], b.loc[shared, c]
        if x.dtype.kind in "fi":
            same = np.isclose(x.astype(float), y.astype(float), rtol=1e-9) | (x.isna() & y.isna())
        else:
            same = x.astype(str) == y.astype(str)
        conflicts += int((~same).sum())
    return {"subset_rows": len(sub), "full_rows": len(full), "shared_rows": len(shared),
            "shared_cols": len(cols), "conflicts": conflicts}


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
        for key, func in FUNCS.items():
            name = "recon_f1" if key == "mine_yours" else f"recon_f1_{key}"
            c[name] = raw[f"{func}_best_f1_score_per_class"]
            c[COVERAGE[key]] = raw[f"{func}_best_average_f1"]
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
    cols = ["criterion", "scope", "type", "sweep", "provenance", "trainer_dir", "expansion",
            "dict_size", "hyperparameter", "hyperparameter_value", "l0", "num_alive_features",
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


SWEEP_HEADER = ["trainer dir", "sweep", "pub", "exp", "dict", "hyperparam", "value", "L0",
                "alive feat", "FVE", "recon F1", "coverage", "recon F1 bm", "cov bm",
                "cov bm x1.5", "status"]


def sweep_row(r) -> list[str]:
    return [r.trainer_dir, r.sweep, "y" if r.provenance == "published" else "n",
            str(r.expansion), str(r.dict_size), r.hyperparameter, q(r.hyperparameter_value),
            q(r.l0), str(r.num_alive_features), q(r.frac_variance_explained), q(r.recon_f1),
            q(r.coverage), q(r.recon_f1_bm), q(r.coverage_bm), q(r.coverage_bm_rescaled),
            "collapsed" if r.status == "collapsed" else ""]


PICK_HEADER = ["type", "trainer dir", "pub", "hyperparam", "value", "dict", "L0", "alive feat",
               "coverage", "cov bm", "cov bm x1.5", "recon F1", "recon F1 bm"]


def pick_row(r) -> list[str]:
    return [r.type, r.trainer_dir, "y" if r.provenance == "published" else "n", r.hyperparameter,
            q(r.hyperparameter_value), str(r.dict_size), q(r.l0), str(r.num_alive_features),
            q(r.coverage), q(r.coverage_bm), q(r.coverage_bm_rescaled), q(r.recon_f1),
            q(r.recon_f1_bm)]


def picks(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return frame.loc[frame.groupby("trainer_class")[metric].idxmax()].sort_values(
        metric, ascending=False)


def picks_table(frame: pd.DataFrame, metric: str) -> str:
    return md_table([pick_row(r) for _, r in picks(frame, metric).iterrows()], PICK_HEADER)


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
                if g.empty:
                    cells.append("--")
                    continue
                best = g.loc[g[metric].idxmax()]
                mark = "" if best.provenance == "published" else "*"
                cells.append(f"{q(best[metric])} ({best.trainer_dir}{mark})")
            rows.append([f"{blo}-{bhi}", metric, *cells])
    return md_table(rows, header)


def winner_diff(new: pd.DataFrame, old: pd.DataFrame, noise: float) -> str:
    """noise: the run-to-run spread on the board metrics, so a flip below it is not a real change."""
    rows = []
    for metric in CRITERIA:
        a = picks(new, metric).set_index("trainer_class")
        b = picks(old, metric).set_index("trainer_class")
        for t in TYPE_ORDER:
            n, o = a.loc[t], b.loc[t]
            gain = n[metric] - o[metric]
            if n.trainer_dir == o.trainer_dir:
                verdict = "same"
            elif n.provenance == "unpublished":
                verdict = "an unpublished row now wins"
            else:
                verdict = "tie flip, within run-to-run noise" if abs(gain) < noise else "changed"
            rows.append([metric, TYPE_LABEL[t], f"{o.trainer_dir} ({q(o[metric])})",
                         f"{n.trainer_dir} ({q(n[metric])})", q(gain, 2), verdict])
    return md_table(rows, ["criterion", "type", "published-only winner", "full-sweep winner",
                           "delta", "verdict"])


def report(d, pub, ctl, cfg, agree, prov, check) -> str:
    alive = d[d.status == "alive"]
    dead = d[d.status == "collapsed"]
    lo, hi = MATCHED_BAND
    band = alive[(alive.dict_size == MATCHED_DICT_SIZE) & alive.l0.between(lo, hi)]
    rec = picks(band, "coverage_bm")
    order = lambda frame, metric: list(picks(frame, metric).trainer_class)
    if order(band, "coverage_bm") != order(alive, "coverage_bm"):
        raise ValueError("matched-band and whole-sweep type orderings diverged; rewrite the claim")
    rank = {t: i for i, t in enumerate(order(alive, "coverage_bm"))}
    flip = pd.Series([rank[t] for t in order(alive, "recon_f1")]).corr(
        pd.Series(range(len(rank))), method="spearman")

    mlp5 = ctl[(ctl.control == "raw MLP neurons") & (ctl.name == "layer_5")].iloc[0]
    mlp7 = ctl[(ctl.control == "raw MLP neurons") & (ctl.name == "layer_7")].iloc[0]
    rnd = ctl[ctl.control == "random-weight model"]
    rand_best = rnd.loc[rnd.coverage.idxmax()]
    rescale_gap = float((alive.coverage_bm_rescaled - alive.coverage).abs().mean())

    top_cov = alive.loc[alive.coverage.idxmax()]
    top_covbm = alive.loc[alive.coverage_bm.idxmax()]
    top_f1 = alive.loc[alive.recon_f1.idxmax()]
    fr = alive.frac_recovered
    std = alive[alive.trainer_class == "StandardTrainer"]
    topk = alive[alive.trainer_class == "TrainerTopK"]
    new_topk = topk[topk.provenance == "unpublished"]
    topk80 = topk[(topk.dict_size == MATCHED_DICT_SIZE) & (topk.l0 == 80)].iloc[0]
    topk_winner = topk.loc[topk.coverage.idxmax()]
    seam = lambda t, s, ds, i: d[(d.trainer_class == t) & (d.sweep == s) & (d.dict_size == ds)
                                 & (d.trainer_idx % 20 == i)].iloc[0]
    cov_check = (d.coverage - d.coverage_published).abs().dropna()
    l0_gap = agree[agree.column == "l0"].iloc[0]
    f1_gap = agree[agree.column == f"{FUNCS['mine_yours']}_best_f1_score_per_class"].iloc[0]
    edge = band[(band.l0 < lo * (1 + L0_EDGE_MARGIN)) | (band.l0 > hi * (1 - L0_EDGE_MARGIN))]
    fve_gap = (d.frac_variance_explained - d.frac_variance_explained_published).abs()
    fve_gap_low = fve_gap[d.l0 < ALIVE_L0].max()
    fve_gap_mid = fve_gap[(d.l0 >= 20) & (d.l0 < 100)].median()
    fve_gap_high = fve_gap[d.l0 >= 100].median()

    parts = [f"""# Karvonen's OthelloGPT SAE sweep: the complete record

All {len(d)} SAEs he trained on layer 5 of OthelloGPT, five trainer types --
{int((d.provenance == 'published').sum())} that appear in his published results CSV and
{int((d.provenance == 'unpublished').sum())} TopK checkpoints he evaluated but never published.
Metrics come from each checkpoint's own `results.csv`, hyperparameters from its `config.json`,
coverage recomputed from its saved per-feature counts with his own code. Regenerated by
`build_report.py`; no number below was typed by hand.

## The answer: one config per type

Criterion: **blank-masked coverage** (`best_average_f1` on the blank-masked board function), at
matched dictionary size {MATCHED_DICT_SIZE} and matched sparsity L0 in [{lo:.0f}, {hi:.0f}].
Reasoning in *Which criterion, and why*. Karvonen defines no composite selection score of his own
-- confirmed by reading his code -- so the criterion is ours, and it is stated rather than buried
in a weighted sum. `pub` = y if the row is in his published CSV, n if it is one of the 40 he never
published.

{md_table([pick_row(r) for _, r in rec.iterrows()], PICK_HEADER)}

`cov bm x1.5` puts the blank-masked column back on the same 0-1 scale as the unmasked one; see
trap 3. All four board metrics run the same way: higher is better.

**The ordering is not an artifact of the band.** Ranking each type by its best blank-masked
coverage anywhere in the sweep, at either dictionary size, gives the same five-way order
(`build_report.py` raises if that ever stops being true). It *is* an artifact of the metric: rank
the same five types by reconstruction F1 and the order all but inverts, Spearman {q(flip, 2)}.

## Sources, and the join between them

Three files per checkpoint carry what this report needs, and they are not the same as the
published aggregate:

{md_table([
    ["`config.json`", "320", "hyperparameters", "sparsity penalty or k, dict_size, lr, seed"],
    ["`results.csv`", "320", "primary metrics",
     "l0, FVE, num_alive_features, frac_recovered, reconstruction F1 for all 3 board functions"],
    ["`indexing_None_n_inputs_1000_results.pkl`", "320", "coverage",
     "per-(threshold, feature, square, class) counts; coverage recomputed from these"],
    [f"`{AGGREGATE}`", "280", "cross-check only", "his published numbers, and which rows are published"],
], ["source", "n", "role here", "what it gives"])}

Joining the per-checkpoint `results.csv` files to the published aggregate on (group dir, trainer
dir) gives **{int(agree.n.max())} matched, 0 aggregate-only, and
{int((d.provenance == 'unpublished').sum())} per-checkpoint-only** rows. Every one of the
{int((d.provenance == 'unpublished').sum())} extra rows is TopK.

**The two are separate evaluation runs, except for TopK.** On the 40 shared TopK rows the largest
disagreement across all {len(agree)} shared numeric columns is
{q(agree.topk_max_abs.max(), 2)} -- CSV float round-trip, i.e. the aggregate's TopK block *is* the
per-checkpoint output, pasted in. On the other 240 rows most columns differ, because the
2024-05-23 groups were evaluated once at training time and again for the 2024-06-10 aggregate.

{md_table([[r.column, str(int(r.n)), q(r.median_abs, 2), q(r.max_abs, 2), q(r.median_rel, 2),
            str(int(r.n_identical))]
           for _, r in agree[agree.kind == "sae quality"].sort_values("median_rel", ascending=False).iterrows()],
          ["SAE-quality column", "n", "median abs diff", "max abs diff", "median rel", "n identical"])}

{md_table([[r.column.replace("games_batch_to_", "").replace("_BLRRC", ""), str(int(r.n)),
            q(r.median_abs, 2), q(r.max_abs, 2), q(r.median_rel, 2), str(int(r.n_identical))]
           for _, r in agree[(agree.kind == "board") & agree.column.str.contains("best_f1_score_per_class|best_average|num_squares|board_count")].iterrows()],
          ["board column", "n", "median abs diff", "max abs diff", "median rel", "n identical"])}

**Which do we trust?** For the board columns the question barely arises: the two runs agree to a
median {q(f1_gap.median_rel, 2)} relative on reconstruction F1, worst case
{q(f1_gap.max_abs, 2)} absolute, and they score the identical evaluation set
(`num_squares` and `board_reconstruction_board_count` are bit-identical on all
{int(f1_gap.n)} shared rows). For the SAE-quality columns neither run is more trustworthy than
the other, because **both are single-batch estimates**: Karvonen computes l0, FVE, `frac_alive`,
`cossim` and `l2_ratio` from one batch of 25 activations. Their disagreement is therefore a direct
measurement of that noise -- median {q(l0_gap.median_rel, 2)} relative on l0, up to
{q(l0_gap.max_abs, 2)} absolute on the densest rows -- not evidence that one file is wrong. This
report takes the
per-checkpoint value as primary (it is the run the 40 new rows come from, so the TopK sweep stays
internally consistent) and keeps the aggregate's l0 alongside it in `karvonen_sweep_full.csv` as
`l0_published`.

**Coverage is not in the per-checkpoint files.** They carry 84 columns against the aggregate's
{prov['shared_cols'] + 184}; what is lost is `best_average_f1` (coverage) for all three board
functions, its argmax index, the serialised `all_average_f1s` vector, and 342
`f1_threshold_*` columns. What is gained is `best_f1_score_per_square`, six columns the aggregate
does not have. Since coverage is the metric this report ranks on, it is **recomputed** for all
{len(d)} checkpoints by `recompute_coverage.py`, which runs Karvonen's own
`analysis.get_all_f1s` + `f1_analysis.best_f1_average` over each checkpoint's saved counts.

That recomputation is a reproduction, not a re-estimate. Across all {int(cov_check.notna().sum())}
published rows it reproduces his coverage column to a median {q(cov_check.median(), 2)} and a
maximum {q(cov_check.max(), 2)} -- more than an order of magnitude tighter than the run-to-run
spread on reconstruction F1 ({q(f1_gap.max_abs, 2)}). `f1_analysis.py:159-232` shows why: his
coverage columns were themselves produced by loading these same pickles, so the published
aggregate is a mix -- SAE-quality and reconstruction columns from the 2024-06-10 re-evaluation,
coverage columns derived from the 2024-05-27 saved counts. **Coverage for the 40 unpublished rows
therefore stands on exactly the same footing as his published coverage.**

## What the four metrics are

Karvonen scores each SAE against three board-state functions. Two are used here; the third,
`games_batch_to_valid_moves`, is summarised at the end.

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

Both come from 1000 games x 59 positions and are solid. They measure different things and they do
not agree: across the {len(alive)} surviving SAEs their Spearman correlation is
{q(alive.coverage.corr(alive.recon_f1, method='spearman'))}.

## Read this before you rank anything

**1. The SAE-quality columns are single-batch estimates; the board columns are not.** Karvonen
computes l0, FVE, `frac_alive`, `cossim` and `l2_ratio` from one batch of 25 activations, and his
two runs disagree on l0 by a median {q(l0_gap.median_rel, 2)} relative
({int(l0_gap.n_identical)} of {int(l0_gap.n)} rows identical -- the TopK rows, where L0 == k by
construction, plus the dead rows where it is 0). So treat "matched L0" as matched to within about
10%, not exactly: of the {len(band)} configs in the matched band, {len(edge)} sit within
{L0_EDGE_MARGIN:.0%} of an edge and could fall the other side under the aggregate's numbers.
Nothing in the recommendation turns on those.

FVE is worse than noisy at the sparse end. The two runs agree to a median
{q(fve_gap_high, 2)} absolute above L0 100 and {q(fve_gap_mid, 2)} between L0 20 and 100, but on
the {int((d.l0 < ALIVE_L0).sum())} rows below L0 5 they disagree by up to {q(fve_gap_low, 2)} --
one run reports 0.8255 where the other reports 0.1751 for the same checkpoint. FVE at very low L0
is not comparable across runs and is not used for anything here, including the aliveness cut,
which is thresholded on L0 instead ({q(ALIVE_L0)}; collapsed rows top out at
{q(d[d.status == 'collapsed'].l0.max())} and the next surviving row is at
{q(alive.l0.min())}, and that split is identical under both runs).

The **board metrics carry no such caveat**: 1000 games x 59 positions, and the two runs agree to a
median {q(f1_gap.median_rel, 2)} relative.

**2. {len(dead)} of the {len(d)} rows are collapsed, not merely sparse.** All are
`StandardTrainer`, all from the original sweep, all at l1_penalty >=
{q(dead.hyperparameter_value.min())}. The original othello L1 grid ran to 0.9 while the corrected
one stops at 0.08, so from the fourth grid point on the SAE trains to nothing.
{int((d.l0 == 0).sum())} have L0 and `frac_variance_explained` exactly 0. They are flagged
`collapsed` in `karvonen_sweep_full.csv` and excluded from every selection here. Note their
`num_alive_features` is still {int(dead.num_alive_features.min())}-{int(dead.num_alive_features.max())}:
that column does not go to zero on a dead SAE.

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

**4. `-v2` rows are a second sweep, and the two grids overlap at the seams.** On 2024-05-26 the
standard and gated grids were re-run: standard `linspace(0.035, 0.9, 20)` -> `linspace(0.035,
0.08, 20)`, gated `linspace(0.2, 1.0, 20)` -> `linspace(1.0, 2.0, 20)`. p_anneal and gated_anneal
never were, which is why only those two types carry `-v2` rows. The configs confirm it directly --
`trainer19-v2` records l1_penalty
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

**5. `frac_recovered` is saturated and ranks nothing.** Across the {len(alive)} surviving SAEs it
spans {q(fr.min(), 6)} to {q(fr.max(), 6)}, with {int(((fr - 1).abs() < 1e-3).sum())} of them
within 0.001 of 1.000, and it comes from the same single batch. Do not put it in a selection rule.

**6. `frac_alive` is not the fraction of live features.** On every non-TopK row it equals exactly
`l0 / dict_size` -- the mean fraction of the dictionary active *per token*, mislabelled. On every
TopK row it equals exactly `1 / dict_size`, a constant. Use `num_alive_features`.

## Context: the floor and the neuron bar

Control rows come from Karvonen's published control CSVs, so they are on the aggregate run.

{md_table([
    ["random-weight OthelloGPT, best of 40 SAEs", q(rand_best.l0), str(int(rand_best.num_alive_features)),
     q(rand_best.coverage), q(rand_best.coverage_bm), q(rand_best.coverage_bm_rescaled),
     q(rand_best.recon_f1), q(rand_best.recon_f1_bm)],
    ["raw MLP neurons, layer 5 (2048, no SAE)", q(mlp5.l0), str(int(mlp5.num_alive_features)),
     q(mlp5.coverage), q(mlp5.coverage_bm), q(mlp5.coverage_bm_rescaled),
     q(mlp5.recon_f1), q(mlp5.recon_f1_bm)],
    ["best SAE, by coverage", q(top_cov.l0), str(int(top_cov.num_alive_features)),
     q(top_cov.coverage), q(top_cov.coverage_bm), q(top_cov.coverage_bm_rescaled),
     q(top_cov.recon_f1), q(top_cov.recon_f1_bm)],
    ["best SAE, by blank-masked coverage", q(top_covbm.l0), str(int(top_covbm.num_alive_features)),
     q(top_covbm.coverage), q(top_covbm.coverage_bm), q(top_covbm.coverage_bm_rescaled),
     q(top_covbm.recon_f1), q(top_covbm.recon_f1_bm)],
    ["best SAE, by reconstruction F1", q(top_f1.l0), str(int(top_f1.num_alive_features)),
     q(top_f1.coverage), q(top_f1.coverage_bm), q(top_f1.coverage_bm_rescaled),
     q(top_f1.recon_f1), q(top_f1.recon_f1_bm)],
], ["row", "L0", "alive feat", "coverage", "cov bm", "cov bm x1.5", "recon F1", "recon F1 bm"])}

*The floor is high.* SAEs trained on a **randomly initialised** OthelloGPT reach coverage
{q(rand_best.coverage)} and reconstruction F1 {q(rand_best.recon_f1)}. Any claim about board
representations has to clear that, not zero. (That sweep is standard-trainer only and inherits the
same broken L1 grid, so {int((rnd.l0 == 0).sum())} of its 40 rows are dead; the row above is the
best survivor.)

*No SAE in the sweep beats raw neurons on coverage.* Layer-5 MLP neurons -- 2048 of them, read
directly, L0 {q(mlp5.l0)} -- reach coverage {q(mlp5.coverage)}, above the best SAE
({q(top_cov.coverage)}). On reconstruction F1 the best SAE ({q(top_f1.recon_f1)}) does beat the
neurons ({q(mlp5.recon_f1)}). Caveat: the neurons are the MLP hidden layer while the SAEs sit on
the residual stream, so this is a bar rather than a controlled ablation -- but it is the bar that
matters when the pitch is "the dictionary is more interpretable than the neurons".

## Best per type, one criterion at a time

Nothing is aggregated across these four tables. Each is the argmax of one metric over surviving
rows, both dictionary sizes, both sweeps, published and unpublished together.
"""]

    for metric, label in CRITERIA.items():
        parts.append(f"\n### {label} -- `{metric}`\n\n{picks_table(alive, metric)}\n")

    parts.append(f"""
### Did completing the sweep move a winner?

Left column: the winner using only Karvonen's {int((d.provenance == 'published').sum())} published
rows *and his published metric values*. Right column: the same argmax over all {len(d)} rows using
per-checkpoint metrics. Both effects are folded together here on purpose -- that is the actual
difference between the previous table and this one.

{winner_diff(alive, pub[pub.status == "alive"], f1_gap.max_abs)}

The 40 new rows are all TopK, so only TopK can gain a winner from them; everything else can move
only if the run-to-run metric spread flips two near-tied configs, which is what the sole non-TopK
change is: a delta well inside the {q(f1_gap.max_abs, 2)} spread the two runs show on that metric.
The real result is that **TopK's winner changes under both coverage criteria, and the new winner
is a checkpoint Karvonen never published** -- `{topk_winner.trainer_dir}`, k =
{q(topk_winner.hyperparameter_value)} at dict {int(topk_winner.dict_size)}.

## The completed TopK sweep

Karvonen trained 80 TopK checkpoints and published 40. The other 40 have weights, an evaluation
`results.csv` and saved per-feature counts on disk -- everything needed to score them -- and they
come from the same evaluation run as the published 40, so the two halves are directly comparable.
Filling them in makes k = 10, 20, ..., 400 complete at both dictionary sizes.

{md_table([[str(int(ds)), q(g.coverage.max()), g.loc[g.coverage.idxmax()].trainer_dir,
            q(g[g.provenance == 'published'].coverage.max()),
            g[g.provenance == 'published'].loc[g[g.provenance == 'published'].coverage.idxmax()].trainer_dir,
            q(g.coverage_bm.max()), g.loc[g.coverage_bm.idxmax()].trainer_dir]
           for ds, g in topk.groupby("dict_size")],
          ["dict", "best coverage (all 80)", "at", "best coverage (published 40)", "at",
           "best cov bm (all 80)", "at"])}

Of the {len(new_topk)} newly usable TopK rows, {int((new_topk.coverage > topk[topk.provenance == 'published'].coverage.max()).sum())}
beat the best published TopK coverage and
{int((new_topk.coverage_bm > topk[topk.provenance == 'published'].coverage_bm.max()).sum())} beat
the best published TopK blank-masked coverage.

## Best per type at matched L0

Same argmax, one L0 band at a time, both dictionary sizes pooled. `--` means that type has no
surviving config in the band -- itself a finding, since the families never covered the same
sparsity range. A `*` marks a winner that is one of the unpublished checkpoints. Band edges are
soft; see trap 1.

{band_matrix(alive)}

## Which criterion, and why

**Karvonen has no composite selection score.** In his repo every `best_*` is an argmax *within*
one SAE (over the 11 activation thresholds, or over features); the only cross-config ranking
anywhere is a `print_top_n_values(df, key, n)` in `circuits/f1_analysis.ipynb` that sorts by one
metric column at a time to quote maxima in the paper's results text. Every figure is a scatter of
one metric against L0, coloured by trainer class, with the trade-off left to the reader's eye.
Inventing a weighted score would be putting our thumb on his scale, so we do not.

That leaves choosing a criterion and defending it. The recommendation uses **blank-masked coverage
at matched dictionary size and matched L0**:

*Coverage over reconstruction F1*, because reconstruction F1 is a decodability test a dense
uninterpretable basis passes. It is maximised at the densest end of the grid (rho(L0, recon F1) =
{q(std.l0.corr(std.recon_f1, method='spearman'))} for standard SAEs) and its sweep-wide winner has
near-floor coverage. Coverage asks the question a dictionary exists to answer: is there one
feature per board-state property.

{md_table([[TYPE_LABEL[t],
            q(alive[alive.trainer_class == t].l0.corr(alive[alive.trainer_class == t].coverage, method='spearman')),
            q(alive[alive.trainer_class == t].l0.corr(alive[alive.trainer_class == t].recon_f1, method='spearman')),
            q(alive[alive.trainer_class == t].coverage.corr(alive[alive.trainer_class == t].recon_f1, method='spearman')),
            str(len(alive[alive.trainer_class == t]))]
           for t in TYPE_ORDER],
          ["type", "rho(L0, coverage)", "rho(L0, recon F1)", "rho(coverage, recon F1)", "n alive"])}

*Blank-masked over unmasked*, because "this square is empty" is the cheap third of the unmasked
label set -- early-game boards are mostly empty and blankness is close to a function of move
count. Removing it is the entire point of the masked config. The masked column's denominator bug
(trap 3) rescales every row by one constant and so cannot distort a ranking, and the switch does
change the answer at the top: unmasked coverage puts {TYPE_LABEL[top_cov.trainer_class]} first,
blank-masked puts {TYPE_LABEL[top_covbm.trainer_class]} first.

*Matched dict {MATCHED_DICT_SIZE} and matched L0 {lo:.0f}-{hi:.0f}*, because the SpaDE study trains
every architecture at dict 8192 and evaluates its TopK and BatchTopK baselines near L0 80, and
because the families' grids do not span the same sparsity range -- an unconstrained per-type
maximum partly rewards whichever grid happened to reach the most favourable L0.

**What this does not license.** These are Karvonen's numbers from his evaluation code. The SpaDE
study re-implements coverage and board reconstruction with a train/test split for the
feature-to-property assignment and a 60-square (non-middle) mask, and retrains its own baselines.
At a nominally identical config -- TopK, dict {MATCHED_DICT_SIZE}, L0 80 -- these files give
coverage {q(topk80.coverage)} and reconstruction F1 {q(topk80.recon_f1)}, which the study's own
run of the same trainer does not reproduce. Use this sweep to pick *hyperparameters to retrain and
re-evaluate in house*; do not paste these numbers into a table beside SpaDE's.

## The full sweep

One row per checkpoint, all {len(d)}. `pub` = y if the row appears in Karvonen's published results
CSV, n if he computed it but never published it -- cite the two differently. Sorted by dictionary
size then by the sparsity hyperparameter, read from `config.json`. `FVE` is
`frac_variance_explained`, `bm` is blank-masked, `cov bm x1.5` is the trap-3 rescale.
""")

    for t in TYPE_ORDER:
        g = d[d.trainer_class == t]
        n_unpub = int((g.provenance == "unpublished").sum())
        extra = f", {n_unpub} of them unpublished" if n_unpub else ""
        parts.append(f"\n### {TYPE_LABEL[t]} (`{t}`) -- {len(g)} SAEs{extra}\n")
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

    moves = picks(alive, "coverage_moves")
    parts.append(f"""
## The third board function

`games_batch_to_valid_moves` (is this square a legal move?) is used in no selection above. Best per
type by its coverage, for reference:

{md_table([[r.type, r.trainer_dir, "y" if r.provenance == "published" else "n", str(r.dict_size),
            q(r.l0), q(r.coverage_moves), q(r.recon_f1_moves)]
           for _, r in moves.iterrows()],
          ["type", "trainer dir", "pub", "dict", "L0", "coverage (moves)", "recon F1 (moves)"])}

Raw layer-5 MLP neurons score {q(mlp5.coverage_moves)} / {q(mlp5.recon_f1_moves)} here, and layer 7
neurons reach {q(mlp7.coverage_moves)} / {q(mlp7.recon_f1_moves)} -- legal-move information is
concentrated late in the model, unlike board state. Its denominator is 64 (one class), so it sits
on a third scale again.

## Provenance

Every hyperparameter is **read from the released checkpoint's `config.json`**, joined on (group
dir, trainer dir); `build_report.py` raises if any row lacks one. The released standard configs
omit `activation_dim`, `dict_size` and `dict_class`, so for those
{int((cfg.trainer_class == 'StandardTrainer').sum())} checkpoints the dictionary size is measured
from `ae.pt`'s `encoder.weight` shape. The recorded field is `l1_penalty` for standard and gated,
`sparsity_penalty` for the two p-annealed types (the sweep script calls the same variable
`initial_sparsity_penalty_`), and `k` for TopK.

**Cross-check of the index-decoding rule.** An earlier pass inferred `trainer index =
expansion_index * S + sparsity_index` with S = 20 for the four penalty trainers and S = 40 for
top_k, expansion factors {list(EXPANSIONS)} at activation dim {ACT_DIM}, over the grids in trap 4.
Against the config files that rule is **correct everywhere**: no dictionary-size mismatch on any of
the {len(cfg)} checkpoints, and the largest disagreement in a sparsity value is
{q(check.max_value_gap.max(), 2)}, float32 grid rounding.

{md_table([[TYPE_LABEL[t], s, str(int(r.n)), q(r.max_value_gap, 2), str(int(r.dict_size_mismatches))]
           for (t, s), r in check.iterrows()],
          ["type", "sweep", "n checkpoints", "max value gap", "dict size mismatches"])}

**Superset of the older aggregate.** `{AGGREGATE}` ({prov['full_rows']} rows) contains all
{prov['subset_rows']} rows of `{AGGREGATE_SUBSET}` with {prov['conflicts']} conflicting values
across {prov['shared_cols']} shared columns on the {prov['shared_rows']} shared rows.

Regenerate with `python karvonen_othello_sweep/recompute_coverage.py --config-root <root>` (slow,
output committed to results/shared/l0_roster/) then `python karvonen_othello_sweep/build_report.py`
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
    roster = here.parent / "results" / "shared" / "l0_roster"
    ap.add_argument("--coverage", type=Path, default=roster / "karvonen_coverage_recomputed.csv")
    ap.add_argument("--sweep-full", type=Path, default=roster / "karvonen_sweep_full.csv")
    ap.add_argument("--out-dir", type=Path, default=here)
    args = ap.parse_args()

    cfg = read_configs(args.config_root)
    per = read_per_checkpoint(args.config_root)
    agg = read_aggregate(args.data_dir)
    d = assemble(cfg, per, read_coverage(args.coverage), agg)
    pub = published_frame(agg, d)
    ctl = controls(args.data_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.sweep_full, index=False)
    ctl.to_csv(args.out_dir / "controls.csv", index=False)
    agree = agreement(per, agg)
    agree.to_csv(args.out_dir / "run_agreement.csv", index=False)
    selections(d[d.status == "alive"]).to_csv(args.out_dir / "best_per_type.csv", index=False)
    (args.out_dir / "REPORT.md").write_text(reflow(report(
        d, pub, ctl, cfg, agree, superset_check(args.data_dir), decoding_check(cfg))))
    print(f"{len(d)} SAEs ({int((d.provenance == 'unpublished').sum())} unpublished, "
          f"{int((d.status == 'collapsed').sum())} collapsed) -> {args.out_dir}")


if __name__ == "__main__":
    main()
