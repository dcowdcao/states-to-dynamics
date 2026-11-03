"""Attacks that need only the shipped CSVs: information-preservation confounds, the
board-arm collapse, the R3 selectivity model, and the reference-role drift null.

Nothing here imports spade.legality_probe. Reads
analysis/legality_probe/results/{probe_fit,readout,space_quality}.csv.

Usage: python analysis/legality_probe/redteam/csv_attacks.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPADE = ("CosineSpaDETrainer-othello", "SpaDE-lam1e-03-gam1e-04")
NULLS = ("random_proj", "random_topk")


def read():
    fit = pd.read_csv(RESULTS / "probe_fit.csv", keep_default_na=False)
    readout = pd.read_csv(RESULTS / "readout.csv", keep_default_na=False)
    quality = pd.read_csv(RESULTS / "space_quality.csv")
    return fit, readout, quality


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


# ------------------------------------------------------------------ attack 1

def attack_information(fit, quality):
    print("\n" + "=" * 78)
    print("A1  does macro-F1 track L0 / reconstruction quality?")
    print("=" * 78)
    macro = fit[fit.metric == "macro_f1"].pivot(index="space", columns="target", values="value")
    q = quality.set_index("space")
    t = macro.join(q[["d_space", "l0_window", "l0", "variance_explained", "reconstruction_f1"]])
    t["log_l0"] = np.log(t["l0_window"])
    t = t.sort_values("l0_window")
    print(t[["l0_window", "legality", "board", "variance_explained",
             "reconstruction_f1"]].round(4).to_string())

    sae = t.drop(index=["raw", *NULLS])
    nonspade = t.drop(index=["raw", *SPADE])
    for name, sub in [("all 10 encoded spaces", t.drop(index="raw")),
                      ("8 trained SAEs", sae),
                      ("8 encoded, SpaDE pair dropped", nonspade),
                      ("6 trained SAEs, SpaDE pair dropped", sae.drop(index=list(SPADE)))]:
        for predictor in ("log_l0", "variance_explained", "reconstruction_f1"):
            if sub[predictor].notna().sum() < 3:
                continue
            rs, n = spearman(sub[predictor], sub["legality"])
            rp, _ = pearson(sub[predictor], sub["legality"])
            print(f"  {name:36} {predictor:20} spearman {rs:+.3f}  pearson {rp:+.3f}  n {n}")

    print("\n  sparsity-matched null at k=80")
    for space in ("TopKTrainer-othello", "BatchTopKTrainer-othello", "random_topk"):
        row = t.loc[space]
        print(f"    {space:28} L0 {row.l0_window:8.2f}  legality {row.legality:.4f}  "
              f"board {row.board:.4f}")
    for space in t.drop(index=["raw", *NULLS]).index:
        row = t.loc[space]
        print(f"    {space:28} L0 {row.l0_window:8.2f}  legality {row.legality:.4f}  "
              f"vs random_topk {row.legality - t.loc['random_topk','legality']:+.4f}  "
              f"vs random_proj {row.legality - t.loc['random_proj','legality']:+.4f}")

    # Where does random_proj sit if we interpolate the trained spaces' log-L0 trend?
    trained = sae.drop(index=list(SPADE))
    coef = np.polyfit(trained["log_l0"], trained["legality"], 1)
    for space in [*NULLS, *SPADE]:
        pred = np.polyval(coef, t.loc[space, "log_l0"])
        print(f"  log-L0 fit on 6 trained non-SpaDE SAEs -> {space:28} "
              f"predicted {pred:.4f}  actual {t.loc[space,'legality']:.4f}  "
              f"resid {t.loc[space,'legality'] - pred:+.4f}")
    return t


def attack_drop_agreement(readout, table):
    print("\n" + "=" * 78)
    print("A1b drop_agreement_with_raw against L0")
    print("=" * 78)
    d = readout[(readout.metric == "drop_agreement_with_raw")
                & (readout.category == "illegal_pooled")].set_index("space")
    j = table.join(d[["mean", "lo", "hi", "n"]]).sort_values("l0_window")
    j["ci_width"] = j["hi"] - j["lo"]
    print(j[["l0_window", "legality", "mean", "lo", "hi", "ci_width", "n"]].round(4).to_string())
    for predictor in ("log_l0", "legality"):
        rs, n = spearman(j.drop(index="raw")[predictor], j.drop(index="raw")["mean"])
        print(f"  spearman(drop_agreement, {predictor}) over 10 encoded spaces = {rs:+.3f} n {n}")


# ------------------------------------------------------------------ attack 2

def attack_board(fit):
    print("\n" + "=" * 78)
    print("A2  is legality macro-F1 a function of board macro-F1?")
    print("=" * 78)
    macro = fit[fit.metric == "macro_f1"].pivot(index="space", columns="target", values="value")
    t = macro.sort_values("board")
    rs, n = spearman(t["board"], t["legality"])
    rp, _ = pearson(t["board"], t["legality"])
    print(f"  spearman {rs:+.3f}  pearson {rp:+.3f}  over {n} spaces")

    # monotone (isotonic-by-sorting) fit: legality predicted from board by the pooled
    # adjacent-violators solution, then the residual.
    y = isotonic(t["board"].to_numpy(), t["legality"].to_numpy())
    t = t.assign(iso=y, resid=t["legality"].to_numpy() - y)
    # a linear fit is the honest simpler comparator
    lin = np.polyval(np.polyfit(t["board"], t["legality"], 1), t["board"])
    t = t.assign(lin=lin, resid_lin=t["legality"] - lin)
    print(t[["board", "legality", "iso", "resid", "lin", "resid_lin"]].round(4).to_string())
    print(f"  isotonic residual: max |r| {np.abs(t['resid']).max():.4f}  "
          f"rms {np.sqrt((t['resid']**2).mean()):.4f}")
    print(f"  linear  residual: max |r| {np.abs(t['resid_lin']).max():.4f}  "
          f"rms {np.sqrt((t['resid_lin']**2).mean()):.4f}  "
          f"R2 {1 - (t['resid_lin']**2).sum() / ((t['legality']-t['legality'].mean())**2).sum():.4f}")

    print("\n  the wrong-player arm is the same-class-count, same-difficulty comparator")
    wp = macro["legality_wrong_player"]
    rs, _ = spearman(macro["legality"], wp)
    print(f"  spearman(legality, legality_wrong_player) = {rs:+.3f}; "
          f"max |legality - wrong_player| = {(macro['legality']-wp).abs().max():.4f}")
    return t


def isotonic(x, y):
    """Pool-adjacent-violators on y sorted by x; the least-squares monotone fit."""
    order = np.argsort(x)
    v = y[order].astype(float).copy()
    w = np.ones_like(v)
    i = 0
    while i < len(v) - 1:
        if v[i] <= v[i + 1]:
            i += 1
            continue
        total = w[i] * v[i] + w[i + 1] * v[i + 1]
        weight = w[i] + w[i + 1]
        v = np.delete(v, i + 1)
        w = np.delete(w, i + 1)
        v[i], w[i] = total / weight, weight
        i = max(i - 1, 0)
    out = np.repeat(v, w.astype(int))
    result = np.empty_like(out)
    result[order] = out
    return result


# ------------------------------------------------------------------ attack 3

def attack_r3(readout, fit):
    print("\n" + "=" * 78)
    print("A3  is R3's collapse selective, or what uniform degradation predicts?")
    print("=" * 78)
    rules = readout[(readout.category == "") & (readout.metric == "p_pred_correct_rule")]
    at = rules[rules.r == -1].pivot(index="space", columns="rule", values="mean")
    macro = readout[(readout.category == "illegal_pooled") & (readout.metric == "macro_f1")
                    & (readout.r == -1)].set_index("space")["mean"]
    fitm = fit[fit.metric == "macro_f1"].pivot(index="space", columns="target", values="value")
    t = at.join(macro.rename("macro_f1_r-1")).join(fitm[["legality", "board"]])
    t = t.sort_values("macro_f1_r-1")
    print(t.round(4).to_string())

    print("\n  model 1: per-rule recall as a monotone function of the space's own r=-1 macro-F1")
    for rule in ("R1 placement", "R2 bracketing", "R3 turn"):
        rs, n = spearman(t["macro_f1_r-1"], t[rule])
        print(f"    spearman(macro_f1@r=-1, {rule:14}) {rs:+.3f} n {n}")

    print("\n  model 2: raw's per-class recall powered by the space's relative fidelity.")
    print("  If a space keeps a fraction p of raw's information and each square is an")
    print("  independent draw, recall_space(c) ~ recall_raw(c) * p is the flat model; the")
    print("  question is whether R3's shortfall exceeds R1/R2's in the SAME space.")
    raw = t.loc["raw"]
    for space in t.index:
        row = t[["R1 placement", "R2 bracketing", "R3 turn"]].loc[space]
        ratio = row / raw[["R1 placement", "R2 bracketing", "R3 turn"]]
        print(f"    {space:28} ratio to raw  R1 {ratio.iloc[0]:.3f}  R2 {ratio.iloc[1]:.3f}  "
              f"R3 {ratio.iloc[2]:.3f}   R3/R2 {ratio.iloc[2]/ratio.iloc[1]:.3f}")

    print("\n  fit-set per-class recall (independent corpus, no branches involved)")
    rec = fit[(fit.target == "legality") & (fit.metric == "per_class_recall")
              & fit["class"].isin(["legal", "R1", "R2", "R3"])]
    print(rec.pivot(index="space", columns="class", values="value").round(4).to_string())

    print("\n  same panel for the OTHER low-performing space: random_topk")
    print("  and the pooled-category p_pred_correct_rule at r=-1 by category")
    cats = readout[(readout.metric == "p_pred_correct_rule") & (readout.r == -1)
                   & (readout.category != "")]
    print(cats.pivot(index="space", columns="category", values="mean").round(4).to_string())
    return t


# ------------------------------------------------------------------ attack 4/5

def attack_drift_null(readout):
    print("\n" + "=" * 78)
    print("A4  the r<0 -> r=0 drop, measured on branches that contain no violation")
    print("=" * 78)
    d = readout[readout.metric.isin(["drop_macro_f1", "excess_macro_f1"])]
    piv = d.pivot_table(index="space", columns=["metric", "role", "category"], values="mean")
    cols = [("drop_macro_f1", "reference", "legal"),
            ("drop_macro_f1", "control_legal", "legal"),
            ("drop_macro_f1", "control_legal_2", "legal"),
            ("drop_macro_f1", "illegal", "illegal_pooled"),
            ("excess_macro_f1", "illegal", "illegal_pooled")]
    print(piv[cols].round(4).to_string())

    print("\n  macro_f1 curve over r for raw, by role (is the r<0 block flat?)")
    m = readout[(readout.metric == "macro_f1") & (readout.space == "raw")
                & readout.role.isin(["reference", "illegal"])
                & readout.category.isin(["legal", "illegal_pooled"])]
    print(m.pivot(index="r", columns="role", values="mean").round(4).head(14).to_string())


def attack_bootstrap(readout, fit):
    print("\n" + "=" * 78)
    print("A5  reported n, and CI widths")
    print("=" * 78)
    print("  probe_fit n column:", sorted(fit.n.unique()))
    print("  readout n column, by role:")
    print(readout.groupby("role")["n"].agg(["min", "max"]).to_string())
    print("\n  drop_agreement_with_raw n and CI half-width, illegal_pooled")
    d = readout[(readout.metric == "drop_agreement_with_raw")
                & (readout.category == "illegal_pooled")]
    out = d.set_index("space")[["mean", "lo", "hi", "n"]].copy()
    out["half_width"] = (out["hi"] - out["lo"]) / 2
    out["implied_sd"] = out["half_width"] / 1.96 * np.sqrt(out["n"])
    print(out.round(5).to_string())

    print("\n  agreement_with_raw at r<0 vs r=0 (raw's own row must be exactly 1)")
    a = readout[(readout.metric == "agreement_with_raw")
                & (readout.category == "illegal_pooled")]
    print(a.pivot(index="space", columns="r", values="mean")[[-8, -1, 0, 1, 24]].round(4).to_string())


def main():
    fit, readout, quality = read()
    table = attack_information(fit, quality)
    attack_drop_agreement(readout, table)
    attack_board(fit)
    attack_r3(readout, fit)
    attack_drift_null(readout)
    attack_bootstrap(readout, fit)


if __name__ == "__main__":
    sys.exit(main())
