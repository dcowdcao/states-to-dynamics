"""Joins the shipped CSVs with the re-derived ones and prints every table in RED_TEAM.md.

Usage: python analysis/legality_probe/redteam/verdict_tables.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SHIPPED = HERE.parent / "results"
MINE = HERE / "results"
SPADE = ["CosineSpaDETrainer-othello", "SpaDE-lam1e-03-gam1e-04"]
NULLS = ["random_proj", "random_topk"]


def spearman(a, b):
    return float(np.corrcoef(pd.Series(np.asarray(a, float)).rank(),
                             pd.Series(np.asarray(b, float)).rank())[0, 1])


def load():
    fit = pd.read_csv(SHIPPED / "probe_fit.csv", keep_default_na=False)
    readout = pd.read_csv(SHIPPED / "readout.csv", keep_default_na=False)
    quality = pd.read_csv(SHIPPED / "space_quality.csv").set_index("space")
    rt_fit = pd.read_csv(MINE / "rt_fit.csv")
    frame = pd.read_csv(MINE / "rt_frame.csv").set_index("space")
    window = pd.read_csv(MINE / "rt_window.csv")
    ship = fit[fit.metric == "macro_f1"].pivot(index="space", columns="target", values="value")
    mine = rt_fit.pivot_table(index="space", columns="arm", values="macro_f1")
    seeds = rt_fit[rt_fit.arm == "legality"].pivot(index="space", columns="seed",
                                                   values="macro_f1")
    table = pd.DataFrame({
        "l0": quality.l0_window, "ship_legality": ship.legality, "my_legality": mine.legality,
        "seed_spread": (seeds[1] - seeds[0]).abs(),
        "ship_board": ship.board, "board_oracle": mine.board_oracle_legality,
        "linear_r2": mine.linear_r2, "occupancy": frame.occupancy_macro_f1,
        "colour": frame.colour_accuracy_given_both_occupied,
        "frameless": frame.legality_frameless_macro_f1,
        "p_pred_legal": frame.p_predicts_legal, "p_pred_R3": frame.p_predicts_R3,
    })
    return fit, readout, table, window


def a1(t):
    print("\n== A1  nulls, sparsity, and information preservation ==")
    t = t.sort_values("l0")
    print(t[["l0", "ship_legality", "my_legality", "seed_spread", "linear_r2",
             "occupancy"]].round(4).to_string())
    encoded = t.drop(index="raw")
    trained = encoded.drop(index=NULLS)
    ordinary = trained.drop(index=SPADE)
    for name, sub in [("10 encoded", encoded), ("8 trained", trained),
                      ("6 ordinary trained", ordinary),
                      ("8 non-SpaDE encoded", encoded.drop(index=SPADE))]:
        print(f"  {name:22} spearman(log L0, legality) {spearman(np.log(sub.l0), sub.my_legality):+.3f}"
              f"   spearman(linear_r2, legality) {spearman(sub.linear_r2, sub.my_legality):+.3f}"
              f"   spearman(occupancy, legality) {spearman(sub.occupancy, sub.my_legality):+.3f}"
              f"   n {len(sub)}")
    coefficients = np.polyfit(np.log(ordinary.l0), ordinary.my_legality, 1)
    for space in NULLS + SPADE:
        predicted = np.polyval(coefficients, np.log(t.loc[space, "l0"]))
        print(f"  log-L0 trend of the 6 ordinary SAEs -> {space:28} predicted {predicted:.4f} "
              f"actual {t.loc[space,'my_legality']:.4f}  residual "
              f"{t.loc[space,'my_legality']-predicted:+.4f}")
    matched = t.loc[["TopKTrainer-othello", "BatchTopKTrainer-othello", "random_topk"]]
    print("  k = 80 matched:  " + "  ".join(
        f"{s} {v:.4f}" for s, v in matched.my_legality.items()))


def a2(t):
    print("\n== A2  is the legality arm anything but the board arm? ==")
    t = t.assign(residual=t.my_legality - t.board_oracle).sort_values("board_oracle")
    print(t[["ship_board", "board_oracle", "my_legality", "residual", "seed_spread"]]
          .round(4).to_string())
    print(f"  pearson(board_oracle, legality) over 11 spaces {np.corrcoef(t.board_oracle, t.my_legality)[0,1]:+.4f}"
          f"   spearman {spearman(t.board_oracle, t.my_legality):+.3f}")
    print(f"  the oracle beats the direct legality probe in {int((t.residual < 0).sum())}/11 spaces; "
          f"residual mean {t.residual.mean():+.4f} range {t.residual.min():+.4f}..{t.residual.max():+.4f}")
    ordinary = t.drop(index=SPADE)
    print(f"  excluding the SpaDE pair the residual is {ordinary.residual.min():+.4f}.."
          f"{ordinary.residual.max():+.4f}, span {ordinary.residual.max()-ordinary.residual.min():.4f}, "
          f"against a seed spread of {ordinary.seed_spread.max():.4f}")


def a3(t, readout, fit):
    print("\n== A3  is R3's collapse selective? ==")
    rules = readout[(readout.category == "") & (readout.metric == "p_pred_correct_rule")
                    & (readout.r == -1)].pivot(index="space", columns="rule", values="mean")
    macro = readout[(readout.category == "illegal_pooled") & (readout.metric == "macro_f1")
                    & (readout.r == -1)].set_index("space")["mean"]
    per_class = fit[(fit.target == "legality") & (fit.metric == "per_class_f1")].pivot(
        index="space", columns="class", values="value")
    panel = rules.join(macro.rename("macro_r-1")).join(
        per_class[["legal", "R1", "R2", "R3"]].add_prefix("fit_f1_")).join(
        t[["occupancy", "colour", "frameless", "my_legality", "p_pred_legal", "p_pred_R3"]])
    panel["frameless_gain"] = panel.frameless - panel.my_legality
    print(panel.sort_values("macro_r-1").round(4).to_string())

    ordinary = panel.drop(index=SPADE)
    for predictor in ("macro_r-1", "R2 bracketing"):
        coefficients = np.polyfit(ordinary[predictor], ordinary["R3 turn"], 1)
        print(f"\n  linear model  R3 ~ {predictor}, fit on the 9 non-SpaDE spaces "
              f"(slope {coefficients[0]:+.3f}, R2 "
              f"{np.corrcoef(ordinary[predictor], ordinary['R3 turn'])[0,1]**2:.3f})")
        for space in SPADE:
            predicted = np.polyval(coefficients, panel.loc[space, predictor])
            print(f"    {space:28} predicted {predicted:+.4f}  actual "
                  f"{panel.loc[space,'R3 turn']:.4f}  shortfall "
                  f"{panel.loc[space,'R3 turn'] - predicted:+.4f}")

    print("\n  class prevalence in the fit set vs the damage ordering")
    balance = pd.read_csv(SHIPPED / "class_balance.csv", keep_default_na=False)
    pooled = balance[balance.position == ""].set_index("label")["fraction"]
    print("   ", pooled.to_dict())


def a4(readout, window):
    print("\n== A4/A5  the r<0 control and the bootstrap ==")
    drop = readout[readout.metric.isin(["drop_macro_f1", "drop_agreement_with_raw"])]
    piv = drop.pivot_table(index="space", columns=["metric", "role", "category"], values="mean")
    cols = [("drop_macro_f1", "reference", "legal"),
            ("drop_macro_f1", "control_legal", "legal"),
            ("drop_macro_f1", "illegal", "illegal_pooled"),
            ("drop_agreement_with_raw", "reference", "legal"),
            ("drop_agreement_with_raw", "control_legal", "legal"),
            ("drop_agreement_with_raw", "illegal", "illegal_pooled")]
    out = piv[cols]
    out.columns = ["dropF1_ref", "dropF1_ctrl", "dropF1_ill",
                   "dropAgr_ref", "dropAgr_ctrl", "dropAgr_ill"]
    print(out.round(4).to_string())

    identical = readout[(readout.metric == "macro_f1") & (readout.r == -1)]
    spread = identical.groupby("space")["mean"].agg(lambda s: s.max() - s.min())
    print(f"\n  max spread of macro_f1 at r = -1 across the 13 branch sets, by space: "
          f"{spread.max():.5f} (all sets are one measurement)")

    ill = window[window.set == "illegal_pooled"].set_index("space")
    ill = ill.assign(
        game_ci=ill.drop_agreement_hi - ill.drop_agreement_lo,
        row_ci=ill.drop_agreement_row_hi - ill.drop_agreement_row_lo)
    ill["design_effect"] = (ill.game_ci / ill.row_ci) ** 2
    print("\n  my game bootstrap vs a row bootstrap on drop_agreement_with_raw")
    print(ill[["drop_agreement", "game_ci", "row_ci", "design_effect",
               "drop_agreement_sd_over_games", "drop_agreement_sd_over_rows",
               "branch_weighted_f1_before", "game_weighted_f1_before"]].round(4).to_string())


def main():
    fit, readout, table, window = load()
    a1(table)
    a2(table)
    a3(table, readout, fit)
    a4(readout, window)


if __name__ == "__main__":
    main()
