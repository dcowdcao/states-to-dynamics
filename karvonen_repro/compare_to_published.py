"""Decide whether a harness run reproduced Karvonen's published Othello table.

    python karvonen_repro/compare_to_published.py --harness <run>/results.csv

Criteria and their justification live in karvonen_repro/PROTOCOL.md. Exit code is
0 for REPRODUCED, 1 for FAILED.
"""

import argparse, re, sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / (
    "external/Karvonen_Research/autoencoders/saved_data/"
    "06-10-othello-trained-results_with_07-10_topk.csv"
)
VALIDATION_SET = Path(__file__).resolve().parent / "validation_set.csv"

FUNCS = [
    "games_batch_to_state_stack_mine_yours_BLRRC",
    "games_batch_to_state_stack_mine_yours_blank_mask_BLRRC",
    "games_batch_to_valid_moves_BLRRC",
]

# Published frac_alive was produced by an evaluation.py that is not in the tree:
# for the 240 non-top_k rows it equals l0/dict_size exactly, for the 40 top_k rows
# it is exactly 1/dict_size. Never reportable as a pass.
EXCLUDED = {"frac_alive": "published values come from code not in the tree (PROTOCOL.md section 2)"}


class ComparisonError(Exception):
    pass


@dataclass(frozen=True)
class Tol:
    tier: str  # G gate, D deterministic, R reconstruction, S sampling
    abs: float = 0.0
    rel: float = 0.0
    exact_str: bool = False
    elementwise: bool = False
    advisory: bool = False


def tolerances() -> dict[str, Tol]:
    tols = {
        "loss_original": Tol("G", abs=1e-4),
        "loss_zero": Tol("G", abs=1e-4),
        "eval_sae_n_inputs": Tol("G"),
        "eval_results_n_inputs": Tol("G"),
        "board_reconstruction_n_inputs": Tol("G"),
        "trainer_class": Tol("G", exact_str=True),
        "sae_class": Tol("G", exact_str=True),
        "loss_reconstructed": Tol("D", abs=1e-3),
        "frac_recovered": Tol("D", abs=5e-4),
        "num_alive_features": Tol("R", rel=0.05),
        "l0": Tol("S", rel=0.25),  # tightened to rel 1e-3 for TrainerTopK below
        "cossim": Tol("S", abs=0.01),
        "l2_ratio": Tol("S", abs=0.02),
        "l2_loss": Tol("S", rel=0.15),
        "l1_loss": Tol("S", rel=0.40),
        "frac_variance_explained": Tol("S", abs=0.02, advisory=True),
    }
    for f in FUNCS:
        tols[f"{f}_num_squares"] = Tol("G")
        tols[f"{f}_board_reconstruction_board_count"] = Tol("G")
        tols[f"{f}_best_idx"] = Tol("R", abs=1)
        for col in ("zero", "best", "last"):
            tols[f"{f}_{col}_f1_score_per_class"] = Tol("R", abs=0.01)
        for col in ("zero_L0", "best_L0"):
            tols[f"{f}_{col}"] = Tol("R", rel=0.05)
        for stat in (
            "num_true_positive_squares",
            "num_false_positive_squares",
            "num_false_negative_squares",
            "multiple_classes",
            "num_true_and_false_positive_squares",
        ):
            tols[f"{f}_best_{stat}"] = Tol("R", rel=0.05)
        for col in ("high_precision_counts_per_T", "high_precision_and_recall_counts_per_T"):
            tols[f"{f}_{col}"] = Tol("R", rel=0.10, elementwise=True)
    return tols


ABBREV = {
    "games_batch_to_state_stack_mine_yours_blank_mask_BLRRC": "blank_mask",
    "games_batch_to_state_stack_mine_yours_BLRRC": "mine_yours",
    "games_batch_to_valid_moves_BLRRC": "valid_moves",
}


def abbreviate(col: str) -> str:
    for long, s in ABBREV.items():
        col = col.replace(long, s)
    return col


def checkpoint_id(autoencoder_path: str) -> str:
    parts = autoencoder_path.rstrip("/").split("/")
    return f"{parts[-2].replace('othello-trained_model-layer_5-', '')}/{parts[-1]}"


def numbers(cell: object) -> list[float]:
    """Pull the numeric payload out of a serialised tensor or JSON list."""
    found = re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(cell))
    if not found:
        raise ComparisonError(f"no numbers in list-valued cell: {cell!r}")
    return [float(x) for x in found]


def check(published: object, reproduced: object, tol: Tol) -> tuple[bool, str]:
    if pd.isna(published) or pd.isna(reproduced):
        return False, f"nan (published={published!r} reproduced={reproduced!r})"
    if tol.exact_str:
        ok = str(published) == str(reproduced)
        return ok, f"{published!r} vs {reproduced!r}"
    if tol.elementwise:
        p, r = numbers(published), numbers(reproduced)
        if len(p) != len(r):
            return False, f"length {len(p)} vs {len(r)}"
        worst = max(zip(p, r), key=lambda pr: abs(pr[0] - pr[1]) - tol.rel * abs(pr[0]))
        ok = abs(worst[0] - worst[1]) <= max(tol.abs, tol.rel * abs(worst[0]))
        return ok, f"worst element {worst[0]:.6g} vs {worst[1]:.6g}"
    delta = abs(float(published) - float(reproduced))
    ok = delta <= max(tol.abs, tol.rel * abs(float(published)))
    return ok, f"{float(published):.6g} vs {float(reproduced):.6g} (delta {delta:.3g})"


def load(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "autoencoder_path" not in df.columns:
        raise ComparisonError(f"{label} csv has no autoencoder_path column: {path}")
    df = df.assign(checkpoint=df["autoencoder_path"].map(checkpoint_id))
    dupes = df["checkpoint"][df["checkpoint"].duplicated()].tolist()
    if dupes:
        raise ComparisonError(f"{label} csv has duplicate checkpoints: {dupes}")
    return df.set_index("checkpoint")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", type=Path, required=True)
    ap.add_argument("--published", type=Path, default=PUBLISHED)
    ap.add_argument("--validation-set", type=Path, default=VALIDATION_SET)
    args = ap.parse_args()

    pub, rep = load(args.published, "published"), load(args.harness, "harness")
    wanted = pd.read_csv(args.validation_set)
    wanted = [f"{r.group}/{r.trainer}" for r in wanted.itertuples()]

    missing_pub = [c for c in wanted if c not in pub.index]
    if missing_pub:
        raise ComparisonError(f"validation set names checkpoints absent from the published csv: {missing_pub}")
    absent = [c for c in wanted if c not in rep.index]
    checkpoints = [c for c in wanted if c in rep.index]
    if not checkpoints:
        raise ComparisonError("harness csv contains none of the validation-set checkpoints")

    tols = tolerances()
    results: dict[str, dict[str, tuple[bool, str]]] = {}
    for col, tol in tols.items():
        if col not in pub.columns:
            raise ComparisonError(f"published csv is missing {col}; the tolerance table is stale")
        results[col] = {}
        for cp in checkpoints:
            if col not in rep.columns:
                results[col][cp] = (False, "column absent from harness csv")
                continue
            t = Tol("S", rel=1e-3) if col == "l0" and pub.at[cp, "trainer_class"] == "TrainerTopK" else tol
            results[col][cp] = check(pub.at[cp, col], rep.at[cp, col], t)

    short = {cp: f"{i:02d}" for i, cp in enumerate(checkpoints)}
    print("checkpoints")
    for cp, i in short.items():
        print(f"  {i}  {cp}")
    print("\npass/fail per column per checkpoint  (. pass, F fail)\n")
    print(f"{'tier column':<48} {' '.join(short.values())}")
    for col, tol in tols.items():
        cells = " ".join("." if results[col][cp][0] else "F" for cp in checkpoints)
        flag = "*" if tol.advisory else " "
        print(f"{tol.tier}{flag} {abbreviate(col):<45} {cells}")
    print("\n* advisory: counted and reported, never fails the run (PROTOCOL.md section 3)")

    for col, reason in EXCLUDED.items():
        print(f"EXCLUDED {col}: {reason}")
    unchecked = sorted(set(pub.columns) - set(tols) - set(EXCLUDED) - {"autoencoder_path"})
    print(f"UNCHECKED {len(unchecked)} published columns carry no pass criterion, e.g. {unchecked[:4]}")
    if absent:
        print(f"ABSENT {len(absent)} validation-set checkpoints are not in the harness csv: {absent}")

    failures = [
        (tol.tier, col, cp, why)
        for col, tol in tols.items()
        for cp, (ok, why) in results[col].items()
        if not ok and not tol.advisory
    ]
    if failures:
        print(f"\nfailures ({len(failures)})")
        for tier, col, cp, why in failures:
            print(f"  {tier}  {cp:<28} {abbreviate(col):<50} {why}")

    advisories = [
        (col, cp, why)
        for col, tol in tols.items()
        if tol.advisory
        for cp, (ok, why) in results[col].items()
        if not ok
    ]
    if advisories:
        print(f"\nadvisory deviations ({len(advisories)})")
        for col, cp, why in advisories:
            print(f"     {cp:<28} {abbreviate(col):<50} {why}")

    print("\nset-level")
    set_level_ok = True
    f1 = f"{FUNCS[0]}_best_f1_score_per_class"
    for col in ("l0", f1):
        rho = spearmanr(pub.loc[checkpoints, col], rep.loc[checkpoints, col]).statistic
        ok = rho >= 0.98
        set_level_ok &= bool(ok)
        print(f"  spearman rho {abbreviate(col):<50} {rho:.4f}  {'pass' if ok else 'FAIL (need >= 0.98)'}")
    bias = ((rep.loc[checkpoints, "l0"] - pub.loc[checkpoints, "l0"]) / pub.loc[checkpoints, "l0"]).median()
    ok = abs(bias) <= 0.05
    set_level_ok &= bool(ok)
    print(f"  median signed relative l0 deviation{'':<40} {bias:+.4f}  {'pass' if ok else 'FAIL (need within +/- 5%)'}")

    by_tier = {t: [f for f in failures if f[0] == t] for t in "GDRS"}
    n_recon = sum(1 for col, tol in tols.items() if tol.tier == "R") * len(checkpoints)
    recon_rate = 1 - len(by_tier["R"]) / n_recon
    print(f"\ntier G {len(by_tier['G'])} failures | tier D {len(by_tier['D'])} | "
          f"tier R {len(by_tier['R'])}/{n_recon} ({recon_rate:.1%} pass) | tier S {len(by_tier['S'])}")

    reproduced = (
        not absent
        and not by_tier["G"]
        and not by_tier["D"]
        and recon_rate >= 0.90
        and set_level_ok
    )
    print(f"\nVERDICT: {'REPRODUCED' if reproduced else 'FAILED'}")
    if by_tier["G"]:
        print("tier G failed: the evaluation protocol itself does not match. Fix that before reading anything else.")
    return 0 if reproduced else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ComparisonError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
