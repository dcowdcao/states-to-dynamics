"""Step 3: fit both probes in one space and score every position of the branch tree.

One space per invocation, resumable (`--spaces` filters, existing outputs are skipped), so
the eleven spaces are a SLURM array rather than one long job. The roster is
`legality_probe.spaces.roster()` -- raw plus the eight seed-42 dictionaries -- with that
study's two untrained controls riding along: `random_topk` is the sparsity-matched null and
`random_proj` the information-preservation ceiling, and they are not interchangeable
(paper_style.SPACE_STYLE says why).

The read-out set is the violation-divergence spliced branch tree: 34,546 sequences of 59
moves, each identical to a legal source game except at t0, where it plays the reference
move, a legal control, or one illegal move of a known category. That is the "one illegal
move at a known index" corpus this study needs, and its activations already exist.

**Ground truth exists at t <= t0 and nowhere after it.** The branch keeps the source
game's tokens[t0+1:], which is a legal continuation of the *reference's* board and of no
other; after an illegal move no board exists for it to be legal on, and the engine cannot
replay one (`OthelloBoardState.umpire` asserts). So the probe is written out at all 59
positions -- the curve past t0 is a real measurement and worth seeing -- but every
localization statistic is taken over t in [0, t0], and the legal control branch of the same
game at the same t0 is the matched null for it.

Writes per space, into study_dir("move_legality")/readout/:
    p_illegal_<space>.pt    [34546, 59] fp32   1 - P(legal) from the 2-class probe
    proba4_<space>.pt       [34546, 59, 4] fp32  the 4-class posterior
    scores_<space>.json     val/test metrics and the l2 sweep for both probes

Usage: python -m spade.move_legality.readout [--spaces raw] [--smoke]
"""

import argparse
import json
from dataclasses import asdict
from functools import cache

import pandas as pd
import torch

from spade.common.config import CTX_LEN, D_MODEL, LAYER
from spade.common.device import select_device
from spade.common.paths import repo_root
from spade.common.study import study_dir
from spade.legality_probe.spaces import NULLS, roster
from spade.move_legality import probe as probe_module
from spade.move_legality.fit_set import CLASSES, SPLITS
from spade.spaces import load_space
from spade.violation_divergence.extract_acts import spans

ENCODE_ROWS = 4096      # fit-set rows per encode call
ROW_BATCH = 248         # branch rows per span; `spans` never splits a game
N_CLASSES = (2, len(CLASSES))
SMOKE_ROWS = 2000


def fit_set() -> tuple[dict, torch.Tensor]:
    d = study_dir("move_legality")
    fit = {name: torch.load(d / "fit_set" / f"{name}.pt", weights_only=True)
           for name in ["game_id", "position", "token", "category", "label", "split"]}
    acts = torch.load(d / "acts" / f"resid_post_{LAYER}.pt", map_location="cpu",
                      weights_only=True)
    assert acts.shape == (len(fit["label"]), D_MODEL), acts.shape
    return fit, acts


@cache
def branch_set(smoke: bool) -> tuple[dict, torch.Tensor]:
    """The read-out tree and its 2.1 GB of acts, loaded once and only when a probe needs them."""
    d = study_dir("divergence")
    meta = torch.load(d / "branches" / "branches.pt", map_location="cpu", weights_only=True)
    acts = torch.load(d / "acts" / f"resid_post_{LAYER}.pt", map_location="cpu",
                      weights_only=True)
    assert acts.shape[:2] == (len(meta["t0"]), CTX_LEN), acts.shape
    if smoke:
        keep = slice(0, SMOKE_ROWS)
        meta, acts = {k: v[keep] for k, v in meta.items()}, acts[keep]
    return meta, acts


def encode(fn, acts: torch.Tensor, device: str, rows: int = ENCODE_ROWS) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([fn(acts[i:i + rows].float().to(device)).half().cpu()
                          for i in range(0, len(acts), rows)])


def split_rows(fit: dict, rows: torch.Tensor) -> dict[str, torch.Tensor]:
    """`rows` cut by the split its source game belongs to."""
    game_split = fit["split"][fit["game_id"][rows].long()]
    return {name: rows[game_split == i] for i, name in enumerate(SPLITS)}


def fit_probe(fit: dict, feats: torch.Tensor, n_classes: int, space: str,
              device: str) -> tuple[probe_module.LinearProbe, dict]:
    rows = probe_module.balanced(fit, n_classes)
    by_split = split_rows(fit, rows)
    y = {name: probe_module.targets(fit["label"][r], n_classes) for name, r in by_split.items()}
    probe, sweep = probe_module.fit(
        feats[by_split["train"]], y["train"], feats[by_split["val"]], y["val"],
        space, n_classes, device)
    scores = {name: asdict(probe_module.score(probe, feats[by_split[name]], y[name]))
              for name in SPLITS}
    return probe, {"n_classes": n_classes, "l2": probe.l2, "l2_sweep": sweep,
                   "n_rows": {name: len(r) for name, r in by_split.items()},
                   "scores": scores}


def read_branches(probes: dict[int, probe_module.LinearProbe], fn, acts: torch.Tensor,
                  game_id: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    """[n, 59] p_illegal and [n, 59, 4] posterior, one `spans` batch at a time.

    Never materializes the [n, 59, 8192] code, which is 33 GB in a dictionary space.
    """
    out = {"p_illegal": torch.empty(len(acts), CTX_LEN),
           "proba4": torch.empty(len(acts), CTX_LEN, len(CLASSES))}
    with torch.no_grad():
        for k, (start, end) in enumerate(spans(game_id, ROW_BATCH)):
            rows = acts[start:end].float().to(device)
            code = fn(rows.flatten(0, 1)).unflatten(0, rows.shape[:2])
            out["p_illegal"][start:end] = probes[2].p_illegal(code).cpu()
            out["proba4"][start:end] = probes[len(CLASSES)].proba(code).cpu()
            if k % 20 == 0:
                print(f"read out {end}/{len(acts)} branches", flush=True)
    return out


def quality_rows(report: dict) -> list[dict]:
    """One tidy row per (space, n_classes, split): what the probe scores off the fit set."""
    return [{"space": report["space"], "d_space": report["d_space"], "n_classes": n,
             "split": split, "l2": probe["l2"], "n_rows": probe["n_rows"][split],
             **{k: round(v, 5) for k, v in probe["scores"][split].items() if isinstance(v, float)},
             **{f"f1_{c}": round(f, 5) for c, f in probe["scores"][split]["per_class_f1"].items()}}
            for n, probe in report["probes"].items() for split in SPLITS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default="", help="comma-separated filter")
    parser.add_argument("--smoke", action="store_true",
                        help=f"{SMOKE_ROWS} branch rows, into readout_smoke/")
    args = parser.parse_args()

    device = select_device()
    torch.set_grad_enabled(False)
    out = study_dir("move_legality") / ("readout_smoke" if args.smoke else "readout")
    out.mkdir(parents=True, exist_ok=True)

    fit, fit_acts = fit_set()

    keep = set(args.spaces.split(",")) if args.spaces else None
    # raw and the nulls need no checkpoint, so a filter to them never opens the SAE registry
    if keep is not None and keep <= {"raw", *NULLS}:
        chosen = [(n, None) for n in ("raw", *NULLS) if n in keep]
    else:
        chosen = [s for s in roster() + [(n, None) for n in NULLS]
            if keep is None or s[0] in keep]

    for name, entry in chosen:
        report_path = out / f"scores_{name}.json"
        if report_path.exists():
            print(f"skip {name} (cached)", flush=True)
            continue
        fn = load_space(name, entry, device)
        feats = encode(fn, fit_acts, device)
        print(f"{name}: fit set encoded to {tuple(feats.shape)}", flush=True)

        probes, report = {}, {"space": name, "d_space": feats.shape[-1], "probes": {}}
        for n_classes in N_CLASSES:
            with torch.enable_grad():
                probes[n_classes], report["probes"][str(n_classes)] = fit_probe(
                    fit, feats, n_classes, name, device)
            probe_module.save(probes[n_classes])
            test = report["probes"][str(n_classes)]["scores"]["test"]
            print(f"{name:28} {n_classes}-class  test acc {test['accuracy']:.4f}  "
                  f"macro-F1 {test['macro_f1']:.4f}  AUROC {test['auroc']:.4f}", flush=True)
        del feats

        meta, branch_acts = branch_set(args.smoke)
        read = read_branches(probes, fn, branch_acts, meta["game_id"], device)
        for key, tensor in read.items():
            torch.save(tensor, out / f"{key}_{name}.pt")
        report_path.write_text(json.dumps(report, indent=2))

    if args.smoke:
        return
    rows = [row for path in sorted(out.glob("scores_*.json"))
            for row in quality_rows(json.loads(path.read_text()))]
    csv = repo_root() / "analysis" / "move_legality" / "results" / "probe_quality.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"wrote {csv} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
