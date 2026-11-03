"""Independent re-derivation of the legality-probe study: own probe module, own fitting
loop, own labels, own metrics, own bootstrap.

`spade.legality_probe` is never imported. What is imported is infrastructure that is not
under test — `spade.spaces.load_space` (the checkpoint loader), `study_dir`, and the
Othello engine — plus nothing else. The shipped CSVs are compared against these arrays,
never used to produce them.

Adds four quantities the study does not report, because they are the attacks:

  linear_r2          held-out R^2 of a least-squares map code -> standardized raw. The
                     information-preservation predictor A1 needs.
  board_oracle_f1    legality macro-F1 obtained by decoding the 3-class board arm and
                     computing legality from the predicted board. Same 4 classes, same
                     metric as the legality arm, so the residual carries no class-count
                     confound.
  seed spread        every arm fitted at two seeds; the between-space residuals have to
                     clear this.
  row bootstrap      the same CI resampled over branch rows instead of source games.

Usage: python analysis/legality_probe/redteam/rederive.py [--spaces a,b] [--smoke]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels_indep import board_from_disc_labels, legal_squares, legality  # noqa: E402

from spade.common.config import LAYER  # noqa: E402
from spade.common.paths import add_karvonen_to_path  # noqa: E402
from spade.common.study import study_dir  # noqa: E402
from spade.spaces import batched_apply, load_space, representative_spaces  # noqa: E402

add_karvonen_to_path()

from circuits.othello_engine_utils import OthelloBoardState, to_string  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
WINDOW = tuple(range(-8, 25))
N_BEFORE = sum(r < 0 for r in WINDOW)
R0 = WINDOW.index(0)
RM1 = WINDOW.index(-1)
N_TRAIN = 8000
ROW_BATCH = 248
ENCODE_BATCH = 8192
ROLE_DIM, LR, BETAS, WEIGHT_DECAY, EPOCHS, BATCH = 52, 1e-2, (0.9, 0.99), 2.0, 120, 1024
CATEGORY_NAMES = ["legal", "occupied_own", "occupied_opp", "empty_detached",
                  "empty_own_only", "empty_no_bracket", "opponent_only"]
LABEL_OF_CATEGORY = np.array([0, 1, 1, 2, 2, 2, 3], dtype=np.int8)
RULE_OF_CATEGORY = {1: "R1 placement", 2: "R1 placement", 3: "R2 bracketing",
                    4: "R2 bracketing", 5: "R2 bracketing", 6: "R3 turn"}
N_BOOT = 2000
BOOT_SEED = 12345


# ---------------------------------------------------------------- my TPR probe

class Probe(torch.nn.Module):
    """logit(s, c) = role_s^T (sum_d h_d B_d) filler_c, written from the equation."""

    def __init__(self, d_model, n_classes, role_dim=ROLE_DIM, filler_dim=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.binding = torch.nn.Parameter(
            torch.randn(d_model, role_dim, filler_dim, generator=g) / d_model ** 0.5)
        self.role = torch.nn.Parameter(
            torch.randn(64, role_dim, generator=g) / role_dim ** 0.5)
        self.filler = torch.nn.Parameter(
            torch.randn(n_classes, filler_dim, generator=g) / filler_dim ** 0.5)

    def forward(self, h):
        binding = torch.einsum("bd,drf->brf", h, self.binding)
        return torch.einsum("brf,sr,cf->bsc", binding, self.role, self.filler)


def fit_probe(acts, labels, n_classes, filler_dim, seed, device):
    """acts [n, d] float32 on device, labels [n, 64] int64 on device."""
    torch.manual_seed(seed)
    probe = Probe(acts.shape[-1], n_classes, filler_dim=filler_dim, seed=seed).to(device)
    optimiser = torch.optim.AdamW(probe.parameters(), lr=LR, betas=BETAS,
                                  weight_decay=WEIGHT_DECAY)
    for _ in range(EPOCHS):
        for batch in torch.randperm(acts.shape[0], device=device).split(BATCH):
            logits = probe(acts[batch])
            correct = logits.log_softmax(-1).gather(-1, labels[batch].unsqueeze(-1)).squeeze(-1)
            loss = -n_classes * correct.mean(0).sum()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
    return probe.requires_grad_(False)


@torch.no_grad()
def predict(probe, acts, device, batch=4096):
    return torch.cat([probe(acts[i:i + batch].to(device)).argmax(-1).to(torch.int8).cpu()
                      for i in range(0, len(acts), batch)]).numpy()


# ---------------------------------------------------------------- metrics

def tally(preds, truth, n_classes):
    out = np.empty((*preds.shape[:-1], 3, n_classes), dtype=np.int32)
    for c in range(n_classes):
        p, a = preds == c, truth == c
        out[..., 0, c] = (p & a).sum(-1)
        out[..., 1, c] = (p & ~a).sum(-1)
        out[..., 2, c] = (~p & a).sum(-1)
    return out


def f1_of(counts):
    tp, fp, fn = counts[..., 0, :], counts[..., 1, :], counts[..., 2, :]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def macro_f1(preds, truth, n_classes):
    return float(f1_of(tally(preds.reshape(-1, 64), truth.reshape(-1, 64),
                             n_classes).sum(0)).mean())


def group_sum(values, group, n_groups):
    order = np.argsort(group, kind="stable")
    sorted_group = group[order]
    starts = np.flatnonzero(np.r_[True, sorted_group[1:] != sorted_group[:-1]])
    out = np.zeros((n_groups, *values.shape[1:]), dtype=np.float64)
    out[sorted_group[starts]] = np.add.reduceat(values[order].astype(np.float64), starts, 0)
    return out


def boot_design(n, seed=BOOT_SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1 / n), size=n_boot).astype(np.float32)


def boot_ci(per_unit, statistic, n_boot=N_BOOT, seed=BOOT_SEED):
    """per_unit [n, ...] summable quantities; statistic maps a pooled sum to a scalar."""
    design = boot_design(len(per_unit), seed, n_boot)
    flat = per_unit.reshape(len(per_unit), -1).astype(np.float32)
    draws = statistic((design @ flat).reshape(n_boot, *per_unit.shape[1:]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------- inputs

def fit_targets():
    d = study_dir("legality") / "fit_set"
    tokens = torch.load(d / "tokens.pt", weights_only=True).long().numpy()
    position = torch.load(d / "position.pt", weights_only=True).long().numpy()
    game = torch.load(d / "game_id.pt", weights_only=True).long().numpy()
    shipped = torch.load(d / "labels.pt", weights_only=True).numpy()

    legal = np.empty((len(position), 64), dtype=np.int8)
    board = np.empty_like(legal)
    for i, (g, p) in enumerate(zip(game, position)):
        state = OthelloBoardState()
        state.update(to_string(tokens[g, : p + 1].tolist()))
        legal[i] = legality(state.state, state.next_hand_color)
        flat = state.state.reshape(64)
        board[i] = np.where(flat == 0, 0, np.where(flat == state.next_hand_color, 2, 1))
    assert (legal == shipped).all(), "my labels disagree with the shipped fit set"
    # the oracle is exact on true boards, or the oracle itself is broken
    for i in range(0, len(legal), 977):
        assert (legality(board_from_disc_labels(board[i]), 1) == legal[i]).all()
    assert game[N_TRAIN - 1] != game[N_TRAIN], "the 8000/2000 split is not on a game boundary"
    return legal, board


def branch_inputs():
    d = study_dir("divergence")
    meta = torch.load(d / "branches" / "branches.pt", weights_only=True)
    acts = torch.load(d / "acts" / f"resid_post_{LAYER}.pt", map_location="cpu",
                      weights_only=True)
    pos = meta["t0"].unsqueeze(1) + torch.tensor(WINDOW)
    window = acts[torch.arange(len(pos)).unsqueeze(1), pos]
    return meta, window


def reference_truth(seq, t0):
    offset_of = {t0 + r: j for j, r in enumerate(WINDOW)}
    board = OthelloBoardState()
    truth = np.zeros((len(WINDOW), 64), dtype=np.int8)
    valid = np.zeros(len(WINDOW), dtype=bool)
    for s, token in enumerate(seq[: max(offset_of) + 1]):
        board.umpire(to_string(token))
        if s in offset_of and legal_squares(board.state, board.next_hand_color):
            truth[offset_of[s]] = legality(board.state, board.next_hand_color)
            valid[offset_of[s]] = True
    return truth, valid


def spans(game_id, size=ROW_BATCH):
    change = np.flatnonzero(np.diff(game_id)) + 1
    bounds = [0, *change.tolist(), len(game_id)]
    out, lo = [], 0
    for prev, nxt in zip(bounds, bounds[1:]):
        if nxt - lo > size:
            out.append((lo, prev))
            lo = prev
    out.append((lo, len(game_id)))
    return out


# ---------------------------------------------------------------- per space

def linear_r2(code, raw, device):
    """Best held-out R^2 over a ridge grid for the least-squares map code -> standardized
    raw. A grid is needed because the codes are 8192-wide on 8000 training rows, where an
    unregularised solve interpolates and generalises worse than the mean."""
    x, y = code[:N_TRAIN].to(device).float(), raw[:N_TRAIN].to(device).float()
    xh, yh = code[N_TRAIN:].to(device).float(), raw[N_TRAIN:].to(device).float()
    gram, cross = (x.T @ x).double(), (x.T @ y).double()
    scale = torch.diagonal(gram).mean()
    eye = torch.eye(gram.shape[0], device=device, dtype=torch.float64)
    total = ((yh - y.mean(0)) ** 2).sum()
    best = -np.inf
    for exponent in range(-6, 3):
        w = torch.linalg.solve(gram + (10.0 ** exponent) * scale * eye, cross).float()
        best = max(best, float(1 - ((yh - xh @ w) ** 2).sum() / total))
    return best


def run_space(name, entry, legal_labels, board_labels, raw_acts, window, meta, w, device):
    started = time.time()
    encode = load_space(name, entry, device)
    code = batched_apply(encode, raw_acts, device).float()
    standard = batched_apply(load_space("raw", None, device), raw_acts, device).float()

    rows = []
    fitted = {}
    for arm, labels, n_classes, filler in [("legality", legal_labels, 4, 3),
                                           ("board", board_labels, 3, 2)]:
        for seed in (0, 1):
            probe = fit_probe(code[:N_TRAIN].to(device), torch.from_numpy(labels[:N_TRAIN])
                              .long().to(device), n_classes, filler, seed, device)
            preds = predict(probe, code[N_TRAIN:], device)
            per_class = f1_of(tally(preds, labels[N_TRAIN:], n_classes).sum(0))
            rows.append({"space": name, "arm": arm, "seed": seed,
                         "macro_f1": macro_f1(preds, labels[N_TRAIN:], n_classes),
                         **{f"f1_class{c}": float(per_class[c]) for c in range(n_classes)}})
            fitted.setdefault(arm, {})[seed] = (probe, preds)

    board_preds = fitted["board"][0][1]
    oracle = np.stack([legality(board_from_disc_labels(p), 1) for p in board_preds])
    oracle_f1 = f1_of(tally(oracle, legal_labels[N_TRAIN:], 4).sum(0))
    rows.append({"space": name, "arm": "board_oracle_legality", "seed": 0,
                 "macro_f1": macro_f1(oracle, legal_labels[N_TRAIN:], 4),
                 **{f"f1_class{c}": float(oracle_f1[c]) for c in range(4)}})
    rows.append({"space": name, "arm": "linear_r2", "seed": 0,
                 "macro_f1": linear_r2(code, standard, device)})
    del code, standard

    probe = fitted["legality"][0][0]
    game_id = meta["game_id"].numpy()
    preds = np.empty((len(window), len(WINDOW), 64), dtype=np.int8)
    bitwise_code, bitwise_pred = True, True
    for start, end in spans(game_id):
        block = window[start:end]
        with torch.no_grad():
            span_code = batched_apply(encode, block.flatten(0, 1).float(), device
                                      ).half().unflatten(0, block.shape[:2])
            span = predict(probe, span_code.flatten(0, 1).float(), device
                           ).reshape(*block.shape[:2], 64)
        for g in np.unique(game_id[start:end]):
            rows_g = np.flatnonzero(game_id[start:end] == g)
            bitwise_code &= bool(span_code[rows_g, :N_BEFORE].eq(
                span_code[rows_g[0], :N_BEFORE]).all())
            bitwise_pred &= bool((span[rows_g, :N_BEFORE] == span[rows_g[0], :N_BEFORE]).all())
        del span_code
        preds[start:end] = span
    print(f"{name:28} probes fitted, window decoded in {time.time()-started:.0f}s  "
          f"bitwise code {bitwise_code} pred {bitwise_pred}", flush=True)
    return rows, preds, {"prefix_bitwise_code": bitwise_code,
                         "prefix_bitwise_pred": bitwise_pred}


# ---------------------------------------------------------------- window metrics

def window_metrics(name, preds, raw_preds, w):
    truth, valid, gi = w["truth"], w["valid"], w["game_index"]
    n_games = w["n_games"]
    keep = valid[gi][..., None, None]
    counts = tally(preds, truth[gi], 4) * keep
    agree = (preds == raw_preds).mean(-1)
    played = preds[np.arange(len(preds)), :, w["square"]]

    out = []
    for label, rows in w["sets"].items():
        per_game = group_sum(counts[rows], gi[rows], n_games)
        curve = f1_of(per_game.sum(0)).mean(-1)
        before = per_game[:, :N_BEFORE].sum(1)
        drop = float(f1_of(before.sum(0)).mean() - f1_of(per_game[:, R0].sum(0)).mean())
        lo, hi = boot_ci(np.stack([before, per_game[:, R0]], 1),
                         lambda d: f1_of(d[:, 0]).mean(-1) - f1_of(d[:, 1]).mean(-1))

        # game-weighted vs branch-weighted r<0 F1: at r<0 a game's branches are identical,
        # so branch weighting silently weights games by how many categories they admit.
        n_rows = np.bincount(gi[rows], minlength=n_games)
        present = n_rows > 0
        game_f1 = np.where(present, f1_of(before / np.maximum(n_rows, 1)[:, None, None]
                                          ).mean(-1), np.nan)
        agree_g = group_sum(agree[rows], gi[rows], n_games) / np.maximum(n_rows, 1)[:, None]
        agree_g = np.where(present[:, None], agree_g, np.nan)
        drop_agree = agree_g[:, :N_BEFORE].mean(1) - agree_g[:, R0]
        ok = np.isfinite(drop_agree)
        g_lo, g_hi = boot_ci(np.stack([drop_agree[ok], np.ones(ok.sum())], 1),
                             lambda d: d[:, 0] / d[:, 1])
        design = boot_design(int(len(rows)), BOOT_SEED + 1)
        row_drop = agree[rows][:, :N_BEFORE].mean(1) - agree[rows][:, R0]
        row_draws = (design @ row_drop.astype(np.float32)) / len(rows)
        out.append({
            "space": name, "set": label, "n_rows": int(len(rows)),
            "n_games": int(present.sum()),
            "macro_f1_r-1": float(curve[RM1]), "macro_f1_r0": float(curve[R0]),
            "drop_macro_f1": drop, "drop_lo": lo, "drop_hi": hi,
            "branch_weighted_f1_before": float(f1_of(before.sum(0)).mean()),
            "game_weighted_f1_before": float(np.nanmean(game_f1)),
            "drop_agreement": float(np.nanmean(drop_agree)),
            "drop_agreement_lo": g_lo, "drop_agreement_hi": g_hi,
            "drop_agreement_row_lo": float(np.percentile(row_draws, 2.5)),
            "drop_agreement_row_hi": float(np.percentile(row_draws, 97.5)),
            "agreement_r-1": float(np.nanmean(agree_g[:, RM1])),
            "agreement_r0": float(np.nanmean(agree_g[:, R0])),
            "drop_agreement_sd_over_games": float(np.nanstd(drop_agree, ddof=1)),
            "drop_agreement_sd_over_rows": float(row_drop.std(ddof=1)),
            "p_pred_legal_at_square_r-1": float((played[rows, RM1] == 0).mean()),
            "p_pred_legal_at_square_r0": float((played[rows, R0] == 0).mean()),
        })

    rule_rows = []
    for c in range(1, 7):
        rows = w["by_category"][c]
        if not len(rows):
            continue
        target = LABEL_OF_CATEGORY[c]
        for r_index, r in [(RM1, -1), (R0, 0)]:
            hit = (played[rows, r_index] == target)
            dist = np.bincount(played[rows, r_index], minlength=4) / len(rows)
            rule_rows.append({"space": name, "category": CATEGORY_NAMES[c],
                              "rule": RULE_OF_CATEGORY[c], "r": r, "n_rows": int(len(rows)),
                              "p_correct_rule": float(hit.mean()),
                              **{f"p_pred_{k}": float(dist[k]) for k in range(4)}})
    return out, rule_rows


def build_window(meta):
    game_of = meta["game_id"].numpy()
    category, is_control = meta["category"].numpy(), meta["is_control"].numpy()
    games = sorted(set(game_of.tolist()))
    index = {g: i for i, g in enumerate(games)}
    truth = np.zeros((len(games), len(WINDOW), 64), dtype=np.int8)
    valid = np.zeros((len(games), len(WINDOW)), dtype=bool)
    is_reference = (category == 0) & (is_control == 0)
    for g in games:
        rows = np.flatnonzero(game_of == g)
        reference = rows[is_reference[rows]][0]
        truth[index[g]], valid[index[g]] = reference_truth(
            meta["tokens"][reference].long().tolist(), int(meta["t0"][reference]))
    gi = np.array([index[g] for g in game_of])
    sets = {"reference": np.flatnonzero(is_reference),
            "control_legal": np.flatnonzero(is_control == 1),
            "control_legal_2": np.flatnonzero(is_control == 2),
            "illegal_pooled": np.flatnonzero(category != 0)}
    for c in range(1, 7):
        sets[CATEGORY_NAMES[c]] = np.flatnonzero(category == c)
    return {"truth": truth, "valid": valid, "game_index": gi, "n_games": len(games),
            "square": np.array([to_string(int(t)) for t in meta["branch_token"]]),
            "sets": sets, "by_category": {c: np.flatnonzero(category == c) for c in range(1, 7)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    legal_labels, board_labels = fit_targets()
    raw_acts = torch.load(study_dir("legality") / "acts" / f"resid_post_{LAYER}.pt",
                          map_location="cpu", weights_only=True)
    meta, window = branch_inputs()
    if args.smoke:
        keep = (meta["game_id"] < 40).numpy()
        meta = {k: v[keep] for k, v in meta.items()}
        window = window[keep]
    w = build_window(meta)
    print(f"{len(window)} branches, {w['n_games']} games, "
          f"{int((~w['valid']).sum())}/{w['valid'].size} window positions masked", flush=True)

    roster = [(n, e) for n, e in representative_spaces()
              if n == "raw" or (e is not None and e.seed == 42)]
    roster += [("random_proj", None), ("random_topk", None)]
    if args.spaces:
        keep = set(args.spaces.split(",")) | {"raw"}
        roster = [s for s in roster if s[0] in keep]
    assert roster[0][0] == "raw"

    fit_rows, window_rows, rule_rows, checks, raw_preds = [], [], [], {}, None
    for name, entry in roster:
        rows, preds, check = run_space(name, entry, legal_labels, board_labels, raw_acts,
                                       window, meta, w, device)
        raw_preds = preds if raw_preds is None else raw_preds
        fit_rows += rows
        checks[name] = check
        a, b = window_metrics(name, preds, raw_preds, w)
        window_rows += a
        rule_rows += b
        pd.DataFrame(fit_rows).to_csv(OUT / "rt_fit.csv", index=False)
        pd.DataFrame(window_rows).to_csv(OUT / "rt_window.csv", index=False)
        pd.DataFrame(rule_rows).to_csv(OUT / "rt_rules.csv", index=False)
        (OUT / "rt_checks.json").write_text(json.dumps(checks, indent=2))
    print(pd.DataFrame(fit_rows).pivot_table(index="space", columns="arm",
                                             values="macro_f1").round(4).to_string())


if __name__ == "__main__":
    main()
