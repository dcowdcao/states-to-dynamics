"""Subspace-capture metrics for the manifold-capture study (paper Fig 6 + App E).

For each (space, concept, variant), how well decoder atoms capture the concept
manifold M (the standardized-space corpus, raw or per-t-centered; M's mean is
subtracted before every R^2):

  greedy        span protocol: greedily add the decoder atom capturing the most
                residual variance; R^2 of orthogonal projection onto the span
  greedy_codes  code-masked protocol: same atom order, R^2 of the centered
                masked-code reconstruction — what the SAE actually computes
  pca / random  optimal-linear and random-orthonormal baselines (space="baseline")

Per space it also records support size and receptive-field spread (App E),
k_M (PCA components to 90% VE) and ve_manifold (full-code reconstruction R^2).

Variants: "raw" and "centered" share M across a corpus's concepts (every
position carries every concept), so those results are computed once per
(corpus, variant) and emitted per concept — they measure capture of the JOINT
corpus manifold. The "curve" variant is the per-concept Fig 6 analog: M is the
concept's value-binned centroid curve (per-t-centered ambient except for t
itself), which averages out nuisance variance and traces the concept's own
manifold; its k_M and greedy curves are genuinely concept-specific.

Appends to results/manifold_capture/capture_{curves,metrics}.csv; spaces whose
rows all exist are skipped (resumable).

Usage: python -m spade.manifold_capture.subspace_capture [--spaces TopKTrainer-othello]
       python -m spade.manifold_capture.subspace_capture --selftest
"""

import argparse
import csv
from pathlib import Path

import torch

from spade.common.device import select_device
from spade.common.paths import results_dir
from spade.manifold_capture import concepts as C
from spade.manifold_capture import spaces as S
from spade.spaces import batched_apply

SUPPORT_PCTL = 0.10      # threshold = this quantile of an atom's nonzero activations
SUPPORT_ROW_FRAC = 0.10  # supporting: fires above threshold on >= this fraction of rows
MIN_SUPPORT_ROWS = 30    # ... and on >= this many rows
RF_CAP = 2000            # pairwise-distance subsample cap
KM_VE = 0.90
CURVE_MIN_BIN = 30       # value bins with fewer rows are dropped from the centroid curve
RANDOM_SEED = 0
RF_SEED = 0
ATOM_CHUNK = 1024
SCORE_CHUNK = 2048
RECON_BATCH = 8192

CURVE_FIELDS = ["space", "concept", "variant", "method", "n", "r2"]
METRIC_FIELDS = ["space", "concept", "variant", "support_size", "rf_spread",
                 "k_M", "ve_manifold", "low_ve"]


def _sq(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).sum(dtype=torch.float64)


def pca_curve(M_c: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(cumulative VE over all principal components, k_M = components to 90% VE)."""
    evals = torch.linalg.eigvalsh((M_c.T @ M_c).double()).flip(0).clamp_min(0)
    cum = (evals.cumsum(0) / evals.sum()).float().cpu()
    return cum, int((cum >= KM_VE).nonzero()[0]) + 1


def random_curve(M_c: torch.Tensor, max_n: int) -> torch.Tensor:
    """Cumulative VE of projection onto fixed-seed random orthonormal directions."""
    g = torch.Generator().manual_seed(RANDOM_SEED)
    d = M_c.shape[1]
    q, _ = torch.linalg.qr(torch.randn(d, min(max_n, d), generator=g).to(M_c))
    return ((M_c @ q).pow(2).sum(0, dtype=torch.float64).cumsum(0) / _sq(M_c)).float().cpu()


def greedy_span(M_c: torch.Tensor, D_cand: torch.Tensor, max_n: int) -> tuple[list[int], list[float]]:
    """Greedy pursuit over decoder columns (reference find_support_greedy): each
    step adds the atom whose single direction captures the most residual
    variance, then re-fits the orthogonal projection onto the selected span."""
    total = _sq(M_c)
    d_norms = D_cand.pow(2).sum(0).clamp_min(1e-10)
    residual = M_c.clone()
    order, curve = [], []
    for _ in range(min(max_n, D_cand.shape[1])):
        scores = torch.cat([
            (residual @ D_cand[:, j:j + SCORE_CHUNK]).pow(2).sum(0)
            for j in range(0, D_cand.shape[1], SCORE_CHUNK)]) / d_norms
        scores[order] = -torch.inf
        best = int(scores.argmax())
        if scores[best] <= 0:
            break
        order.append(best)
        u, s, _ = torch.linalg.svd(D_cand[:, order], full_matrices=False)
        basis = u[:, s > 1e-8]
        residual = M_c - (M_c @ basis) @ basis.T
        curve.append(float(1 - _sq(residual) / total))
    assert all(b >= a - 1e-5 for a, b in zip(curve, curve[1:])), "greedy curve not monotone"
    return order, curve


def codes_curve(M_c: torch.Tensor, Z_c: torch.Tensor, D_sel: torch.Tensor) -> list[float]:
    """R^2 of the centered masked-code reconstruction at each prefix of the
    greedy order (Z_c columns and D_sel columns follow that order)."""
    total = _sq(M_c)
    recon = torch.zeros_like(M_c)
    curve = []
    for j in range(Z_c.shape[1]):
        recon += Z_c[:, j:j + 1] * D_sel[:, j]
        curve.append(float(1 - _sq(M_c - recon) / total))
    return curve


def value_bins(vals: torch.Tensor) -> list[torch.Tensor]:
    """Row-index groups per distinct concept value (all concepts are integer- or
    class-valued), dropping bins below CURVE_MIN_BIN rows."""
    groups = [(vals == v).nonzero().flatten() for v in vals.unique().tolist()]
    return [g for g in groups if len(g) >= CURVE_MIN_BIN]


def bin_means(X: torch.Tensor, bins: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([X[b].mean(0) for b in bins])


def support_stats(codes: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(threshold [F], supporting [F], firing [F]) — App E: threshold is the
    10th percentile of an atom's nonzero activations; supporting = fires above
    it on >=10% of manifold rows and >=30 rows. firing = any nonzero."""
    n = len(codes)
    thrs, above, firing = [], [], []
    for j in range(0, codes.shape[1], ATOM_CHUNK):
        chunk = codes[:, j:j + ATOM_CHUNK].to(device)
        k = (chunk > 0).sum(0)
        vals = chunk.sort(0).values  # ascending: zeros first, nonzeros in last k slots
        pos = (n - k).double() + SUPPORT_PCTL * (k - 1).clamp_min(0).double()
        lo, hi = pos.floor().long().clamp(max=n - 1), pos.ceil().long().clamp(max=n - 1)
        thr = torch.lerp(vals.gather(0, lo[None])[0], vals.gather(0, hi[None])[0],
                         (pos - pos.floor()).float())
        thr[k == 0] = torch.inf
        thrs.append(thr.cpu())
        above.append((chunk > thr).sum(0).cpu())
        firing.append((k > 0).cpu())
    thr, above = torch.cat(thrs), torch.cat(above)
    supporting = (above >= MIN_SUPPORT_ROWS) & (above.float() >= SUPPORT_ROW_FRAC * n)
    return thr, supporting, torch.cat(firing)


def mean_pairwise(X: torch.Tensor, g: torch.Generator) -> float:
    if len(X) > RF_CAP:
        X = X[torch.randperm(len(X), generator=g)[:RF_CAP]]
    m = len(X)
    return float(torch.cdist(X, X).sum() / (m * (m - 1)))


def rf_spread(M: torch.Tensor, codes: torch.Tensor, thr: torch.Tensor,
              supporting: torch.Tensor, device: str) -> float:
    """Median over supporting atoms of (mean pairwise distance among the atom's
    above-threshold rows) / (manifold mean pairwise distance), in ambient M."""
    if not supporting.any():
        return float("nan")
    g = torch.Generator().manual_seed(RF_SEED)
    Md = M.to(device)
    base = mean_pairwise(Md, g)
    spreads = [mean_pairwise(Md[(codes[:, j] > thr[j]).nonzero().flatten()], g) / base
        for j in supporting.nonzero().flatten().tolist()]
    return float(torch.tensor(spreads).median())


def r2_recon(M: torch.Tensor, recon: torch.Tensor) -> float:
    return float(1 - _sq(M - recon) / _sq(M - M.mean(0)))


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spaces", default="", help="comma-separated filter (smoke tests)")
    ap.add_argument("--concepts", default="", help="comma-separated filter")
    ap.add_argument("--cap", type=int, default=60000)
    ap.add_argument("--max_atoms", type=int, default=128)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    device = select_device()
    out_dir = results_dir("manifold_capture")
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_path, metrics_path = out_dir / "capture_curves.csv", out_dir / "capture_metrics.csv"
    curves, metrics = read_rows(curves_path), read_rows(metrics_path)
    curve_keys = {(r["space"], r["concept"], r["variant"], r["method"]) for r in curves}
    metric_keys = {(r["space"], r["concept"], r["variant"]) for r in metrics}

    space_list = [(n, e) for n, e in S.spaces() if n != "raw"]
    if args.spaces:
        space_list = [s for s in space_list if s[0] in args.spaces.split(",")]
    all_concepts = C.concepts()
    if args.concepts:
        all_concepts = [c for c in all_concepts if c.name in args.concepts.split(",")]

    for corpus in ["games", "rules"]:
        cs = [c for c in all_concepts if c.corpus == corpus]
        if not cs:
            continue
        expected = [(c, v) for c in cs for v in c.variants] + [(c, "curve") for c in cs]
        need_baseline = [(c, v) for c, v in expected
            if ("baseline", c.name, v, "pca") not in curve_keys]
        need_spaces = [(n, e) for n, e in space_list
            if not all((n, c.name, v) in metric_keys for c, v in expected)]
        if not need_baseline and not need_spaces:
            print(f"skip corpus {corpus} (all rows exist)")
            continue

        print(f"corpus {corpus}: loading")
        data = C.load_corpus(corpus)
        idx = C.subsample(len(data["acts"]), args.cap)
        acts, t, legal = data["acts"][idx], data["t"][idx], data["legal"][idx]
        vals_all = {c.name: C.values(c, data)[idx] for c in cs}
        del data
        ambient = batched_apply(S.encode_fn("raw", None, device), acts, device)
        Ms = {"raw": ambient}
        if any(v == "centered" for _, v in expected) or any(c.name != "t" for c in cs):
            Ms["centered"] = C.center_by_t(ambient, t, legal)  # curve targets need it too

        base = {}  # variant -> (pca cum over all components, random cum, k_M)
        for v, M in Ms.items():
            M_c = (M - M.mean(0)).to(device)
            pca_cum, k_m = pca_curve(M_c)
            assert pca_cum[-1] >= 0.99, f"pca reaches only {pca_cum[-1]:.4f} at n={len(pca_cum)}"
            rand_cum = random_curve(M_c, args.max_atoms)
            assert (rand_cum <= pca_cum[:len(rand_cum)] + 1e-6).all(), "random beats pca"
            base[v] = (pca_cum, rand_cum, k_m)
            del M_c

        # per-concept centroid-curve manifolds ("curve" variant): t bins raw
        # ambient (t IS the swept variable); every other concept bins the
        # per-t-centered ambient so the curve traces the concept, not t
        curve_data = {}  # concept -> (M_curve cpu, M_c device, bins, k_M, pca, rand)
        for c in cs:
            bins = value_bins(vals_all[c.name])
            assert len(bins) >= 2, f"{c.name}: fewer than 2 populated value bins"
            src = ambient if c.name == "t" else Ms["centered"]
            M_curve = bin_means(src, bins)
            M_c = (M_curve - M_curve.mean(0)).to(device)
            pca_cum, k_m = pca_curve(M_c)
            rand_cum = random_curve(M_c, args.max_atoms)
            curve_data[c.name] = (M_curve, M_c, bins, k_m, pca_cum, rand_cum)

        for c, v in need_baseline:
            if v == "curve":
                _, _, _, _, pca_cum, rand_cum = curve_data[c.name]
            else:
                pca_cum, rand_cum, _ = base[v]
            for method, cum in [("pca", pca_cum[:args.max_atoms]), ("random", rand_cum)]:
                curves += [{"space": "baseline", "concept": c.name, "variant": v,
                            "method": method, "n": i + 1, "r2": f"{float(r):.6f}"}
                    for i, r in enumerate(cum)]
        write_rows(curves_path, CURVE_FIELDS, curves)

        for name, entry in need_spaces:
            print(f"{corpus}/{name}: encoding")
            codes = batched_apply(S.encode_fn(name, entry, device), acts, device)
            S.check_codes(name, codes)
            thr, supporting, firing = support_stats(codes, device)
            n_support = int(supporting.sum())
            if entry is not None:
                D, bias = S.decoder(entry, device)
                assert D.shape == (ambient.shape[1], codes.shape[1])
                recon = torch.cat([
                    (codes[i:i + RECON_BATCH].to(device) @ D.T + bias).cpu()
                    for i in range(0, len(codes), RECON_BATCH)])
                ves = {"raw": r2_recon(ambient, recon)}
                if "centered" in Ms:
                    ves["centered"] = r2_recon(Ms["centered"], C.center_by_t(recon, t, legal))
                cand = (supporting if n_support >= args.max_atoms else firing) \
                    .nonzero().flatten()
                D_cand = D[:, cand.to(device)]

            rfs = {}
            for v, M in Ms.items():
                relevant = [c for c in cs if v in c.variants]
                done = all((name, c.name, v) in metric_keys for c in relevant) and \
                    (entry is None or all((name, c.name, v, "greedy") in curve_keys
                                          for c in relevant))
                if done and relevant:
                    rfs[v] = float(next(r["rf_spread"] for r in metrics
                        if r["space"] == name and r["variant"] == v))
                    continue
                rf = rfs[v] = rf_spread(M, codes, thr, supporting, device)
                if entry is not None:
                    M_c = (M - M.mean(0)).to(device)
                    order, span_curve = greedy_span(M_c, D_cand, args.max_atoms)
                    assert span_curve, f"{name}/{v}: greedy selected nothing"
                    sel = cand[torch.tensor(order)]
                    Z = codes[:, sel]
                    cc = codes_curve(M_c, (Z - Z.mean(0)).to(device), D_cand[:, order])
                    del M_c
                for c in relevant:
                    if entry is not None and (name, c.name, v, "greedy") not in curve_keys:
                        for method, curve in [("greedy", span_curve), ("greedy_codes", cc)]:
                            curves += [{"space": name, "concept": c.name, "variant": v,
                                        "method": method, "n": i + 1, "r2": f"{r:.6f}"}
                                for i, r in enumerate(curve)]
                    if (name, c.name, v) not in metric_keys:
                        metrics.append({
                            "space": name, "concept": c.name, "variant": v,
                            "support_size": n_support, "rf_spread": f"{rf:.4f}",
                            "k_M": base[v][2],
                            "ve_manifold": f"{ves[v]:.6f}" if entry is not None else "",
                            "low_ve": name in S.LOW_VE})
                print(f"{corpus}/{name}/{v}: support={n_support} rf={rf:.3f}")

            for c in cs:  # per-concept centroid-curve capture
                if (name, c.name, "curve") in metric_keys:
                    continue
                M_curve, M_c, bins, k_m, _, _ = curve_data[c.name]
                if entry is not None:
                    order, span_curve = greedy_span(M_c, D_cand, args.max_atoms)
                    Z = bin_means(codes, bins)[:, cand[torch.tensor(order)]]
                    cc = codes_curve(M_c, (Z - Z.mean(0)).to(device), D_cand[:, order])
                    for method, curve in [("greedy", span_curve), ("greedy_codes", cc)]:
                        curves += [{"space": name, "concept": c.name, "variant": "curve",
                                    "method": method, "n": i + 1, "r2": f"{r:.6f}"}
                            for i, r in enumerate(curve)]
                    recon_src = recon if c.name == "t" else C.center_by_t(recon, t, legal)
                    ve = r2_recon(M_curve, bin_means(recon_src, bins))
                metrics.append({
                    "space": name, "concept": c.name, "variant": "curve",
                    "support_size": n_support,
                    "rf_spread": f"{rfs.get('centered', rfs['raw']):.4f}",
                    "k_M": k_m,
                    "ve_manifold": f"{ve:.6f}" if entry is not None else "",
                    "low_ve": name in S.LOW_VE})
                print(f"{corpus}/{name}/curve:{c.name}: k_M={k_m} bins={len(bins)}")
            write_rows(curves_path, CURVE_FIELDS, curves)
            write_rows(metrics_path, METRIC_FIELDS, metrics)
            del codes
        del acts, ambient, Ms


def selftest():
    """Synthetic 3-atom manifold in 16-D: greedy must recover the true atoms."""
    torch.manual_seed(0)
    n, d = 400, 16
    z = torch.rand(n, 3) + 0.1
    atoms = torch.randn(d, 3)
    M = z @ atoms.T
    M_c = M - M.mean(0)
    D = torch.cat([atoms, torch.randn(d, 7)], 1)
    codes = torch.zeros(n, 10)
    codes[:, :3] = z
    codes[:, 3:] = torch.rand(n, 7) * (torch.rand(n, 7) < 0.05)

    order, curve = greedy_span(M_c, D, 5)
    assert sorted(order[:3]) == [0, 1, 2] and curve[2] > 0.99, (order, curve)

    sel = torch.tensor(order[:3])
    Z = codes[:, sel]
    cc = codes_curve(M_c, Z - Z.mean(0), D[:, sel])
    assert cc[-1] > 0.99, cc

    thr, supporting, firing = support_stats(codes, "cpu")
    assert supporting[:3].all() and not supporting[3:].any(), supporting
    assert firing.all()

    rf = rf_spread(M, codes, thr, supporting, "cpu")
    assert 0.5 < rf < 1.5, rf

    pca_cum, k_m = pca_curve(M_c)
    assert pca_cum[-1] > 0.999 and k_m <= 3, (pca_cum[-1], k_m)
    rand = random_curve(M_c, 8)
    assert (rand <= pca_cum[:8] + 1e-6).all()
    print(f"selftest ok: greedy r2@3={curve[2]:.4f} codes r2@3={cc[-1]:.4f} "
          f"support={int(supporting.sum())} rf_spread={rf:.3f} k_M={k_m}")


if __name__ == "__main__":
    main()
