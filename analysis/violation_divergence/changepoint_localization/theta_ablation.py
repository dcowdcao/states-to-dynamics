"""Curvature ablation: the changepoint pipeline with `theta` in place of `d`."""
import numpy as np
from pathlib import Path

B = Path("/n/netscratch/ba_lab/Lab/dcao")
DIV, STR = B/"spade_divergence"/"curvature_discriminant", B/"spade_straightness"/"geometry"
REF = "raw"
SPACES = ["raw", "StandardTrainer-othello", "PAnnealTrainer-othello", "GatedSAETrainer-othello",
          "GatedAnnealTrainer-othello", "TopKTrainer-othello", "BatchTopKTrainer-othello",
          "SpaDE-lam1e-03-gam1e-04", "CosineSpaDETrainer-othello"]
SHORT = {"raw":"raw L5","StandardTrainer-othello":"Standard","PAnnealTrainer-othello":"PAnneal",
         "GatedSAETrainer-othello":"GatedSAE","GatedAnnealTrainer-othello":"GatedAnneal",
         "TopKTrainer-othello":"TopK","BatchTopKTrainer-othello":"BatchTopK",
         "SpaDE-lam1e-03-gam1e-04":"SpaDE","CosineSpaDETrainer-othello":"CosineSpaDE"}
N_LEGAL, VIOLATION, M, PAIRED, SEED, N_BOOT, N_SHIFT, OFF = 2000, 2, 4, True, 42, 1000, 20, 2

stem = lambda s: "resid_post_5" if s == REF else f"sae_{s}"
load = lambda d, s, k, suf="": np.load(d/f"{stem(s)}{suf}.npz")[k].astype(np.float64)

def scan(g, cand, m=M, paired=PAIRED):
    c = np.concatenate([np.zeros((len(g), 1)), g.cumsum(1)], 1)
    w = lambda a: (c[:, a+m] - c[:, a]) / m
    return np.sqrt(m/2)*(w(cand) - w(cand-m)) if paired else np.sqrt(m)*w(cand)

def auroc(pos, neg):
    x = np.concatenate([pos, neg]); order = np.argsort(x, kind="mergesort"); sx = x[order]
    r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j+1 < len(x) and sx[j+1] == sx[i]: j += 1
        r[i:j+1] = (i+j)/2 + 1; i = j+1
    ranks = np.empty(len(x)); ranks[order] = r
    p, n = len(pos), len(neg)
    return (ranks[:p].sum() - p*(p+1)/2) / (p*n)

def boot_ci(stat, n, seed=SEED):
    rng = np.random.default_rng(seed)
    d = [stat(rng.integers(n, size=n)) for _ in range(N_BOOT)]
    return float(np.quantile(d, .025)), float(np.quantile(d, .975))

def shift_chance(s, cand, t0, seed=SEED):
    rng = np.random.default_rng(seed); hits = []
    for _ in range(N_SHIFT):
        off = rng.integers(1, s.shape[1], (len(s), 1))
        idx = (np.arange(s.shape[1]) + off) % s.shape[1]
        hits.append(cand[np.take_along_axis(s, idx, 1).argmax(1)] + OFF == t0)
    return float(np.mean(hits))

meta = dict(np.load(DIV/"meta.npz"))
gid, cat, ctl = meta["game_id"], meta["category"], meta["is_control"]
games, rng = np.unique(gid), np.random.default_rng(SEED)
pick = lambda mask, g: np.flatnonzero((gid == g) & mask)
ARMS = {"illegal":  np.array([rng.choice(pick((cat > 0) & (ctl == 0), g)) for g in games]),
        "control":  np.array([pick((cat == 0) & (ctl == 1), g)[0] for g in games]),
        "control2": np.array([pick((cat == 0) & (ctl == 2), g)[0] for g in games]),
        "reference":np.array([pick((cat == 0) & (ctl == 0), g)[0] for g in games])}
T0 = meta["t0"][ARMS["reference"]]
held = set(meta["held_out_legal"].tolist())
FIT = [i for i in range(N_LEGAL) if i not in held]

def run(key, cand, sign=+1):
    rows = []
    for sp in SPACES:
        legal = load(STR, sp, key, "_raw")[:N_LEGAL][FIT]
        mu, sd = np.nanmean(legal, 0), np.nanstd(legal, 0)
        x = load(DIV, sp, key)
        g_all = sign * -(x - mu) / sd
        nan = int(np.isnan(g_all).sum())
        g_all = np.nan_to_num(g_all, nan=0.0)
        s = {a: scan(g_all[r], cand) for a, r in ARMS.items()}
        peak = {a: v.max(1) for a, v in s.items()}
        hit = {a: cand[v.argmax(1)] + OFF == T0 for a, v in s.items()}
        f = lambda i: auroc(peak["illegal"][i], peak["control"][i])
        lo, hi = boot_ci(f, len(T0))
        rows.append(dict(space=sp, auroc=f(np.arange(len(T0))), lo=lo, hi=hi,
                         ref=auroc(peak["illegal"], peak["reference"]),
                         null=auroc(peak["control2"], peak["control"]),
                         hit=float(hit["illegal"].mean()), hit_ctl=float(hit["control"].mean()),
                         chance=shift_chance(s["illegal"], cand, T0), nan=nan))
    return rows

def show(title, rows):
    print("\n" + title)
    hdr = ("space", "AUROC vs ctl", "95% CI", "vs ref", "null", "hit", "chance")
    print("%-12s%14s%18s%9s%8s%8s%8s" % hdr)
    for r in sorted(rows, key=lambda r: -r["auroc"]):
        ci = "[%.3f, %.3f]" % (r["lo"], r["hi"])
        print("%-12s%14.4f%18s%9.4f%8.4f%8.4f%8.4f" % (
            SHORT[r["space"]], r["auroc"], ci, r["ref"], r["null"], r["hit"], r["chance"]))
    print("   NaNs filled to 0 after standardization: %d" % sum(r["nan"] for r in rows))

wd = load(DIV, REF, "d").shape[1]; wt = load(DIV, REF, "theta").shape[1]
print(f"widths: d={wd} theta={wt}")
cand_d = np.arange(12, wd-12+1); cand_m = np.arange(12, wt-12+1)
print(f"native d candidates={len(cand_d)}  matched candidates={len(cand_m)} "
      f"(moves {cand_m[0]+OFF}..{cand_m[-1]+OFF})")
show(f"A. STEP LENGTH d, native window ({len(cand_d)} cand) -- reproduces published", run("d", cand_d))
show(f"B. STEP LENGTH d, matched window ({len(cand_m)} cand)", run("d", cand_m))
show(f"C. CURVATURE theta, matched window, same sign (positive = straighter)", run("theta", cand_m))
show(f"D. CURVATURE theta, matched window, FLIPPED sign (positive = curvier)", run("theta", cand_m, sign=-1))
