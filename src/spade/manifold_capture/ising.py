"""Pairwise Ising model over binarized SAE codes (paper Sec. 5 / Def. 6).

Pseudo-likelihood fit, one l1-logistic regression per node: with spins in
{-1,+1}, P(s_i=+1 | s_-i) = sigmoid(2 h_i + 2 sum_j J_ij s_j), so couplings
are half the logistic weights. Fields are absorbed into intercepts and not
returned. Per-node regularization is chosen by EBIC (gamma=0.25) over C_GRID.

Selftest: python -m spade.manifold_capture.ising --selftest
"""

import sys

import numpy as np

MAX_NODES = 300
MAX_SAMPLES = 30000
C_GRID = (0.05, 0.1, 0.5, 1.0, 5.0)
EBIC_GAMMA = 0.25
EDGE_FLOOR = 1e-4
FIRE_QUANTILE = 0.10
SEED = 42


def thresholds(codes: np.ndarray) -> np.ndarray:
    """Per-atom FIRE_QUANTILE of nonzero code values (inf if the atom never fires)."""
    out = np.full(codes.shape[1], np.inf)
    for a in range(codes.shape[1]):
        nz = codes[:, a][codes[:, a] > 0]
        if len(nz):
            out[a] = np.quantile(nz, FIRE_QUANTILE)
    return out


def binarize(codes: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Spins in {-1,+1}: +1 where code >= per-atom threshold."""
    return np.where(codes >= thr, 1, -1).astype(np.int8)


def fit_ising(s: np.ndarray) -> np.ndarray:
    """Symmetrized couplings J [m, m] from spins s [N, m] in {-1,+1}."""
    from sklearn.linear_model import LogisticRegression
    n, m = s.shape
    assert m <= MAX_NODES, f"{m} nodes > {MAX_NODES}"
    assert n <= MAX_SAMPLES, f"{n} samples > {MAX_SAMPLES}"
    x = s.astype(np.float32)
    w_full = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        y = s[:, i]
        if y.min() == y.max():
            continue
        xi = np.delete(x, i, axis=1)
        best_ebic, best_w = np.inf, None
        for c in C_GRID:
            clf = LogisticRegression(l1_ratio=1.0, C=c, solver="liblinear",
                                     max_iter=1000, tol=1e-4, random_state=SEED)
            clf.fit(xi, y)
            margin = y * (xi @ clf.coef_[0] + clf.intercept_[0])
            ll = -np.logaddexp(0.0, -margin).sum()
            df = np.count_nonzero(clf.coef_[0])
            ebic = -2 * ll + df * np.log(n) + 2 * EBIC_GAMMA * df * np.log(m - 1)
            if ebic < best_ebic:
                best_ebic, best_w = ebic, clf.coef_[0]
        w_full[i, np.arange(m) != i] = best_w / 2  # J_ij = w_ij / 2 for ±1 spins
    return (w_full + w_full.T) / 2


def communities(J: np.ndarray) -> np.ndarray:
    """Louvain labels [m] on the |J|-weighted graph (edges above EDGE_FLOOR)."""
    import networkx as nx
    m = J.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(m))
    for i, j in zip(*np.triu_indices(m, k=1)):
        if abs(J[i, j]) > EDGE_FLOOR:
            g.add_edge(int(i), int(j), weight=float(abs(J[i, j])))
    labels = np.empty(m, dtype=np.int64)
    for c, comm in enumerate(nx.community.louvain_communities(
            g, weight="weight", seed=SEED)):
        labels[list(comm)] = c
    return labels


def signed_cohesion(J: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Def. 6: rho(G) = mean sign(J_ab) over intra-community pairs (nan for singletons)."""
    out = {}
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        if len(members) < 2:
            out[int(c)] = float("nan")
            continue
        sub = J[np.ix_(members, members)]
        out[int(c)] = float(np.sign(sub[np.triu_indices(len(members), k=1)]).mean())
    return out


def _selftest():
    """Planted 2-block Ising (positive intra, zero cross) -> pipeline must
    recover both blocks with rho ~ +1."""
    rng = np.random.default_rng(0)
    block, m, n = 6, 12, 3000
    j_true = np.zeros((m, m))
    j_true[:block, :block] = 0.2
    j_true[block:, block:] = 0.2
    np.fill_diagonal(j_true, 0.0)

    s = rng.choice([-1.0, 1.0], size=(n, m))
    for _ in range(200):
        for i in range(m):
            p = 1.0 / (1.0 + np.exp(-2.0 * (s @ j_true[:, i])))
            s[:, i] = np.where(rng.random(n) < p, 1.0, -1.0)

    J = fit_ising(s.astype(np.int8))
    labels = communities(J)
    assert len(set(labels[:block])) == 1 and len(set(labels[block:])) == 1, labels
    assert labels[0] != labels[block], labels
    rho = signed_cohesion(J, labels)
    assert all(r > 0.9 for r in rho.values()), rho
    cross = np.abs(J[:block, block:]).mean()
    intra = J[:block, :block][np.triu_indices(block, k=1)].mean()
    print(f"selftest ok: rho={rho}, mean intra J={intra:.3f}, mean |cross J|={cross:.4f}")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--selftest"], "importable module; only --selftest runs"
    _selftest()
