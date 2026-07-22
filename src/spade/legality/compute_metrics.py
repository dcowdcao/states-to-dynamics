"""Legality step 3 — the heavy geometry/probe analysis over every registered SAE + baselines.

Encodes the paired activations through each SAE space and asks whether/where legal vs
illegal moves separate: per-t centroid geometry, kNN manifold distances, weight-decay-tuned
logistic probes (+ shuffled and held-out-token controls, a prefix square-legality probe),
token-matched contrasts, difference-in-differences quads, feature signatures, PCA. Writes
aggregate CSVs/JSON to results/legality/ and PCA inputs to analysis/geometry/legal_vs_illegal/.

Entrypoint: `python -m spade.legality.compute_metrics`
"""

# def load_space(entry): ...          # encode acts through one SAE
# def run_probes(...): ...            # logistic probes + controls
# def token_matched_pairs(...): ...
# def did_quads(...): ...             # difference-in-differences
# def analyze_space(entry): ...       # everything for one SAE
# def write_outputs(...): ...         # aggregate.csv, probe_results.csv, feature_signatures.json, ...
# def main(): ...
