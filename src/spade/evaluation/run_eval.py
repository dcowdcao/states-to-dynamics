"""Pipeline stage 3 — evaluate one SAE checkpoint under the Karvonen protocol.

Computes the 5 headline metrics (loss/frac recovered, L0, variance explained, coverage,
reconstruction F1) and appends one row (file-locked, idempotent) to
`results/<experiment>/all_results.csv`. Loop over checkpoints in the shell driver.

Entrypoint: `python -m spade.evaluation.run_eval --checkpoint_dir checkpoints/main/trainer_3
             --test_data_dir data/activations --output_dir results/main --device auto`
"""

# def compute_l0(f): ...
# def compute_variance_explained(x, x_hat): ...
# def compute_frac_recovered(model, sae, ...): ...   # behavioral fidelity (next-move logits)
# def main(): ...   # load sae via karvonen_adapter, run metrics, append CSV row
