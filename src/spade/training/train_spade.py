"""Train the SpaDE (distance) SAE.  Novel — spade.saes.spade.SpaDETrainer.

Uses data-sampled prototype init (via core). Supports the λ/γ sweep flags.
Entrypoint: `python -m spade.training.train_spade --save_dir checkpoints/spade_sweep
             --lambda_init 1e-3 --gamma 1e-2 --device cuda:0`
"""

# def config(lambda_init, gamma): return {"trainer_class": "SpaDETrainer", ...}
# def main(): run_training(config(args.lambda_init, args.gamma), ...)   # + --spade_pairs sweep
