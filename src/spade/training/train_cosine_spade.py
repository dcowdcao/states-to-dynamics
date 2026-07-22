"""Train the CosineSpaDE SAE.  Novel — spade.saes.spade_cosine.CosineSpaDETrainer.

Uses data-sampled prototype init (via core). Supports the γ sweep.
Entrypoint: `python -m spade.training.train_cosine_spade --save_dir checkpoints/cosine_spade
             --gamma 1.0 --device cuda:0`
"""

# def config(gamma): return {"trainer_class": "CosineSpaDETrainer", ...}
# def main(): run_training(config(args.gamma), ...)   # + --cosine_spade_sweep grid
