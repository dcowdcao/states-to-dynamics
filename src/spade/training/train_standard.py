"""Train the Standard (ReLU) SAE.  Baseline from circuits.dictionary_learning.StandardTrainer.

Entrypoint: `python -m spade.training.train_standard --save_dir checkpoints/main --device cuda:0`
"""

# def config(): return {"trainer_class": "StandardTrainer", ...}   # Pareto-selected hparams
# def main(): run_training(config(), ...)   # shared flags: --save_dir --device --seed --smoke_test
