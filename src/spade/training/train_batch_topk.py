"""Train the BatchTopK SAE.  Baseline from circuits.dictionary_learning.BatchTopKTrainer.

k=80 to match TopKTrainer for a direct head-to-head comparison. Per-batch top-k selection
at train time + a learned EMA threshold for inference (threshold_beta=0.999,
threshold_start_step=1000), so per-token L0 varies around 80 instead of being pinned at 80.
lr is omitted — the trainer auto-picks 2e-4 / sqrt(8192/16384) ≈ 2.83e-4 (same auto-LR
convention as TopKTrainer).

Entrypoint: `python -m spade.training.train_batch_topk --save_dir checkpoints/batch_topk --device cuda:0`
"""

from spade.training.core import TrainParams, run_cli  # also puts circuits on sys.path

from circuits.dictionary_learning.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer


def config(p: TrainParams) -> dict:
    return {
        "trainer": BatchTopKTrainer,
        "dict_class": BatchTopKSAE,
        "activation_dim": p.activation_dim,
        "dict_size": p.dict_size,
        "k": 80,
        "auxk_alpha": 1 / 32,
        "threshold_beta": 0.999,
        "threshold_start_step": 1000,
        "warmup_steps": p.warmup_steps,
        "steps": p.steps,
        "seed": p.seed,
        "layer": p.layer,
        "lm_name": p.model_name,
        "wandb_name": "BatchTopKTrainer-othello",
        "device": p.device,
    }


def main():
    run_cli(config, __doc__)


if __name__ == "__main__":
    main()
