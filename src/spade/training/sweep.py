"""Sweep drivers: run one SAE type over a grid of sparsity settings.

Sweeping is a grid problem, not a per-type problem, so it lives beside core.py rather than
inside train_spade.py and train_cosine_spade.py — otherwise each entrypoint grows a copy.
Every off-default λ/γ point in the repo was produced here, under the `SpaDE-lam..-gam..` /
`CosineSpaDE-gam..` names that the eval CSVs and the checkpoint registry match on.

Two hazards this module owns. Trainer indices restart at 0 in every sweep directory, so a
rerun with a different grid silently overwrites checkpoints and desynchronizes them from
their results; a sweep therefore refuses to start on top of a directory that already holds
trainers. And a grid that spans more than one directory must be reassembled by config
name, never by index — see spade.multiseed.aggregate_cosine_sweep.

Pick the range from the loss balance (spade.training.sweep_loss_balance), not by
extrapolating an L0 trend.

Usage:
    python -m spade.training.sweep --save_dir checkpoints/spade_sweep \\
        --spade_sweep --spade_lambdas "1e-3,1e-4" --spade_gammas "1e-4,1e-3,1e-2,1e-1"
    python -m spade.training.sweep --save_dir checkpoints/cosine_spade_sweep \\
        --cosine_spade_sweep --cosine_spade_gammas "1e-1,5e-1,1.0,2.0,5.0"
    python -m spade.training.sweep --save_dir checkpoints/batch_topk_sweep \\
        --batch_topk_sweep --batch_topk_ks "10,20,40,60,80,110,140,200,280,400"
    python -m spade.training.sweep --save_dir checkpoints/main \\
        --spade_pairs "1e-7:5e-3,8e-8:1e-3"
"""

import argparse
from pathlib import Path

from spade.saes.spade import SpaDETrainer
from spade.saes.spade_cosine import CosineSpaDETrainer
from spade.training import (
    core,
    train_batch_topk,
    train_cosine_spade,
    train_gated,
    train_gated_anneal,
    train_panneal,
    train_spade,
    train_standard,
    train_topk,
)

# The eight canonical configs. This order is what fixes trainer_N in checkpoints/main —
# TopK is trainer_3 there — so it is not free to change.
BASELINES = [
    train_panneal.config,
    train_standard.config,
    train_gated.config,
    train_topk.config,
    train_gated_anneal.config,
    train_spade.config,
    train_batch_topk.config,
    train_cosine_spade.config,
]


def _fmt(x: float) -> str:
    """`1e-04` for a one-significant-figure point, `1.62e-03` for the rest.

    `:.0e` collides on a log-spaced grid — 0.00162 and 0.00214 both render `2e-03`, and two
    trainers under one name desync the eval rows from their weights. Plain `:g` would fix
    that but rewrite every existing name, and the literals `SpaDE-lam1e-03-gam1e-04` etc.
    are pinned in saes.registry, figures.paper_style, multiseed.aggregate_seeds and
    export.viewer_weights. Stripping the zero mantissa digits keeps those byte-identical.
    """
    mantissa, exponent = f"{x:.2e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{exponent}"


def spade_config(p: core.TrainParams, lam: float, gam: float) -> dict:
    return {
        "trainer": SpaDETrainer,
        "activation_dim": p.activation_dim,
        "dict_size": p.dict_size,
        "lambda_init": lam,
        "gamma_reg": gam,
        "lr": 3e-4,
        "warmup_steps": p.warmup_steps,
        "steps": p.steps,
        "seed": p.seed,
        "layer": p.layer,
        "lm_name": p.model_name,
        "wandb_name": f"SpaDE-lam{_fmt(lam)}-gam{_fmt(gam)}",
        "device": p.device,
    }


def cosine_config(p: core.TrainParams, gam: float) -> dict:
    return {
        "trainer": CosineSpaDETrainer,
        "activation_dim": p.activation_dim,
        "dict_size": p.dict_size,
        "gamma_reg": gam,
        "lr": 3e-4,
        "warmup_steps": p.warmup_steps,
        "steps": p.steps,
        "seed": p.seed,
        "layer": p.layer,
        "lm_name": p.model_name,
        "wandb_name": f"CosineSpaDE-gam{gam:g}",
        "device": p.device,
    }


def batch_topk_config(p: core.TrainParams, k: int) -> dict:
    """k sets L0 directly, so the grid is the L0 axis. No second body: the canonical
    entrypoint's config takes k, and this only renames the run."""
    return train_batch_topk.config(p, k, f"BatchTopK-k{k}")


# Karvonen's five, so they can move along the L0 axis as our three now can. His own sweep
# cannot stand in: he trains on raw activations where we standardize, so his penalty scale
# does not transfer (his Gated l1 0.958 -> L0 138.5; ours at 1.368 -> L0 50.3), and
# commit 8b95ec9 rewrote standard/p_anneal's recon loss under our pin. Same reason
# aggregate.py refuses to mix the two pipelines under one type name.
#     module, knob, run-name prefix
BASELINES_SWEPT = {
    "standard": (train_standard, "l1_penalty", "Standard"),
    "gated": (train_gated, "l1_penalty", "Gated"),
    "panneal": (train_panneal, "initial_sparsity_penalty", "PAnneal"),
    "gated_anneal": (train_gated_anneal, "initial_sparsity_penalty", "GatedAnneal"),
    "topk": (train_topk, "k", "TopK"),
}


def baseline_config(p: core.TrainParams, kind: str, value: float) -> dict:
    module, knob, prefix = BASELINES_SWEPT[kind]
    shown = f"{int(value)}" if knob == "k" else f"{value:g}"
    return module.config(p, value, f"{prefix}-{knob}{shown}")


def configs(p: core.TrainParams, args: argparse.Namespace) -> list[dict]:
    """The grid, or the eight baselines; then any --spade_pairs appended to it."""
    if args.baseline_sweep:
        kind = args.baseline_sweep
        knob = BASELINES_SWEPT[kind][1]
        values = [(int(x) if knob == "k" else float(x))
                  for x in args.baseline_values.split(",")]
        trainer_configs = [baseline_config(p, kind, v) for v in values]
        print(f"{kind} sweep: {len(trainer_configs)} configs")
        for c in trainer_configs:
            print(f"  {c['wandb_name']}: {knob}={c[knob]}")
    elif args.batch_topk_sweep:
        ks = [int(x) for x in args.batch_topk_ks.split(",")]
        trainer_configs = [batch_topk_config(p, k) for k in ks]
        print(f"BatchTopK sweep: {len(trainer_configs)} configs")
        for c in trainer_configs:
            print(f"  {c['wandb_name']}: k={c['k']}")
    elif args.spade_sweep:
        lambdas = [float(x) for x in args.spade_lambdas.split(",")]
        gammas = [float(x) for x in args.spade_gammas.split(",")]
        trainer_configs = [spade_config(p, lam, gam) for lam in lambdas for gam in gammas]
        print(f"SpaDE sweep: {len(trainer_configs)} configs")
        for c in trainer_configs:
            print(f"  {c['wandb_name']}: lambda={c['lambda_init']}, gamma={c['gamma_reg']}")
    elif args.cosine_spade_sweep:
        gammas = [float(x) for x in args.cosine_spade_gammas.split(",")]
        trainer_configs = [cosine_config(p, gam) for gam in gammas]
        print(f"CosineSpaDE sweep: {len(trainer_configs)} configs")
        for c in trainer_configs:
            print(f"  {c['wandb_name']}: gamma={c['gamma_reg']}")
    else:
        trainer_configs = [build(p) for build in BASELINES]

    if args.spade_pairs:
        pairs = [q.split(":") for q in args.spade_pairs.split(",")]
        extra = [spade_config(p, float(lam), float(gam)) for lam, gam in pairs]
        trainer_configs += extra
        print(f"Appended {len(extra)} SpaDE pair(s): "
              f"{[c['wandb_name'] for c in extra]}")

    return trainer_configs


def refuse_if_populated(save_dir: str) -> None:
    """Trainer indices restart at 0 here, so writing over an existing sweep loses it."""
    existing = sorted(p.name for p in Path(save_dir).glob("trainer_*"))
    if existing:
        raise FileExistsError(
            f"{save_dir} already holds {len(existing)} trainer dirs ({existing[0]}..{existing[-1]}). "
            f"A sweep restarts trainer_N at 0, so this run would overwrite them and leave their "
            f"eval rows pointing at weights from a different grid. Move or remove the directory."
        )


def main():
    parser = core.arg_parser(__doc__)
    parser.add_argument(
        "--spade_sweep", action="store_true",
        help="Replace all trainer configs with a SpaDE hyperparameter sweep. "
             "Uses --spade_lambdas and --spade_gammas for the grid."
    )
    parser.add_argument(
        "--spade_lambdas", type=str, default="1e-3,1e-4",
        help="Comma-separated lambda_init values for SpaDE sweep"
    )
    parser.add_argument(
        "--spade_gammas", type=str, default="1e-4,1e-3,1e-2,1e-1",
        help="Comma-separated gamma_reg values for SpaDE sweep"
    )
    parser.add_argument(
        "--cosine_spade_sweep", action="store_true",
        help="Replace all trainer configs with a CosineSpaDE gamma sweep. "
             "Uses --cosine_spade_gammas for the grid."
    )
    parser.add_argument(
        "--cosine_spade_gammas", type=str, default="1e-1,5e-1,1.0,2.0,5.0",
        help="Comma-separated gamma_reg values for CosineSpaDE sweep"
    )
    parser.add_argument(
        "--batch_topk_sweep", action="store_true",
        help="Replace all trainer configs with a BatchTopK k sweep. Uses --batch_topk_ks."
    )
    parser.add_argument(
        "--batch_topk_ks", type=str, default="10,20,40,60,80,110,140,200,280,400",
        help="Comma-separated k values for the BatchTopK sweep. k sets L0, so this "
             "default spans Karvonen's whole axis (his TopK sweep runs k=10..400)."
    )
    parser.add_argument(
        "--baseline_sweep", type=str, default="", choices=["", *BASELINES_SWEPT],
        help="Replace all trainer configs with a sweep of one Karvonen baseline type. "
             "Uses --baseline_values for the grid."
    )
    parser.add_argument(
        "--baseline_values", type=str, default="",
        help="Comma-separated values of that type's sparsity knob (l1_penalty, "
             "initial_sparsity_penalty or k). No default: the response is unmeasured, "
             "so the range is a decision the caller has to make and record."
    )
    parser.add_argument(
        "--spade_pairs", type=str, default="",
        help="Comma-separated lambda:gamma SpaDE configs trained IN ADDITION to "
             "the selected configs (e.g. '1e-7:5e-3,8e-8:1e-3')"
    )
    args = parser.parse_args()

    selected = [name for name in ("spade_sweep", "cosine_spade_sweep", "batch_topk_sweep",
                                  "baseline_sweep") if getattr(args, name)]
    if len(selected) > 1:
        raise ValueError(f"pick one sweep, not {sorted('--' + s for s in selected)}.")
    if args.baseline_sweep and not args.baseline_values:
        raise ValueError("--baseline_sweep needs --baseline_values; there is no sane default.")

    refuse_if_populated(args.save_dir)

    core.run_training(
        lambda p: configs(p, args),
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
        smoke_test=args.smoke_test,
        no_wandb=args.no_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
