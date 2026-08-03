# OthelloGPT SAE Experiment

Train and evaluate 8 Sparse Autoencoders on OthelloGPT residual-stream activations
through Karvonen's `trainSAE()` loop, so the comparison is fair by construction: 5
Karvonen baselines + BatchTopK, plus the two prototype-distance variants SpaDE and
CosineSpaDE. The full reference — equations, metric formulas, results — is
[pipeline.md](pipeline.md); this page is the operational one-pager.

## SAE types

| # | Trainer | Architecture | Key hyperparameters |
|---|---------|-------------|---------------------|
| 1 | PAnnealTrainer | AutoEncoder (16×) | Lp^p annealing, sp=0.042 |
| 2 | StandardTrainer | AutoEncoder (16×) | L1 penalty = 0.059 |
| 3 | GatedSAETrainer | GatedAutoEncoder (16×) | L1 penalty = 1.37 |
| 4 | TopKTrainer | AutoEncoderTopK (16×) | k = 80 |
| 5 | GatedAnnealTrainer | GatedAutoEncoder (16×) | Lp^p annealing, sp=0.955 |
| 6 | SpaDETrainer | SpaDEAutoEncoder (16×) | λ_init = 1e-3, γ = 1e-4 |
| 7 | BatchTopKTrainer | Batch-wise TopK (16×) | k = 80 |
| 8 | CosineSpaDETrainer | CosineSpaDEAutoEncoder (16×) | cosine scoring, γ = 1.0 (no λ) |

All use expansion factor 16× (dict_size 8192, activation_dim 512). Baseline
hyperparameters are the Pareto-selected best configs from Karvonen's own sweep
(see [karvonen_reference.md](karvonen_reference.md) §10).

Two hyperparameter facts that override anything you may read in an older document
or in a checkpoint's `config.json`:

- **The default SpaDE configuration is λ=1e-3, γ=1e-4**, with weights under
  `checkpoints/spade_revival/`. It replaces γ=1e-2, whose seed-42 checkpoint is
  degenerate and is excluded from every study and from the viewer.
- **CosineSpaDE's default is γ=1.0**, kept deliberately over the γ-sweep's
  recommendation of γ=20. The sweep selected on reconstruction F₁ (0.547 vs
  0.530); coverage runs the other way (0.537 vs 0.474) and is the deciding metric.
  Do not "fix" this back to 20.

## Layout

```
src/spade/
├── training/
│   ├── core.py                 shared loop: standardization, data-init, trainSAE plumbing
│   ├── sweep.py                λ/γ grid drivers
│   ├── train_{standard,panneal,gated,gated_anneal}.py
│   ├── train_{topk,batch_topk}.py
│   └── train_{spade,cosine_spade}.py
├── saes/
│   ├── spade_base.py           shared SpaDE-family base (autoencoder + trainer)
│   ├── spade.py                SpaDETrainer + SpaDEAutoEncoder
│   ├── spade_cosine.py         CosineSpaDETrainer + CosineSpaDEAutoEncoder
│   └── registry.py             the verified checkpoint roster
├── extraction/extract_activations.py   eval activations + board states
├── evaluation/
│   ├── run_eval.py             fidelity, coverage, F1
│   └── karvonen_adapter.py     SAE loading, StandardizedDictionary, board-state metrics
├── figures/
│   ├── figure_karvonen_extension.py    the Karvonen 2×2 comparison figure
│   └── figure_all_configs_combined.py  same, including every sweep config
└── validation/final_validation.py      fast self-tests

scripts/
├── run_pipeline.sh             local driver (train/extract/eval/figures)
├── smoke_test.sh               ~5–10 min validation
└── slurm/                      FASRC jobs, one dir per program

checkpoints/<experiment>/       trained SAE weights
data/activations/               extracted evaluation data
results/<experiment>/           evaluation CSVs
figures/                        generated plots
```

## Pipeline

Every stage is a module invocation; nothing is run as a script path.

**1. Train all 8 SAEs** — streams activations from OthelloGPT via Karvonen's
`NNsightActivationBuffer`, standardizes (center + scale), and trains all 8
simultaneously for 300M tokens (~36,621 steps).

```bash
for t in standard panneal gated gated_anneal topk batch_topk spade cosine_spade; do
  python -m spade.training.train_$t --save_dir checkpoints/main --device cuda:0
done
```

**2. Extract evaluation data** — layer-5 activations plus engine-replayed
ground-truth board states, required by the coverage and reconstruction-F1 metrics.

```bash
python -m spade.extraction.extract_activations --output_dir data/activations
```

**3. Evaluate** — per checkpoint: Loss Recovered (fidelity), Coverage,
Reconstruction F1, L0, Variance Explained.

```bash
bash scripts/run_pipeline.sh eval
```

**4. Figures**

```bash
bash scripts/run_pipeline.sh figures
```

**Full pipeline / single stages**

```bash
bash scripts/run_pipeline.sh all      # train + extract + eval + figures
bash scripts/run_pipeline.sh train
bash scripts/smoke_test.sh            # 100 steps, ~5-10 min
```

**On FASRC** — see [fasrc_commands.md](fasrc_commands.md) for the end-to-end
cluster workflow.

```bash
bash scripts/slurm/main/run_all.sh    # train + extract in parallel -> eval, chained
```

## Key design decisions

- **Activation standardization.** `act' = scale * (act − mean)` with
  `scale = sqrt(d / E[‖act − mean‖²])`. Applied by `StandardizedBuffer` in
  training and `StandardizedDictionary` at evaluation. It is part of the
  contract, not preprocessing: anything that encodes without applying it is
  silently wrong rather than approximately right.
- **Loss Recovered (fidelity).** `frac_recovered = (loss_recon − loss_zero) /
  (loss_orig − loss_zero)`, computed by running the LM three times. This is the
  y-axis of the main comparison figure, and it is a different quantity from
  variance explained — the two dissociate sharply for the SpaDE family.
- **Lambda (inverse temperature).** SpaDE only, trainable via a softplus
  reparameterization. λ ≤ 1e-7 collapses: below fp32 sparsemax resolution those
  runs train bitwise-identically, so there is no point sweeping there.
- **Single training loop.** All 8 SAEs train simultaneously through `trainSAE()`,
  sharing one activation stream, so no comparison is confounded by data order.
- **W&B logging.** Per-step via Karvonen's built-in logging; project
  `spade-vs-karvonen-othello`.
