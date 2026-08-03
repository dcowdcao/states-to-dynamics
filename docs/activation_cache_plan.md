# Activation cache: plan and cost/benefit

Status: proposal, awaiting sign-off. Touches core training logic, so nothing here
is implemented.

## The duplicated work

`spade.training.core` builds an `NNsightActivationBuffer` that runs OthelloGPT to
produce layer-5 `resid_post` activations, wraps it in `StandardizedBuffer`, and
consumes 300M tokens (36,621 steps × 8192). Every run re-derives those activations
from scratch. Across the CosineSpaDE work alone that is 5 seeds × 2 sweep segments
= 10 identical passes over the same LM.

Caching the activations once and streaming them from disk would remove the LM
from training entirely.

## What it would actually buy — measure before building

Two observed throughputs pin down the split. A 1-trainer run holds 9.07 it/s; a
10-trainer run holds 1.20 it/s. Solving `step = F + n·T`:

| | |
|---|---|
| fixed cost (LM + buffer + loop overhead) | **29.9 ms/step** |
| per-trainer cost | **80.3 ms/step** |

The LM cost is *per step*, not per trainer, so the saving is a constant **~0.30 h
(18 min) per run** no matter how many configs it trains:

| trainers | step | LM share | run time | cache saves |
|---|---|---|---|---|
| 1 | 110 ms | 27.1% | 1.12 h | 0.30 h |
| 5 | 432 ms | 6.9% | 4.39 h | 0.30 h |
| 10 | 833 ms | 3.6% | 8.48 h | 0.30 h |
| 19 | 1556 ms | 1.9% | 15.83 h | 0.30 h |

So on the 8.5 h sweeps this is a **3.6% win for ~300 GB of storage and a new data
path**. It only looks attractive for single-SAE runs (27%).

Caveat on the numbers: the 9.07 it/s point comes from a 100-step smoke run, so it
carries warmup effects. The split is the right order of magnitude but should be
confirmed with a real profile before anything is built on it.

## Where the time actually goes

Per trainer-step the three big matmuls (encode `x @ Aeᵀ`, decode `f @ Adᵀ`, and
the regularizer's `x_n @ ad_n`, forward+backward) total ~618 GFLOP, which is ~4 ms
at a realistic 150 TFLOP/s. Measured per-trainer cost is 80 ms. **~95% of the
trainer step is not matmul.**

The prime suspect is sparsemax: `encode` calls `Sparsemax(dim=-1)` on a
`[8192, 8192]` score matrix in float32, which sorts **67M floats per trainer-step**.
Sorting is bandwidth-bound and scales far worse than the GEMMs around it.

If that is confirmed, the ranked opportunities are:

1. **Sparsemax without a full sort** — the projection only needs the support
   threshold τ, obtainable by bisection on the score range or by `topk` on a
   bounded support (measured L0 never exceeds ~180 of 8192 for CosineSpaDE).
   Potential: most of the 80 ms.
2. **Activation cache** — a flat 18 min/run (this document).
3. `llm_batch_size` above 64, bf16 autocast on the frozen LM — shrinks the 30 ms,
   i.e. a fraction of item 2.

Item 1 is worth ~20× item 2. I would profile before spending storage on this.

## Cache design, if we proceed anyway

**Sizing.** 300M activations × 512 dims:

| format | size | note |
|---|---|---|
| fp32 | 614 GB | matches current training dtype exactly |
| fp16 | **307 GB** | ~1e-3 relative error on standardized activations |
| int8 + per-row fp16 scale | 154 GB | changes training inputs; unsafe for a study whose subject is reconstruction fidelity |

Netscratch has 435 T free (lab dir currently 68 G), so fp16 fits — but confirm the
*lab quota*, not just filesystem free space, before writing 307 GB. The cache
belongs under `study_dir()`, on the same 90-day-retention tier as every other
heavy intermediate, which is a second argument against paying for it.

**Layout.** Shard into ~1000 files of ~300 MB (`shard_%04d.pt`, `[299008, 512]`
fp16) plus a `manifest.json` recording model name, layer, hook point, dtype,
context length, dataset revision, and total rows. Read with `torch.load(mmap=True)`
so page cache does the work and no shard is fully resident.

**Ordering and seeds.** Batch order currently depends on the global seed, and both
`sample_init_activations` (SpaDE prototype init) and the standardization
calibration draw from the head of the stream. A cache reader must therefore take a
seed, shuffle shard order and within-shard offsets from it, and expose the same
`__next__` contract as `StandardizedBuffer`, so the only swap in
`spade.training.core` is which object gets constructed.

**Storage of standardization.** Calibration over 100 batches currently yields a
slightly seed-dependent scale (0.505683 for seed 42 vs 0.505401 for seed 46). With
a shared cache these become identical unless calibration re-samples per seed. That
is arguably an improvement, but it is a behaviour change and must be stated in the
results, not discovered later.

## Validation before it replaces anything

1. Byte-level: re-derive 10k activations live and assert max relative error vs the
   cache is below the fp16 bound (~1e-3).
2. End-to-end: train one CosineSpaDE config (γ=1.0, seed 42) from the cache and
   from the live buffer, and require the Karvonen metrics to agree within the
   measured seed std (L0 ±2.9, coverage ±0.004).
3. Only after both pass does any sweep script switch over.

## Recommendation

Profile a single training step first and confirm the sparsemax hypothesis. If it
holds, do item 1 and skip the cache — 18 min per run does not justify 307 GB and a
second data path. Revisit the cache if the repo moves toward many single-SAE runs,
where it is worth 27%.
