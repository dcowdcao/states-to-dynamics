# SpaDE vs baselines on OthelloGPT (headline results)

**Experiment:** OthelloGPT layer-5 residual stream (512-dim), dict_size 8192,
59,000 activations (1,000 games × 59 turns).
**Evaluation:** Karvonen's board-state reconstruction protocol.
**Dates:** March 2026 for the distance-SpaDE analysis; updated July 2026 with the
CosineSpaDE result and the γ sweep.

> **Scope, and one retired claim.** Sections 3–8 were written when the ~0.43
> variance-explained ceiling was believed to be a consequence of the sparsemax
> simplex — a fundamental limitation of the architecture. **That reading is
> retired.** CosineSpaDE keeps the identical simplex and no-bias decoder, changes
> only the encoder score, and reaches variance explained 0.979. Read §3–§8 as an
> analysis of **distance-SpaDE specifically**, never of the prototype-sparsemax
> family. See `docs/claims_status.md`.

---

## 1. Executive summary

Distance-SpaDE underperforms all five baseline SAEs on OthelloGPT across every
board-state reconstruction metric. Its variance explained is capped around 0.43
regardless of sparsity level, while the baselines reach 0.97–1.00, and that low
reconstruction fidelity cascades into poor coverage (0.06–0.31 against 0.52–0.73)
and low board-state reconstruction F1 (0.26–0.45 against 0.50–0.68).

**TopK (k=80) is the best overall SAE**, with the highest coverage (0.7265),
highest reconstruction F1 (0.6841), and near-perfect loss recovery (0.9996) at a
moderate L0 of 80.

Despite extensive tuning (λ across [5e-8, 8e-8, 9e-8, 1e-7] and γ across [0.005,
0.008, 0.015, 0.025]), no *distance*-SpaDE configuration reaches the Pareto
frontier the baselines define. The gap is not a tuning issue — but, as the
CosineSpaDE result below shows, it is also **not** inherent to the
sparsemax/simplex encoding.

**Update — CosineSpaDE overturns the "fundamental limitation" reading.**
CosineSpaDE keeps SpaDE's sparsemax simplex encoding and its no-decoder-bias
decoder unchanged, and scores atoms by **cosine similarity** instead of negative
squared Euclidean distance. It reaches variance explained **0.979**, coverage
**0.535** and reconstruction F1 **0.530** at L0 **173.6** — competitive with the
baselines. Because it retains the exact simplex constraint yet fully recovers the
variance, the ~0.43 ceiling analyzed in §3–§8 is **specific to squared-distance
scoring interacting with the heavy-tailed norms of standardized activations** (the
"distance-norm pathology"), not a consequence of sparsemax or the convex-
combination decoder.

Two qualifiers on the headline numbers, both from `docs/claims_status.md`:
**VE ≈ 0.43 is single-seed** — the multiseed aggregate for the distance-SpaDE
family is 0.849 ± 0.278, so the cosine-vs-distance contrast is quotable per seed
or qualitatively, never as a clean mean. And **recovering the variance is not the
same as preserving concept structure**: CosineSpaDE at VE 0.98 destroys the
rule-category manifold curve that the dense revived SpaDE preserves.

---

## 2. Baseline comparison

The evaluated SAEs at seed 42, ordered by reconstruction F1:

| SAE type | L0 | Frac recovered | Coverage | Recon F1 | Precision | Recall | Var explained |
|---|---|---|---|---|---|---|---|
| **TopK (k=80)** | **79.9** | **0.9996** | **0.7265** | **0.6841** | **0.7174** | **0.7047** | **0.9867** |
| BatchTopK (k=80) | 80.2 | 0.9999 | 0.6889 | 0.6300 | 0.6563 | 0.6660 | 0.9871 |
| Standard (ReLU) | 1094.9 | 0.9999 | 0.5872 | 0.5679 | 0.4243 | 0.9514 | 0.9978 |
| Gated | 49.7 | 0.9989 | 0.5160 | 0.5673 | 0.6790 | 0.5395 | 0.9824 |
| p-Anneal | 339.5 | 1.0000 | 0.6641 | 0.5410 | 0.5608 | 0.6794 | 0.9986 |
| CosineSpaDE (γ=1.0) | 173.6 | 0.9920 | 0.5346 | 0.5299 | 0.4893 | 0.7271 | 0.9789 |
| SpaDE default (λ=1e-3, γ=1e-4) | 1808.2 | 0.9999 | 0.5697 | 0.5145 | 0.3661 | 0.9559 | 0.9952 |
| Gated+p-Anneal | 32.3 | 0.9981 | 0.5479 | 0.5004 | 0.7470 | 0.4268 | 0.9708 |

_Distance-SpaDE's best swept config is λ=1e-7, γ=0.005 at L0 316 (§5–6); it is
analyzed there rather than listed here. CosineSpaDE is the strongest
SpaDE-family result on the board-state metrics and is baseline-competitive. The
SpaDE default row is the reconstruction-quality point of the family — near-perfect
variance explained at an L0 denser than Standard ReLU._

_The row this table used to carry, "SpaDE (initial)" at L0 15.4 / VE 0.352, is
gone. That is the γ=1e-2 seed-42 checkpoint: its evaluation is degenerate where
seeds 43–46 of the same config give L0 128 and VE 0.973, its `config.json` is a
Mac smoke config over real GPU weights, and it is now excluded from every study
and from the viewer._

_γ-sweep addendum (19 γ ∈ [0.05, 1e4] × 5 seeds): γ sweeps CosineSpaDE's L0 across
179 → 20.5, but only above γ ≈ 16, where the simplex-bounded sparsity regularizer
finally reaches parity with the MSE term; L0 is a sigmoid in log γ inflecting
exactly there. **CosineSpaDE remains strictly dominated at every matched
sparsity** — at L0 ≈ 80–98 TopK reaches coverage 0.736 against 0.474 — so γ moves
it along the L0 axis but never toward the frontier. Past γ=200 fidelity collapses
(0.102 loss recovered at γ=1e4) while variance explained holds at 0.854, so VE
alone is not a health check for this family. The operating point stays **γ = 1.0**:
the sweep's own recommendation of γ=20 was selected on reconstruction F₁ (0.547 vs
0.530) and was declined, because coverage decides and runs the other way (0.537 vs
0.474). Aggregate: `results/multiseed/aggregate_cosine_sweep.csv`; figures:
`figures/cosine_spade_sweep/Figure_4_karvonen_extension.png`,
`Figure_cosine_gamma_response.png`._

### Key observations

- **TopK dominates** across all metrics simultaneously. It is the only SAE that
  achieves both high coverage and high precision, giving the best F1 by a
  substantial margin. BatchTopK sits just behind it at the same L0.
- **Standard (ReLU)** gets high recall (0.951) and poor precision (0.424): at
  L0 ≈ 1095 many features fire, catching most board states along with many false
  positives.
- **Gated and Gated+p-Anneal** give the sparsest representations (L0 50, 32) with
  high precision (0.679, 0.747) at the cost of recall. These are the most
  selective SAEs.
- **p-Anneal** has the second-highest coverage (0.664) but mediocre F1 — its
  features cover many board positions with moderate accuracy each.
- **Distance-SpaDE** is an outlier on every axis wherever it is placed: lowest
  coverage, lowest F1, and critically the lowest variance explained. Precision is
  competitive; recall is what collapses at the over-regularized end.

---

## 3. The variance-explained ceiling in distance-SpaDE

### The problem

Distance-SpaDE's variance explained is capped at approximately 0.43–0.44
regardless of sparsity level. All five baselines achieve 0.97–1.00.

| SpaDE config | L0 | Var explained |
|---|---|---|
| λ=1e-7, γ=5e-3 | 316.2 | 0.4325 |
| λ=8e-8, γ=1e-3 | 898.5 | 0.4442 |
| λ=1e-7, γ=8e-3 | 176.2 | 0.4276 |
| λ=1e-7, γ=1e-2 | 140.9 | 0.4200 |
| λ=1e-7, γ=3e-2 | 100.8 | 0.4119 |

Even pushing L0 to 898, comparable to Standard ReLU's 1095, moves variance
explained by less than 0.03. A 9× increase in L0 for that is what makes the
ceiling look architectural rather than a sparsity setting.

Two things bound how far this table can be pushed. Every row is **seed 42**, and
the multiseed aggregate for the family is 0.849 ± 0.278 — the ceiling is a
property of these configurations at this seed, not a stable population mean. And
every row has λ ≤ 1e-7, a region that sits below fp32 sparsemax resolution, where
configurations collapse or train bitwise-identically to each other.

### The original diagnosis: the simplex constraint

Distance-SpaDE's encoding is

```
f = sparsemax(-lambda * ||x - Ae_i||^2)
```

and sparsemax maps its input onto the **probability simplex**: all coefficients
non-negative, summing to 1. The reconstruction

```
x_hat = f @ Ad.T = sum(f_i * d_i)    where sum(f_i) = 1
```

is then a **convex combination** of decoder columns, geometrically constrained to
the convex hull of the decoder atoms, while standard SAEs compute

```
x_hat = sum(c_i * d_i) + b    where c_i >= 0, unconstrained magnitude
```

whose coefficients can be arbitrarily large, reaching any point in the positive
cone spanned by the decoder columns — a strictly larger representable set.

### Why that diagnosis is wrong — CosineSpaDE recovers the variance

The convex-hull argument is incomplete: the decoder atoms are **not**
norm-constrained, so a convex combination of them can reach well outside any
fixed-radius ball. The reachable set is the convex hull of arbitrarily large
atoms, which can cover the data.

The decisive test is **CosineSpaDE**, which keeps the sparsemax simplex and the
no-bias decoder *unchanged* and only swaps the encoder score from
`-λ·‖x − Ae_i‖²` to `cos(x, Ae_i)`. It reaches **variance explained 0.979** at
L0 173.6.

So the ~0.43 ceiling is not the simplex. It is the **squared-distance score under
standardized activations**: layer-5 norms are heavy-tailed — a few tokens, notably
position 0, have norms ~10× the median — so `‖x − Ae‖²` is dominated by norm
differences, and distance-SpaDE ends up assigning atoms by magnitude rather than
by the direction that carries board state. Cosine scoring is norm-invariant and
sidesteps this. §4–§8 below diagnose *distance*-SpaDE specifically.

---

## 4. Cascade effect — how low variance explained poisons every metric

The low variance explained cascades through Karvonen's evaluation:

```
Low var_explained (~0.43)
  → reconstructed activations are poor approximations of the true ones
  → SAE feature activations carry noise / miss signal components
  → per-feature F1 for (feature, square, class) triples is low
  → few features clear the threshold needed to "cover" a board position
  → coverage is low (0.06-0.31 vs 0.52-0.73 for baselines)
  → board reconstruction F1 is low (0.26-0.45 vs 0.50-0.68)
```

### The frac_recovered paradox

Distance-SpaDE's frac_recovered (behavioral fidelity) can be high — 0.92–0.96 for
well-tuned configs. How can the SAE preserve model behavior while explaining only
43% of activation variance?

Because **frac_recovered measures next-token prediction loss**, not geometric
reconstruction quality. OthelloGPT's next-move prediction may depend on a
relatively low-dimensional subspace of the 512-dim residual stream. SpaDE may
capture the directions most relevant to the output logits while missing directions
that encode board state but matter less to the next-move logits. Its
reconstructions are "behaviorally adequate" but "representationally incomplete."

The distinction matters for interpretability: board-state reconstruction evaluates
whether SAE features capture the model's *world model*, while frac_recovered only
evaluates whether they capture its *output behavior*. Distance-SpaDE partially
preserves behavior and largely fails to recover the internal world model.

---

## 5. Lambda sweep analysis (distance-SpaDE)

Lambda controls the sharpness of the distance-based activation: smaller λ gives a
flatter sparsemax input, so more features activate and L0 rises.

| Config | L0 | Frac recovered | Coverage | Recon F1 | Var explained |
|---|---|---|---|---|---|
| λ=5e-8, γ=1e-3 | **8192** | **−0.222** | 0.478 | 0.005 | −0.00003 |
| λ=8e-8, γ=1e-3 | 898.5 | 0.965 | 0.422 | 0.480 | 0.444 |
| λ=9e-8, γ=1e-3 | 898.5 | 0.965 | 0.422 | 0.480 | 0.444 |
| λ=1e-7, γ=1e-2 | 164.3 | 0.950 | 0.224 | 0.449 | 0.424 |

1. **Sharp phase transition between λ=5e-8 and λ=8e-8.** At λ=5e-8 SpaDE collapses
   completely: all 8192 features activate, the reconstruction is essentially the
   centroid of all decoder atoms, and frac_recovered goes *negative* (−0.222),
   meaning the reconstruction increases the model's loss beyond zero ablation. A
   fully degenerate mode.
2. **λ=8e-8 and λ=9e-8 produce identical results** — same L0, same F1, same
   variance explained. They are not merely in the same regime: under the multiseed
   harness these configurations train **bitwise-identically**, because λ at this
   magnitude is below fp32 resolution in the sparsemax scores. The same holds
   between 8e-8 and 1e-7 at γ=1e-3.
3. **λ=1e-7 is the best of the values tested here**, giving tunable sparsity
   (L0 ~100–316 with γ) and the highest frac_recovered among them.
4. **The narrow window between 5e-8 and 8e-8 is not worth probing.** The original
   write-up proposed a fine-grained sweep there to see whether the collapse is
   sharp or gradual. It is a floating-point artifact, not a phase boundary — the
   λ values that matter are 1e-3 and 1e-4, which is where the default now sits.

---

## 6. Gamma sweep analysis — the precision/recall trade

Gamma controls sparsity pressure in the loss:
`loss = MSE_recon + gamma * sum(||x - Ad_j||^2 * f_j)`. Higher gamma penalizes
activating features far from the input, encouraging sparser representations.

At fixed λ=1e-7:

| γ | L0 | Precision | Recall | F1 | Coverage | Frac recovered |
|---|---|---|---|---|---|---|
| 0.005 | 316.2 | 0.343 | 0.747 | 0.421 | 0.314 | 0.959 |
| 0.008 | 176.2 | 0.399 | 0.651 | 0.447 | 0.247 | 0.952 |
| **0.01** | **140.9** | **0.438** | **0.591** | **0.453** | **0.203** | **0.943** |
| 0.025 | 100.8 | 0.480 | 0.503 | 0.438 | 0.168 | 0.921 |

The trade-off is clear and monotonic. **Lower γ** (less sparsity pressure)
activates more features (L0 316) for higher recall (0.747) — more board positions
get voted on — but precision drops to 0.343, because many activated features are
noisy or weakly associated with board states. **Higher γ** activates fewer
features (L0 101) for higher precision (0.480) — the survivors are more confident —
but recall falls to 0.503 as positions lose coverage. This mirrors Standard
ReLU's behavior at L0 1095: recall 0.951, precision 0.424, the same
spray-and-pray pattern.

**F1 peaks at γ ≈ 0.01** with F1 = 0.453. That is still 0.231 below TopK (0.684),
0.115 below Standard ReLU (0.568) and 0.047 below Gated+p-Anneal (0.500). Even at
optimal gamma, distance-SpaDE is the weakest architecture here for board-state
reconstruction.

**Coverage declines monotonically**, 0.314 → 0.168 as gamma increases: the set of
board positions any SpaDE feature reconstructs shrinks with sparsity. At γ=0.025
only 16.8% of positions are covered, under a quarter of TopK's 72.6%.

---

## 7. Pareto frontier analysis

An SAE is on the Pareto frontier if no other achieves better quality at equal or
lower L0.

### L0 vs reconstruction F1

```
L0      SAE                 F1
32.3    Gated+p-Anneal      0.500   <-- Pareto optimal
49.7    Gated               0.567   <-- Pareto optimal
79.9    TopK                0.684   <-- Pareto optimal
100.8   SpaDE (γ=0.025)     0.438   (dominated by Gated at L0=50)
140.9   SpaDE (γ=0.01)      0.453   (dominated by Gated at L0=50)
176.2   SpaDE (γ=0.008)     0.447   (dominated by Gated at L0=50)
316.2   SpaDE (γ=0.005)     0.421   (dominated by Gated at L0=50)
339.5   p-Anneal            0.541   <-- Pareto optimal
898.5   SpaDE (λ=8e-8)      0.480   (dominated by p-Anneal at L0=340)
1094.9  Standard (ReLU)     0.568   <-- Pareto optimal
```

**No distance-SpaDE configuration lies on the Pareto frontier.** Every point is
dominated by a baseline achieving higher F1 at equal or lower L0.

### Head-to-head at comparable L0

| Comparison | L0 | Coverage | F1 | Var explained |
|---|---|---|---|---|
| **Gated+p-Anneal** | 32 | 0.548 | 0.500 | 0.971 |
| SpaDE (γ=0.025) | 101 | 0.168 | 0.438 | 0.412 |
| *Gap* | *3× more features* | *−69%* | *−12%* | *−58%* |

| Comparison | L0 | Coverage | F1 | Var explained |
|---|---|---|---|---|
| **TopK (k=80)** | 80 | 0.726 | 0.684 | 0.987 |
| SpaDE (γ=0.008) | 176 | 0.247 | 0.447 | 0.428 |
| *Gap* | *2.2× more features* | *−66%* | *−35%* | *−57%* |

| Comparison | L0 | Coverage | F1 | Var explained |
|---|---|---|---|---|
| **Gated** | 50 | 0.516 | 0.567 | 0.982 |
| SpaDE (γ=0.005) | 316 | 0.314 | 0.421 | 0.433 |
| *Gap* | *6.3× more features* | *−39%* | *−26%* | *−56%* |

Distance-SpaDE consistently needs 2–6× more active features while scoring worse on
every metric. The variance-explained gap (~57%) is remarkably stable across all
three comparisons, which is what identifies it as the binding constraint.

---

## 8. What these results reveal about OthelloGPT

### Representations use directions and magnitudes, not point prototypes

Distance-SpaDE's inductive bias assumes activations decompose by proximity to
prototype points — "this activation is closest to atom 42, next-closest to atom
107." Its failure suggests OthelloGPT layer 5 does not organize its
representations as clusters around such points.

The success of standard SAEs, TopK especially, is consistent with the residual
stream using **superposition of linear directions with varying magnitudes**, where
the critical information is carried not only by *which* directions are active but
by *how strongly*.

The caveat is load-bearing: CosineSpaDE is a **direction**-prototype method that
keeps the simplex, and it succeeds. So the failure is specific to point /
squared-distance prototypes, not to prototype-style or simplex encoding in
general, and the original phrasing of this section — that the simplex constraint
discards the magnitude information — is exactly the reading §3 retires.

### Board-state information is linearly recoverable

TopK's coverage of 0.726 at L0 80 shows that most of the 8×8 board state is
encoded sparsely and linearly-recoverably within 80 features of the 512-dim
residual stream. Strong evidence for the linear representation hypothesis applied
to OthelloGPT's world model.

The 27% gap from perfect coverage likely reflects board positions encoded in
deeper or earlier layers, information about game phase or move legality that
static board-state labels do not capture, and the limits of a 16× expansion.

### The behavioral vs representational fidelity gap

Distance-SpaDE's high frac_recovered (~0.95) at low variance explained (~0.43)
says roughly half the activation variance in layer 5 is not critical for next-move
prediction. That "dark variance" may be redundant or compressed copies of
information available in other layers, board-state detail used by downstream
layers rather than the layer-5 logit computation, or training artifacts. If it
holds up, it implies significant representational redundancy in OthelloGPT's
intermediate layers, which matters for pruning and distillation.

---

## 9. Recommendations for future work

### 9.1 Architectural modifications to SpaDE

> **Largely settled by CosineSpaDE.** The single most effective fix turned out to
> be changing the *score function* (squared distance → cosine similarity), which
> recovers variance explained to 0.979 while keeping the sparsemax simplex and the
> no-bias decoder. Ideas 1–3 below are therefore likely unnecessary for closing the
> variance gap, though they may still help at the extreme-low-L0 end.
> Recommendations 4–7 remain open.

1. **Remove the simplex constraint.** Replace sparsemax with ReLU on the
   negative-distance scores, or a "shifted sparsemax" that subtracts a bias before
   projecting, so coefficients need not sum to 1:
   ```
   f = ReLU(-lambda * ||x - Ae_i||^2 + bias_i)
   ```
2. **Add a learnable scale factor** — multiply each sparsemax coefficient by a
   learned per-feature magnitude `alpha_i > 0`, preserving distance-based feature
   selection while letting the reconstruction escape the convex hull.
3. **Add a decoder bias**, `x_hat = f @ Ad.T + b_dec`, shifting the whole hull.

### 9.2 Understanding the phase transition

4. ~~Fine-grained lambda sweep between 5e-8 and 8e-8.~~ **Withdrawn.** That window
   is below fp32 sparsemax resolution; runs in it collapse or come out
   bitwise-identical, so there is no transition there to characterize.

### 9.3 Broader evaluation

5. **Test on different layers.** The prototype-proximity bias might suit layers
   whose representations genuinely cluster — early layers with more local
   features, or the embedding layer.
6. **Test on language models.** OthelloGPT is small and domain-specific; SpaDE may
   fare differently where representations are organized differently.

### 9.4 Hybrid approaches

7. **Distance-gated, magnitude-decoded hybrid.** Use the distance computation to
   *select* active features (top-k by proximity), then compute unconstrained ReLU
   coefficients for the selected ones:
   ```
   selected = top_k(-||x - Ae_i||^2)
   f = ReLU(W_enc[selected] @ x + b_enc[selected])
   x_hat = f @ Ad[selected].T + b_dec
   ```

---

## Appendix A: all distance-SpaDE configurations tested

Seed 42 throughout.

| Config | λ | γ | L0 | Frac recovered | Coverage | Recon F1 | Var explained |
|---|---|---|---|---|---|---|---|
| Lambda sweep | 5e-8 | 1e-3 | 8192 | −0.222 | 0.478 | 0.005 | −0.00003 |
| Lambda sweep | 8e-8 | 1e-3 | 898.5 | 0.965 | 0.422 | 0.480 | 0.444 |
| Lambda sweep | 9e-8 | 1e-3 | 898.5 | 0.965 | 0.422 | 0.480 | 0.444 |
| Gamma sweep | 1e-7 | 5e-3 | 316.2 | 0.959 | 0.314 | 0.421 | 0.433 |
| Gamma sweep | 1e-7 | 8e-3 | 176.2 | 0.952 | 0.247 | 0.447 | 0.428 |
| Gamma sweep | 1e-7 | 1e-2 | 140.9 | 0.943 | 0.203 | 0.453 | 0.420 |
| Gamma sweep | 1e-7 | 2.5e-2 | 100.8 | 0.921 | 0.168 | 0.438 | 0.412 |
| Default | 1e-3 | 1e-4 | 1808.2 | 1.000 | 0.570 | 0.515 | 0.995 |

The former "Initial" row (λ=9.77e-5, γ=0.2, L0 15.4) is dropped. Those λ/γ values
were read off a `config.json` that is a Mac smoke-test config carrying the stale
pre-canonicalization class defaults; the weights are the γ=1e-2 production run,
and that checkpoint is excluded everywhere.

## Appendix B: figures and data

| What | Where |
|---|---|
| Baseline comparison, all 8 types | `figures/Figure_4_karvonen_extension.png` |
| Every sweep config combined | `figures/Figure_4_all_configs_combined.png` |
| CosineSpaDE γ sweep (Karvonen 2×2) | `figures/cosine_spade_sweep/Figure_4_karvonen_extension.png` |
| CosineSpaDE γ response, with the parity line | `figures/cosine_spade_sweep/Figure_cosine_gamma_response.png` |
| Seed-42 baselines | `results/main/all_results.csv` |
| BatchTopK, CosineSpaDE, SpaDE default | `results/{batch_topk,cosine_spade,spade_revival}/all_results.csv` |
| λ sweep, γ sweep at λ=1e-7 | `results/{lambda_sweep,gamma_sweep_lam1e-07}/all_results.csv` |
| Multiseed aggregates (the error bars) | `results/multiseed/aggregate*.csv` |
