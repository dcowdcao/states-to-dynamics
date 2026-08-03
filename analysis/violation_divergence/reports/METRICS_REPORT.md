# A2 metrics: divergence kernels and the per-space driver

> **Correction, read first.** This report contradicts itself about `half_life`.
> Judgement 1's "Superseded at integration" note is the current behaviour —
> half-life is **elapsed since the peak**, with `peak_r` stored beside it — while
> request 3 in "Requests for the integrator" still asserts the old absolute-offset
> definition. The red team flagged exactly that stale paragraph (`../RED_TEAM.md`,
> F6). The prose is left as written rather than silently patched: this is a record
> of a review, not a live spec. For what the study currently claims, read
> `docs/claims_status.md` and `../RED_TEAM.md` — several metrics discussed below,
> `peak` and `peak_r` above all, were later shown to be max-of-noise.

`divergence.py` (pure kernels), `test_divergence.py` (hand-worked selftest),
`compute_divergence.py` (per-space driver). The two judgement calls the design
doc left open are settled below; the rest is the npz layout, what the driver was
run against, and three requests for the integrator.

## Judgement 1 — half-life when δ̂ never halves

`persistence` returns `half_life = +inf` for those rows. `torch.isinf` is the
censoring flag; no extra key.

This is right-censoring, and the censored rows are not a nuisance — they are the
result. NaN would be the tempting alternative and it is the wrong one: every
`nanmean`/`nanmedian` downstream silently drops exactly the branches the model
never recovered from, so every summary would be biased toward "the model
recovers", which is the opposite of the study's hypothesis. `+inf` keeps the
column ordered, so a median, and any quantile below the censoring rate, is still
a correct estimate; the censored fraction reported on its own ("in 52.5% of
branches δ̂ never falls to half its peak within 24 moves") is the cleanest
statement of the finding; and a *mean* over the column returns `inf`, refusing
loudly rather than returning a wrong number. Summaries should report the median
plus the censored fraction, never the mean.

**Superseded at integration (commit e93f9bb).** This section originally flagged
that half-life was the **absolute offset r** rather than the elapsed
`r − peak_r`, which made rows peaking late incomparable to rows peaking at
`r = 0`. It is now elapsed, with `peak_r` stored beside it so the absolute
offset is a sum away. The red team verified the change: over all finite rows
`max(peak_r + half_life)` is exactly 24 and never above.

## Judgement 2 — `topk_share` and `participation_ratio` read Δ²; "share" is of ‖Δ‖₂²

Both operate on the **magnitude**, via `Δ_j²` (`topk(Δ²)` is `topk(|Δ|)`), and
the share's denominator is **‖Δ‖₂², not ‖Δ‖₂ and not ‖Δ‖₁**.

Magnitude, not signed Δ: ranking by signed value would select only the features
the violation switched *on*, and would score a one-sided divergence as sparse
when it is merely one-sided. A feature the violation switches off is as much
part of the divergence as one it switches on. Both functions are even in Δ, so
this costs nothing.

The denominator: with `p_j = Δ_j² / ‖Δ‖₂²`, top-k share is `Σ_{top k} p_j` and
participation ratio is `1 / Σ_j p_j²`. The two numbers are then the k-th partial
sum and the inverse-Simpson effective count of **one** distribution, so they can
be read against each other — a top-10 share of 0.9 is consistent with PR ≈ 3 and
inconsistent with PR ≈ 300, and that consistency check is the point of reporting
both. The alternatives break it. A share of ‖Δ‖₂ itself (the square root) is not
additive over disjoint coordinate groups, so it is not a share. An L1 share is
additive but describes a different distribution from the PR, and reading the two
side by side would be a category error. Cost of the choice: an L2-energy share is
numerically larger than the L1 share of the same Δ, so the definition has to be
stated wherever the number is — it is in the module docstring, here, and it
belongs in F4's caption.

## npz layout — `$DIR/divergence/<space>.npz`

float32 unless noted. Event-locked, `[n_branch, 33]` at offsets `r = s − t0`:

| key | |
|---|---|
| `window` | int `[33]`, the offsets themselves, −8..24 |
| `delta` | δ̂ = ‖Δ‖ / ‖z_ref‖ (M1) |
| `cos_t0` | cos(Δ(r), Δ(0)) (M2) |
| `topk_share` | share of Δ's squared norm in its 10 largest \|Δ_j\| (M4) |
| `pr` | participation ratio of Δ² (M4) |
| `jaccard` | support Jaccard — **SAE spaces only** (M4) |
| `dl0` | L0(branch) − L0(reference) — **SAE spaces only** (M4) |

Per branch row, `[n_branch]`:

| key | |
|---|---|
| `peak`, `peak_r` (int), `half_life`, `plateau` | `persistence(delta)` (M3) |
| `topk_idx` | int `[n_branch, 10]`, the 10 largest \|Δ_j\| at r = 0 — SAE only |
| `game_id`, `t0`, `category`, `is_control`, `n_flipped` | per-row metadata |
| `row` | index into `branches.pt` |

Rows are the 13531 non-reference branches; the 2000 reference rows are dropped,
their divergence against themselves being zero. `row` is the join key for
anything else per-row (`branch_token`, `n_legal`, `tokens`) — carrying those too
would duplicate `branches.pt`.

Three keys go beyond the doc's interface block and are worth naming:
`persistence`'s four outputs (so A3 and A4 cannot disagree on the definition of
`delta_peak_r_L5` merely by re-deriving it differently — A5 re-derives on
purpose), `topk_idx` (M4's cross-event feature recurrence is not computable
without it, and re-encoding eight dictionaries to recover ten integers per row is
not a good trade), and `row`.

`jaccard` and `dl0` are SAE-only because a dense space has every coordinate
nonzero: raw-space Jaccard is identically 1 and raw ΔL0 identically 0. F4 should
draw raw as the dense reference for top-10 share and PR, where it means
something, and leave it off the Jaccard panel — a constant line at 1.0 reads as a
target nobody hit (FIGURE_GUIDELINES §4.1).

M5 (behavioural KL) reads `$DIR/acts/logits.pt` directly and is not a space here.

## Selftest

`python -m spade.violation_divergence.test_divergence` — no data, no model, no GPU:

```
ok test_align_maps_offset_r_to_position_t0_plus_r
ok test_align_never_wraps_off_the_board
ok test_constructed_delta_recovers_by_hand
ok test_delta_on_exactly_k_coordinates
ok test_direction_cosine_persistent_rotated_and_undefined
ok test_identical_branches_do_not_diverge
ok test_persistence_peak_half_life_and_plateau
ok test_support_jaccard_matches_the_hand_count
ok test_window_stays_on_the_board_for_every_t0
ok test_zero_reference_norm_is_undefined
10 tests passed
```

`test_align_never_wraps_off_the_board` is the one that earns its keep: a raw
`values[i, t0 + r]` with `t0 + r < 0` wraps to the end of the game and returns a
number, silently. `align` masks instead.

## What the driver was run against

**A synthetic fixture with a closed-form answer** (built and verified in a
scratch directory, not committed). Each branch row copies its game's reference
activations exactly below `t0` and adds, at `s ≥ t0`, one fixed random direction
of norm `SCALE · exp(−(s−t0)/τ) · ‖z_ref_s‖`, so the true divergence is
`δ̂(r) = SCALE · exp(−r/τ)` exactly — SCALE 0.25 illegal / 0.15 control, τ from 3
to 58 so both decayed and censored half-lives occur. 40 games, 286 rows, 246
branches, activations borrowed from the straightness study's real `resid_post_5`
so the SAE codes are not degenerate. All 17 spaces ran. Checks:

- `delta` matches the closed form to < 3e-3 absolute in all 9 raw spaces;
  the mean at r = 0 is 0.2337 = (206·0.25 + 40·0.15)/246 to four places.
- `cos_t0` = 1 to 5e-3 wherever δ̂ > 1e-2 (Δ has one direction per row by
  construction), NaN everywhere before t0.
- `peak`, `peak_r`, `half_life`, `plateau` agree exactly with an independent
  numpy re-derivation from the stored `delta`.
- raw-space `pr` = 170.0 = 512/3, the analytic value for a Gaussian direction in
  512 dims; `topk_share` = 0.143.
- `sae_TopKTrainer-othello` gives `dl0` = +0.0 exactly (TopK holds L0 fixed) —
  the codes really are being read.
- **fp16 storage floor: 3.5e-5 in δ̂.** Below roughly 1e-4, δ̂ is quantization
  noise in `acts/*.pt`, not signal, and `cos_t0` there is noise-vs-noise. Wave 2
  should not read structure out of δ̂ at that scale.

**A1's real set**, `$SPADE_DIVERGENCE_DIR` (2000 games, 15531 sequences, 13531
branch rows), read-only — the driver's output went to scratch, since the shared
tree is A1's to write. The prefix-invariance assertion (`δ̂ ≡ 0` for r < 0, exact
equality, all 13531 rows × 8 pre-event offsets) passes in every space, which
simultaneously confirms A1's extraction and my alignment. Depth sweep, means over
all 13531 branches pooled:

| space | δ̂(0) | peak | peak r | plateau r∈[20,24] | half-life censored |
|---|---|---|---|---|---|
| token_embed | 1.4136 | 1.4136 | 0.0 | 0.0000 | 0.0% |
| resid_post_0 | 1.1118 | 1.1118 | 0.0 | 0.2023 | 0.0% |
| resid_post_1 | 1.0277 | 1.0278 | 0.0 | 0.2529 | 0.0% |
| resid_post_2 | 0.9574 | 0.9581 | 0.1 | 0.2885 | 0.0% |
| resid_post_3 | 0.9024 | 0.9068 | 0.5 | 0.3256 | 0.2% |
| resid_post_4 | 0.8397 | 0.8627 | 2.1 | 0.3858 | 3.7% |
| resid_post_5 | 0.7889 | 0.8459 | 4.3 | 0.4760 | 28.6% |
| resid_post_6 | 0.7939 | 0.8800 | 5.0 | 0.5418 | 46.1% |
| resid_post_7 | 0.7839 | 0.8971 | 5.8 | 0.5684 | 52.5% |

Two things in that table are the study, and A3 owns them, but they are worth
recording because they are also the strongest evidence the pipeline is correct.
The token embedding's divergence is a **spike at r = 0 and exactly zero after**
— plateau 0.0000, every row's half-life = 1 — because the embedding at position
s depends only on the token at s and the suffix is shared. So the input floor is
a delta function, and everything at r > 0 in any resid layer is computed rather
than inherited. And δ̂(0) = 1.4136 ≈ √2 says two move-token embeddings are close
to orthogonal at equal norm. With depth, the peak moves later (r 0.0 → 5.8), the
plateau nearly triples (0.20 → 0.57), and censoring goes 0% → 52.5%.

## Runtime and memory, full run (13531 branch rows, 17 spaces)

Measured, CPU, one shared login node:

- **9 raw spaces: 2 m 29 s wall / 47 s user, all nine.** ~17 s each, mostly the
  938 MB `acts/<space>.pt` read from netscratch.
- **SAE spaces:** 2 × 13531 × 59 = 1.60 M encodes per space. Scaling the
  fixture's measured rate (29 k encodes in ~5 s CPU) gives ~4.5 min CPU per
  space, ~40 min for all eight; wall time on a contended login node ran ~2–4×
  that. On one H100 the encode is a 512→8192 GEMM totalling 12.8 TFLOP per space
  plus ~300 GB of memory traffic through the 8192-wide intermediates — seconds of
  compute, so per-space time collapses to the ~10 s load of `resid_post_5.pt`.
  **Budget ~10 min for the whole 17-space run on one GPU**, and put it on the
  GPU rather than waiting out the CPU path.

Memory, `ROW_BATCH = 64`: the code tensors are 64 × 59 × 8192 × 4 B = 124 MB
each, and at most four are live (`Fb`, `Fref`, `Δ²`, `Δ⁴`), so **peak device
memory ≈ 0.7 GB** regardless of dictionary count. Host peak is the acts tensor,
938 MB fp16, one space at a time. Output: 7.9 MB per raw npz measured, ~12 MB per
SAE npz (it carries `jaccard`, `dl0`, `topk_idx`), **≈ 165 MB** for
`$DIR/divergence/` in total.

The driver is resumable per space and `--spaces` filters, so a partial GPU
allocation loses nothing.

## Requests for the integrator

1. **M2's second half has no kernel.** The doc's M2 asks for
   `cos(Δ_b(r), Δ_control(r))` — illegal versus legal branches leaving along the
   same direction — but the interface block specifies only
   `direction_cosine(Zb, Zref, t0)`. I implemented the block as written, so
   nothing in `$DIR/divergence/` answers it. It needs a third argument
   (the control row of the same game), so it is a new kernel, e.g.
   `branch_cosine(Zb, Zc, Zref) -> [n, T]`, plus a `cos_control` npz key.
   Either add it to `divergence.py` or say explicitly that A3 computes it from
   `acts/` on its own.
2. **F4's Jaccard panel should have no raw-space reference line** — see the npz
   section. The raw line belongs on the top-10-share and PR panels only.
3. **`half_life` is the absolute offset, not elapsed since the peak** (Judgement
   1). If wave 2 wants elapsed, change it once, here, rather than in two figure
   scripts.
4. **Delete the `study_dir.py` stub assumption.** `compute_divergence.py` imports
   `violation_divergence.study_dir.divergence_dir`, which is A1's file; this
   worktree never had it, and the runs above supplied it from outside the repo
   via `PYTHONPATH`. Nothing to do at merge except confirm A1's file landed.
