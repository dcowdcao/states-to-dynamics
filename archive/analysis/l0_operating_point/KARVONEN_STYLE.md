# Karvonen's OthelloGPT figures, reproduced

`plot_pareto.py` renders his four OthelloGPT plots. From his data alone the output is
**byte-identical to his published PNGs** — `verify_match.py` reports max channel difference 0
over all four images. Every style parameter below was read out of his
`circuits/f1_analysis.ipynb` (cells 24, 25, 27); nothing was chosen by us except the three
extensions in the last section.

His style, not ours, is the target here: `src/spade/figures/paper_style.py` and
FIGURE_GUIDELINES are deliberately not applied. See *Where this violates FIGURE_GUIDELINES*.

```
python load_points.py    # what landed, the L0 coverage table, the physics check
python plot_pareto.py    # renders/{karvonen_only, styled_karvonen_only, with_spade}
python verify_match.py   # pixel diff of renders/karvonen_only against his PNGs
python combine_figure.py # the two 2x2 composites the paper prints
```

All three figure sets come out of one `plot_pareto.py` run: `renders/karvonen_only` is
rendered from `source == "karvonen"` rows of the same frame, so our data cannot leak into the
figure `verify_match.py` checks. `--ours` narrows the streams; `--karvonen-csv` swaps his CSV.

## Data

| what | source | n |
|---|---|---|
| his points | `external/Karvonen_Research/autoencoders/saved_data/06-10-othello-trained-results_with_07-10_topk.csv` | 280 |
| ours | the `karvonen_repro.evaluate` output dirs in `load_points.OUR_STREAMS` | growing |

His published aggregate is the default, **not** `karvonen_othello_sweep/sweep_full.csv`, because
his figure was made from the aggregate and the two disagree: the 2024-05-23 groups were
evaluated a second time for the 2024-06-10 aggregate, so on 240 of the 280 rows L0 differs by a
median of 5.7 (max 51) and `frac_recovered` by a median of 5.8e-4 — the latter is 13% of the
3-var plot's y-range, i.e. plainly visible. Plotting `sweep_full.csv` would add his 40
unpublished TopK checkpoints but would no longer match his figure. It is still supported:
`--karvonen-csv .../sweep_full.csv` renames `recon_f1_bm`/`coverage_bm` on load.

Columns: x is `l0`; the 3-var y is `frac_recovered`; the reconstruction metric is
`games_batch_to_state_stack_mine_yours_blank_mask_BLRRC_best_f1_score_per_class` and coverage is
the same prefix with `_best_average_f1`. Our `f1_results.csv` files carry his exact schema
because they come out of his own `f1_analysis.add_coverage_to_df`.

## The loader

`load_points.py` puts his rows and ours on one frame — `source, trainer_class, config, seed,
l0, recon_f1, coverage, frac_recovered, variance_explained, dict_size, path` — and is the only
place that decides what a point is. A row it cannot identify raises; nothing is dropped quietly.

**Streams.** `OUR_STREAMS` names one `karvonen_repro.evaluate` output directory per sweep
under `$SPADE_KARVONEN_REPRO_DIR`. Each contributes its `f1_results.csv` once the whole set is
aggregated, plus every `per_checkpoint/**/f1_results.csv` written so far, so a sweep can be
plotted while it is still evaluating; the overlap between the two is deduplicated. Missing
directories are named on stdout and skipped. A new sweep is one new entry in that dict.

**Config identity comes from `config.json`, not from the path.** A trainer index is reused
between sweeps, so `trainer_4` is not a config. The identity is every key in the trainer
config except `device`, `seed` and `wandb_name`, which means a knob a future sweep adds (a
CosineSpaDE `beta`) splits configs rather than silently merging them. The name printed for a
config is built from `DIALS`, the subset that is actually swept; if a new key is missing from
`DIALS`, two identities collide on one name and the loader raises rather than mislabel.

**`lambda_init` is normalized through the trainer's own fp32 round trip.**
`SpaDEAutoEncoder` stores `softplus_inverse(lambda_init)` and `softplus`es it back;
`exp(x) - 1` quantizes in fp32, so everything in [5.96e-8, 1.19e-7] trains the identical
model. `trainer_8` (`lambda_init` 8e-8) and `trainer_9` (1e-7) are byte-identical in all four
seeds. Both become `SpaDE-lam1.19e-07-gam0.001` and plot as one point.

**Deduplication.** Within a (config, seed) group, rows with identical `l0` and identical
metrics are one measurement however the checkpoint was named — the same mirror read twice,
one checkpoint mirrored into two evaluation runs, or the `trainer_8`/`trainer_9` pair. Rows
that *disagree* were trained twice and only `RESOLVED_COLLISIONS` can say which is the point;
without an entry the loader raises and prints both paths and their L0s. One entry exists:
CosineSpaDE gamma=1 at seed 42 was trained twice under different standardization
(`checkpoints/cosine_spade` at L0 175.96, the seed sweep at L0 175.24) and the seed-sweep run
is kept, so gamma=1 is one internally consistent five-seed config. The same trap in the
multi-seed tree is `seed_*/autoencoders/trainer_7` next to
`seed_*/autoencoders_cosine_spade_sweep/trainer_4`: byte-identical CosineSpaDE gamma=1
weights in two directories. They currently carry equal `standardization_params.json` and so
measure equal L0, but the mirrors only ever evaluate the `_cosine_spade_sweep` member — the
same choice, made once, in one place.

**The physics check** runs on our rows before plotting and reports rather than filters:
`0 < L0 < dict_size`, and `recon_f1`/`coverage` above the best his 40 random-weight-model
SAEs reached. That floor is read out of `othello-6-10-random-f1_results.csv` for whichever
metric family is plotted. On the blank-masked family the figures use it is recon 0.0816 /
coverage 0.2723; the often-quoted 0.4267 / 0.4316 belong to the *unmasked*
`..._mine_yours_BLRRC` columns and are the wrong yardstick for these axes.

**The l0_coverage sweep is not plottable from its training-side evals.**
`scripts/slurm/l0_coverage/` evaluates with `spade.evaluation.run_eval` into
`$SPADE_MULTISEED_DIR/seed_*/results/karvonen_l0_coverage/all_results.csv`. That is our own
reimplementation of the protocol, which drifts from his by about 0.2 absolute F1
(`karvonen_repro/README.md`), and its columns are `coverage`/`reconstruction_f1`, not his.
Those numbers must not be mixed onto this figure. The checkpoints at
`$SPADE_MULTISEED_DIR/seed_*/autoencoders_l0_coverage/trainer_*` have to go through
`karvonen_repro.evaluate` into the `l0_coverage` stream directory first. When they do, expect
one new collision: `SpaDE-lam0.001-gam0.01` is retrained there and already exists in the
gamma sweep, so it will need a `RESOLVED_COLLISIONS` entry naming which run is the point.

## The style

Global `rcParams`: `font.size` 18, `axes.labelsize` 18, `xtick.labelsize` 18,
`ytick.labelsize` 18, `legend.fontsize` 16. Default DejaVu Sans, default figure dpi 100.
Every figure: `figsize=(10, 6)`, `plt.tight_layout()`, saved with `bbox_inches="tight"`
(hence the 966x565-ish pixel sizes rather than 1000x600). No title, no grid, no error bars,
default spines and ticks.

Shared axes: `set_xlim(0, 400)`, xlabel `$L_0$ (Lower is sparser)`, legend `loc="lower right"`
with default frame (white, `framealpha=0.8` — his low-F1 points show through it faded, which is
the legend, not a second alpha series). One `ax.scatter` per trainer class, `s=100`, in the
order the classes first appear in his CSV.

`plot_l0_metric` (the 2-var Pareto plot): y is the metric column, colour is
`plt.cm.Set1(i)` with `i` the class's position in that order — flat per-class colour, no
colormap, no edgecolor, no alpha. y-limits are left to matplotlib. y-labels
`Othello Board State Reconstruction $F_1$` and `Coverage of Othello Board State`.

`plot_3var_graph`: y is `frac_recovered`, ylabel `Loss Recovered (Fidelity)`,
`set_ylim(0.996, 1.0005)` for Othello. Marker colour encodes the metric: `cmap="viridis"`,
`norm=Normalize(df[col].min(), df[col].max())` **over the whole dataframe**, `edgecolor="black"`.
Colorbar from `ax.collections[0]`, label = the metric name. Legend handles are faked so they
show shape only: `scatter.legend_elements(prop="sizes")[0][0]` with black edge, white face,
markersize 10 (13 for `d`).

Class order, marker, `Set1` index, label — his five, then ours:

| trainer_class | label | marker | Set1 | colour |
|---|---|---|---|---|
| GatedSAETrainer | Gated SAE | `o` | 0 | red |
| GatedAnnealTrainer | Gated SAE w/ p-annealing | `X` | 1 | blue |
| PAnnealTrainer | Standard w/ p-annealing | `^` | 2 | green |
| StandardTrainer | Standard | `d` | 3 | purple |
| TrainerTopK | TopK SAE | `s` | 4 | orange |
| SpaDETrainer | SpaDE | `*` | 6 | brown |
| CosineSpaDETrainer | Cosine SpaDE | `P` | 7 | pink |
| BatchTopKTrainer | BatchTopK SAE | `v` | 8 | grey |

## What we had to invent

1. **Markers and colours for our three classes.** His `zip(unique_trainers, ["o","X","^","d"])`
   covers four; the fifth (`s`, TopK) is in his rendered PNGs but not in the committed notebook,
   so it was reconstructed from the images. Ours continue his own `Set1` in index order but skip
   index 5, which is `#FFFF33` yellow and unreadable on white — so brown, pink, grey. Markers
   are three unused filled shapes; `*` joins `d` in his markersize-13 legend rule because a star
   reads small at 10. Nothing else about the legend, sizes, or edges changed.
2. **Trainer classes are matched by equality, not `str.contains`.** His `str.contains(trainer)`
   gives the identical subsets on his five classes (checked), but `"SpaDETrainer"` is a substring
   of `"CosineSpaDETrainer"`, so contains would plot every cosine point twice, in two colours.
3. **Classes absent from the dataframe are skipped.** His loop would raise inside
   `legend_elements` on an empty scatter; without this, his CSV alone cannot be plotted with an
   eight-class table.

Two deletions, neither visible: the dead `idx = df[idx_column].values[0]` lookup (used only by a
commented-out title, and it forced callers to pass an `_idx` column name), and the `full_title`
argument, which is never read.

## Multiple seeds

His plot is one point per config, single seed; ours are trained over seeds 42-46. Our rows are
reduced to **the lowest seed present for each config**, which is seed 42 wherever the sweep ran
it, and the number of dropped rows is printed. The reduction groups by config identity, not by
path, so a config trained in two directories still contributes one point. No error bars are
added to these figures — the pixel match depends on it. Seed spread, if we want to show it,
belongs in a separate supplementary figure that is ours and follows FIGURE_GUIDELINES; the
per-seed rows it needs are what `load_points.load()` returns before `one_per_config`.

## The type, where it is ours

`karvonen_only` is his type and stays that way — it is the pixel match. The two composites the
paper prints are built from renders that keep every one of his axes, limits, colours, markers
and point sizes and change only the typeface and its sizes:

- **Face.** `OUR_FONTS` — Times/Nimbus serif, stix mathtext, `pdf.fonttype` 42. This is the
  typeface half of `paper_style.apply_style(camera_ready=True)` and only that half; the rest of
  `apply_style` sets a colour cycle, grids and spines that would overwrite what the figure
  exists to match.
- **Size.** `OUR_TYPE` adds 17 pt labels, 16 pt ticks and legend. Not paper_style's 9/8.5 —
  those are authored for a 5.4in canvas, and this one is 10in. 9/5.4 and 8.5/5.4 pt per inch
  rescaled to 10in is 17 and 16, so at equal printed width these panels carry the same apparent
  type as every other figure in the repo. His 18/18/16 was marginally larger.
- **Subcaptions.** `(a)`–`(d)`, nothing more. His wording ("Coverage of Board State vs. $L_0$")
  restated axis labels the panels already carry. Their size is derived, not chosen: a panel is
  drawn into a cell about half its authored width, so `combine_figure` scales the subcaption by
  the same factor and it lands on the panels' own label size (~8 pt) instead of twice it.

Two consequences worth stating. Point sizes are still his `s=100`, tuned for a 10in canvas, so
the markers read larger than a house figure's. And a composite is two panels wide: place it at
less than full text width and the panel type falls under the 9 pt floor, whatever this section
does.

## Where this violates FIGURE_GUIDELINES

Reported, not fixed, because matching him is the requirement:

- colours are `plt.cm.Set1` literals from his notebook, not `src/spade/figures/paper_style.py`;
  Set1 is not colourblind-safe (its red/green pair is the classic failure) and red vs orange vs
  brown do not separate in grayscale, which is exactly the palette rule the guidelines exist to
  enforce;
- no uncertainty on any point, and no n stated on the figure, though each point is a single
  training run;
- no reference class or null series is styled as such — TopK, our nearest baseline, is just
  another colour;
- the filename/label vocabulary ("2var", "3var") is his; the 18 pt type in a 10x6 inch figure
  is his too, and the composites now carry ours instead (*The type, where it is ours*);
- the 3-var plot's colour normalisation depends on the min/max of whatever rows are loaded, so
  adding our SAEs can rescale the colorbar and change the appearance of his points. It did not
  here (our metric values fall inside his range), but it is a property of his code worth knowing.

## Surprising things in his code

- `plot_3var_graph`'s colorbar is taken from `ax.collections[0]`, the first trainer's scatter.
  Correct only because every scatter shares one `norm` and cmap.
- `c=plt.cm.Set1(i)` in `plot_l0_metric` emits a matplotlib warning on every call (a 4-tuple
  that could be read as value-mapped data). It is his, so it stays.
- `frac_recovered` runs down to -0.81 and L0 up to 570 in his data; both are simply clipped
  off-axis by his limits, silently. Our SpaDE canonical checkpoint at L0 1841 disappears the
  same way — its legend entry shows with no visible points.
- The committed notebook drops TopK silently. `zip(unique_trainers, ["o","X","^","d"])` truncates
  to the shorter list, so the fifth class never enters `trainer_markers` — no error, just a
  missing series (and `label_lookup` has no `TrainerTopK` entry either). The published figures
  came from an edited working copy that is not in the repo.
