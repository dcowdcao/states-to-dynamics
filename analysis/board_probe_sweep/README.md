# Board-probe layer sweep, under the canonical probe

Where in OthelloGPT's residual stream is the board state linearly readable, measured
with the probes every study in this repo now reads through `spade.probes.load.board`.
One bar per `resid_post` layer, **layer 0 being the first block**; 3-way {empty, theirs,
mine} accuracy over all 64 squares, mover-relative, on **n = 1,000 held-out games**
(58,000 positions). The panel carries no title and states the indexing base nowhere on
its axes, so this paragraph is its caption.

This is the canonical-probe answer to the question `analysis/linear_probe/` asked with
the retired ARENA fit. That study stays as the record of what it measured (0.9919 at
layer 5, five seeds, 200 ARENA games) and is not repointed here; it also still writes
an activation cache four other modules read. The two are not comparable line for line —
different corpus, different labels, different split — and only this one describes a
probe anything currently loads.

## What is measured

`probes.load.board(layer)` is loaded, not refit. Its saved card already carries a test
accuracy over the full 3,000-game split (`card_test_accuracy` in the CSV, beside the one
measured here); the two agree to 2.5-3.5 x 10^-4 at every layer, which is the subsample's
own noise. The point of re-measuring is the two things a scalar cannot carry:

- **An interval.** Positions inside a game share a board, so the bootstrap resamples
  **test games**, not rows. 1,000 resamples, 95% percentile CI.
- **A null.** Per square, the class most common in the *train* rows, scored on the same
  test rows the probes are scored on. Fitting the null on the split it is scored on
  would make it a ceiling.

The split is `probes.train.game_split` at seed 42 — the same partition the probes were
fit under, so the test games here are test games there. `N_TEST_GAMES = 1000` takes the
head of that split's 3,000; the split is a permutation, so the head is already a uniform
random subsample and needs no second draw.

## The cut baseline, and why

FIGURE_GUIDELINES §1.6 says bar charts start at zero. **This one does not**, and that is
a deliberate departure. Every layer past 0 reads above 0.91; at a zero baseline the
eight bars are one flat block and the figure says nothing it was drawn to say. The
differences it exists to show — layer 0 to layer 5, and the flat top from 5 to 7 — live
in the second and third decimal.

Two things make the cut defensible where a bare cut baseline is not. Each bar carries
its interval, so the axis cannot manufacture a gap the marks do not support; and every
value is printed above its bar, so the reader never has to measure a height. The cost is
that the null cannot be drawn — it sits far below the cut — so it is a number here and a
row in the CSV rather than a line on the panel.

## Reading it

- Bars are the canonical board probe at each layer, one hue: it is one probe class over
  eight input spaces, not eight conditions.
- **Layer 5 is drawn darker and labelled "chosen layer"** — the project's hook point,
  not the argmax. Layer 6 is the argmax and beats it by ~0.001, inside what a
  single-seed sweep can resolve, which is why the hook point did not move. The panel
  does not mark the argmax; that reading belongs here, not on the bars.
- Error bars: 95% cluster bootstrap over held-out games, 1,000 resamples.
- **Seed 42 only.** The interval is over held-out data and says nothing about the spread
  over fits. The ARENA study's five-seed SD was 3–8 × 10⁻⁵, so fit noise is not what
  separates these layers, but that is evidence from a different corpus.

## Reproducing

    python analysis/board_probe_sweep/layer_sweep.py                 # results/ + figures/
    python analysis/board_probe_sweep/layer_sweep.py --camera_ready  # vector PDF

One entry point: it measures and draws in the same run, so a committed figure and a
committed CSV cannot drift apart. Needs `$SPADE_PROBES_DIR` (corpus + probes) and a GPU
for the read-out pass. Cluster:
`scripts/slurm/studies/board_probe_sweep/submit_board_layer_sweep.sbatch`.
