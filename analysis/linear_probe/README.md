# Linear probe — historical

**Superseded as a probe.** The project's board read-out is `spade.probes.load.board(5)`:
20,000 `adamkarvonen/othello_45MB_games` games, 812k train rows split **by game**,
engine-replay labels, early stopping on validation accuracy, **0.9937** test at layer 5
against this study's 0.9923. Do not fit a new probe here, and do not read
`activations/linear_probe.pt` as one.

The study stays for two reasons. `train_linear_probe.py` also writes the ARENA activation
cache (`activations/{board_states,primary_layer_activations,all_layer_activations,
move_indices}.pt`) that the bracket-flip, rules and geometry code still reads, and
`results/` plus `figures/` are the record of the fit the canonical protocol replaced.

`train_linear_probe.py` replays 200 ARENA games, caches OthelloGPT's residual stream at
every layer, sweeps the layers with quick probes over five initialisations at a fixed 200
epochs, and trains the full probe on the best one for a fixed 500 epochs. Neither budget
has a convergence criterion — that is the defect `spade.probes.train` exists to fix.

### Why the figure was not repointed at the canonical sweep

`$SPADE_PROBES_DIR/probes/board_layer_sweep.csv` is single-seed (42) and holds only
`layer, best_epoch, epochs_run, val_accuracy, test_accuracy` for layers 0–7. It carries no
embedding-space row and no majority-class null, and the canonical corpus has no embedding
activations to compute one from. Drawing it on `layer_sweep_accuracy`'s axes would delete
all three of that figure's structural elements — the ±1 SD seed bars (FIGURE_GUIDELINES
§1.2), the input anchor ruled off from the blocks (§4.1) and the null (§3) — which is a
redesign, not a repoint. The canonical per-layer test accuracies are .9109 .9494 .9720
.9828 .9899 .9937 .9945 .9940 for layers 0–7; **layer 6 edges layer 5 by 0.0009**, inside
the noise this single-seed sweep cannot resolve, and layer 5 stays the hook point.

    python analysis/linear_probe/train_linear_probe.py    # activations/ + results/
    python analysis/linear_probe/parity_fix_effect.py     # results/parity_fix_effect.csv
    python analysis/linear_probe/make_figures.py          # figures/ [--camera-ready]

Cluster: `scripts/slurm/studies/linear_probe/submit_linear_probe.sbatch`. `activations/`
(~210 MB of `.pt`) is gitignored; `results/` and `figures/` are tracked. The games are
`reference/.../part53_othellogpt/board_seqs_id_small.npy`, vendored from ARENA.

## The mover comes from replay, not from the move index

Labels are mover-relative — `{0 empty, 1 theirs, 2 mine}` from the perspective of the player
to move — so every row needs to know whose turn it is. ARENA's construction
(`solutions.py:401-408`) takes that from move-index parity, and **Othello skips a player who
has no legal move**, so one skip puts parity out of phase with the true mover for the rest of
that game. Parity is wrong at 308 of the 12,000 game slots here, and inside the kept turn
window [5, 55) it mislabels **53 of 10,000 rows, in games 70, 75 and 146** — the defect
`docs/claims_status.md` carried, reproducing `bracket_flip/reports/DATA_REPORT.md`'s
9,947/10,000 cross-check exactly.

The mover now comes from `spade.othello.boards.next_mover`, which is R4 applied to the board
(the nominal `next_hand_color`, or its opponent when the nominal mover cannot flip) and never
looks at the move record. `check_mover` asserts it against the colour of the disc the *next*
move actually places, read off the state array that no mover code path touches — index parity
fails that assertion at every skip, which is what makes a desync loud instead of silent. The
count, and the games it lands in, go into `activations/metadata.json`.

**The split is positional and all three games are in the training half.** Rows are
`game * 50 + (turn - 5)` and rows 8000:10000 are games 160–199, so no mislabeled row was ever
in the eval set. The old 0.9896 was therefore *measured* against correct labels, and the fix
reaches it only through what the probe was fit on — which is still worth +0.0023, because 53
whole-board colour inversions are coherent label noise, not random.

## Layer sweep

Mean over probe-init seeds 42–46, held out on 2,000 positions × 64 squares. `previous` is the
committed single-seed ARENA-labelled run (frozen repo `probe_metadata.json`); `fix` is the
paired difference from `parity_fix_effect.py`, which refits **both** labellings on one set of
activations with one set of seeds, so it isolates the labels from the machine.

| space | corrected | SD over 5 seeds | previous | fix (paired) |
|---|---|---|---|---|
| embedding (`resid_pre` block 0) | 0.6271 | 0.00007 | — | −0.0007 |
| layer 0 | 0.8904 | 0.00003 | 0.8903 | +0.0002 |
| layer 1 | 0.9367 | 0.00004 | 0.9360 | +0.0008 |
| layer 2 | 0.9656 | 0.00008 | 0.9652 | +0.0005 |
| layer 3 | 0.9803 | 0.00003 | 0.9789 | +0.0013 |
| layer 4 | 0.9890 | 0.00007 | 0.9867 | +0.0023 |
| **layer 5** | **0.9919** | 0.00005 | 0.9896 | +0.0023 |
| layer 6 | 0.9896 | 0.00005 | 0.9870 | +0.0026 |
| layer 7 | 0.9884 | 0.00008 | 0.9860 | +0.0022 |
| per-square majority class (null) | 0.5449 | — | — | — |

The saved probe is the full 500-epoch fit on layer 5: **0.9923**, against **0.9884** for the
same fit under the old labels. **This is not a null result.** The seed SD is 3–8 × 10⁻⁵, so
every layer past 3 moves by 25–50 SDs, and the 500-epoch probe by 70.

The port reproduces the original, which is what licenses reading the `fix` column as the
labels rather than the machine: run under index-parity labels this code returns 0.8902 /
0.9359 / 0.9652 / 0.9790 / 0.9867 / 0.9896 / 0.9870 / 0.9862 — within 1.5 × 10⁻⁴ of the
`previous` column at every layer and within 2 × 10⁻⁵ at layer 5, and `corrected` − `previous`
matches `fix` to the same tolerance.

Layer 5 is still the best layer, which is what matters downstream: Karvonen's SAEs are
trained there.

The embedding row is the anchor the sweep needs (FIGURE_GUIDELINES §4.1): at position *t* the
input carries the move just played and its position, nothing else, so 0.627 against the 0.545
majority-class null is the floor that licenses "the board state is computed", not inherited.
It is also the one space where the fix makes the probe *worse* — the parity label is partly a
function of the position index, which the positional embedding carries and the board does not,
so the wrong target was the easier one there.

## Figure

`figures/layer_sweep_accuracy.png` (and `.pdf`), from `results/layer_sweep.csv` only.
(a) the whole 0–1 range, anchored at the embedding space and ruled off from the blocks, with
the majority-class null. (b) the same series zoomed, where the layer-to-layer differences
live. It replaces a bar chart whose baseline was cut to 0.86 and which carried no uncertainty
and no n: bars must start at zero (FIGURE_GUIDELINES §1), and marks with intervals are the
form in which a truncated axis stays honest.

The band is ±1 SD over five probe initialisations, which is optimisation noise only — the
games, the 8,000/2,000 split and the activations are all fixed, and at SD ≈ 5 × 10⁻⁵ the bars
are smaller than the markers. That is the honest statement of what varies here, and it is why
the per-seed fits are drawn behind the mean rather than summarised away. Sampling noise over
games is *not* measured; a claim that needs it wants a resample over the 200 games.

## Downstream

`activations/{linear_probe,board_states,primary_layer_activations,all_layer_activations,
move_indices}.pt` are read by `spade.rules.causal.unflip_edit` and four `spade.bracket_flip`
modules (`tpr_probe`, `tpr_intervene`, `sae_arm`, `act_geometry`, plus `redteam`);
`probe_metadata.json`'s
`layer_accuracies` is the capacity reference in `bracket_flip.tpr_probe`. Everything is paired
**by row index with no key**, so the row set and order above are a contract, and
`train_linear_probe.py` asserts them. Anything computed before this fix used the old labels —
`docs/finish_notes/probe-parity-fix.md` lists what.
