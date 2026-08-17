# The board probe's mover came from the move index (branch `fix/probe-parity-fix`)

The gold-standard board-state probe labelled every row from the perspective of "the player to
move", and took that player from move-index parity. Othello skips a player who has no legal
move, so one skip puts parity out of phase with the true mover for the rest of that game. The
real script is ported from the frozen repo, the mover now comes from engine replay, and every
artifact under `analysis/linear_probe/activations/` is regenerated.

- `analysis/linear_probe/train_linear_probe.py` — the port, the fix, the tripwire.
- `analysis/linear_probe/parity_fix_effect.py` — both labellings on one set of activations.
- `analysis/linear_probe/make_figures.py` — the layer sweep, drawing only.
- `analysis/linear_probe/{results,figures}/` — the tracked outputs.
- `src/spade/othello/boards.py` — `next_mover` moved here; `bracket_flip/events.py` had it.
- `reference/.../part53_othellogpt/board_seqs_id_small.npy` — vendored, 6 MB.
- `scripts/slurm/studies/linear_probe/submit_linear_probe.sbatch` — the study had no job.
- `analysis/linear_probe/README.md`, `analysis/README.md`, `docs/claims_status.md`,
  `reference/README.md`.

## The numbers, both ways

Layer sweep, 200 epochs, mean over probe-init seeds 42–46. `previous` is the frozen repo's
committed single-seed `probe_metadata.json`; `index parity` is this code run under the old
labelling; `corrected` is this code as it now stands.

| layer | previous | index parity (this code) | corrected | fix, paired | seed SD |
|---|---|---|---|---|---|
| 0 | 0.8903 | 0.8902 | 0.8904 | +0.0002 | 0.00003 |
| 1 | 0.9360 | 0.9359 | 0.9367 | +0.0008 | 0.00004 |
| 2 | 0.9652 | 0.9652 | 0.9656 | +0.0005 | 0.00008 |
| 3 | 0.9789 | 0.9790 | 0.9803 | +0.0013 | 0.00003 |
| 4 | 0.9867 | 0.9867 | 0.9890 | +0.0023 | 0.00007 |
| **5** | **0.9896** | 0.9896 | **0.9919** | **+0.0023** | 0.00005 |
| 6 | 0.9870 | 0.9870 | 0.9896 | +0.0026 | 0.00005 |
| 7 | 0.9860 | 0.9862 | 0.9884 | +0.0022 | 0.00008 |

The saved probe is the 500-epoch fit at layer 5: **0.9923 corrected, 0.9884 under the old
labels, +0.0039 paired.** Primary layer is still 5. New rows on the axis: the embedding space
(`resid_pre` block 0) at 0.6271, and a per-square majority-class null at 0.5449.

**Not a null result.** The seed SD is 3–8 × 10⁻⁵, so layers 4–7 move by 25–50 SDs and the
500-epoch probe by 70. 53 rows of 8,000 is 0.66% of the fit set, but a mislabeled row is a
whole-board mine/theirs inversion — coherent noise, not random noise, and a linear probe pays
for it. `results/parity_fix_effect.csv` has the 100 fits behind the paired column.

**`previous` and `index parity` agree to 1.5 × 10⁻⁴ at every layer and 2 × 10⁻⁵ at layer 5.**
That is the check that the port reproduces the original rather than quietly changing something
else, and it is what licenses reading the paired column as the labels: `corrected − previous`
matches `fix` to the same tolerance, so the entire discrepancy against the old headline is the
labels and none of it is the rewrite.

**The eval split never saw a bad row, and that is worth stating precisely.** Rows are
`game * 50 + (turn - 5)`, the split is positional, and rows 8000:10000 are games 160–199. The
three desyncing games are **70, 75 and 146** — all in the training half. So 0.9896 was
*measured* against correct labels; what was wrong was the probe being measured, not the
measurement. Had one landed at ≥ 160 the old number itself would have been meaningless.

**One space moves the other way.** The embedding probe is 0.0007 *worse* under correct labels.
Parity is a function of the position index, which the positional embedding carries and the
board does not, so at the input the wrong target was the easier one. It is a useful signature:
if a space gains from parity labels it is reading the clock, not the board.

## The fix, and the tripwire

`replay()` takes the mover from `spade.othello.boards.next_mover(state, next_hand_color)` —
R4 applied to the board: the nominal mover, or the opponent when the nominal mover cannot flip
anything. Nothing consults the move record.

`check_mover()` then asserts the mover named at position *i* equals the colour of the disc
that move *i+1* actually places, read out of the state array, which no mover code path writes.
This is not decoration: **it fails on the first try under `next_hand_color` alone.** The
engine flips `next_hand_color` twice inside `umpire` on a forfeit, so the nominal next mover
is not always the player who moves next, and the brief's "take `next_hand_color` per move"
would still have been wrong at exactly the forfeit positions the bug is about. Index parity
fails the same assertion at every skip.

`ParityAudit` prints, and `metadata.json` records, the desync slot count (308/12,000), the
mislabeled row count (53/10,000), the game indices, and whether any of them sit in the
positional eval range.

`next_mover` came out of `bracket_flip/events.py`, where it was defined and used once, into
`spade.othello.boards` beside `regular_moves`, which it calls. Two studies now need the same
rule, and `violation_divergence/behavioral/make_figures.py` writes the same forfeit fallback
inline for the legal set; a shared helper between two studies is the wrong direction, so the
rule went down to the game layer. `events.py`'s call site is
`next_mover(board.state, board.next_hand_color)`, matching `boards.py`'s `(state, color)` house
signature; behaviour is unchanged.

That module's `probe_labels` is also the independent check on the convention. DATA_REPORT has
it reproducing the committed set on 9,947/10,000 rows with the 53 differing rows being exactly
the parity-wrong ones; the corrected labels differ from the committed set on exactly those 53
rows, which puts the two pipelines at 10,000/10,000 with each other.

## Row order, checked rather than assumed

`bracket_flip/tpr_probe.py` pairs `all_layer_activations[L]` against `board_states.pt` by row
index with no key, so a reorder would be silent. Against the frozen artifacts:

- `move_indices.pt` — **bit-identical**.
- `board_states.pt` — same shape and dtype (`[10000, 8, 8]` int64), **exactly 53 rows differ**,
  all in games 70/75/146, and the frozen file is *exactly* the index-parity encoding.
- `all_layer_activations.pt` — same shape and dtype, max abs difference **2.0 × 10⁻⁴**, i.e.
  the same activations up to device arithmetic.

`train_linear_probe.py` now asserts the row contract itself: shapes, and
`board_states == labels[row_game, move_indices]`, which checks the documented index formula
against the layout through independent indexing rather than through the reshape that built it.
An assert on the reshape would only have restated torch.

## Artifacts overwritten

All under `analysis/linear_probe/activations/` in this worktree, which is what
`spade.common.paths.linear_probe_activations()` resolves to here (gitignored, so no commit
carries them):

| file | what changed |
|---|---|
| `board_states.pt` | 53 of 10,000 rows relabelled |
| `linear_probe.pt` | refit on the corrected labels; eval 0.9896 → 0.9923 |
| `primary_layer_activations.pt` | re-extracted layer 5; ≤ 2 × 10⁻⁴ from the old |
| `all_layer_activations.pt` | re-extracted; ≤ 2 × 10⁻⁴ from the old |
| `move_indices.pt` | bit-identical |
| `metadata.json` | new keys: `row_order`, `mover_source`, the parity audit; `note` gone |
| `probe_metadata.json` | `layer_accuracies` now the 5-seed mean (0.9896 → 0.9919 at layer 5), plus `layer_accuracies_sd`, `sweep_seeds`, `probe_eval_acc`, `embed_accuracy`, `majority_accuracy` |
| `embedding_activations.pt` | new; `resid_pre` at block 0, for the sweep's input anchor |

`probe_metadata.json` is written twice, to `activations/` where `bracket_flip.tpr_probe` reads
it and to `results/` where the figure and README can read it from git.

**The shared copy at `/n/home10/dcao/spade_sae/analysis/linear_probe/activations/` still holds
the old files.** It is gitignored, hand-copied, and outside this worktree, so it was left
alone; anything pointing `$SPADE_LINEAR_PROBE_ACTIVATIONS` at it gets the old labels.

## What was vendored, and why

Only `board_seqs_id_small.npy` (6 MB), verbatim from ARENA_3.0, committed. It is the study's
entire input and the repo's own rule is that a fresh clone must be able to run it; the largest
file already tracked is a 8.9 MB CSV, so this is in-band. Deliberately **not** vendored:

- `board_seqs_square_small.npy` — verified to be exactly `to_string()` of the id array on all
  100,000 × 60 entries, so it is the same data twice. Derived at load.
- `utils.py` — its `OthelloBoardState` differs from `external/Karvonen_Research`'s copy only in
  whitespace and one dead local, and both engines return identical desync counts on these
  games. Two engines in one tree is how conventions drift; the probe now uses the one every
  other study uses.
- `solutions.py`, `tests.py`, both notebooks — nothing reads them, and one notebook is large.

## Mechanics

`sbatch scripts/slurm/studies/linear_probe/submit_linear_probe.sbatch` — job **39705245** for
the numbers, COMPLETED 7:23 exit 0, then **39706784** and **39707576** re-running the committed
file. The reruns reproduce all six `.pt` artifacts and all three tracked result files
**md5-identical**, so the pipeline is deterministic on this machine and the committed sbatch is
the one that produced what is here. Logs under `logs/linear_probe/`. The committed job is CPU
on `sapphire`, and the three gotchas behind that are worth keeping:

- **`kempner_h100` was fully drained or reserved** all night; three GPU submissions never
  started (39703338, 39704600, 39705157). Sixteen CPU cores finish this study in seven minutes,
  so the sbatch asks for those instead of queueing for a scarce H100 it does not need. The
  scripts/slurm README's `kempner_ba_lab / kempner_h100` preamble is the GPU-study default and
  this is the deviation, commented in the file.
- **`kempner_requeue` contains Blackwell nodes that torch 2.5.1+cu124 has no kernels for.**
  Job 39704799 landed on `holygpu7c1734` (RTX PRO 6000 Blackwell) and died with `CUDA error: no
  kernel image is available for execution on the device`. On any requeue partition, pin
  `--gres=gpu:nvidia_h100_80gb_hbm3:1`.
- **The login node gives one core** (`nproc` = 1 under load average 40). A first attempt there
  was on track for ~90 minutes. What bites this job on the login node is the core count, not
  the 8 GB cap — peak RSS is ~1.3 GB.

The numbers are therefore CPU arithmetic. A GPU rerun should be expected to differ in the last
10⁻⁴, the same order as the activation difference against the frozen files, and well above the
5 × 10⁻⁵ seed spread — so quote the sweep to four decimals, not five.

## The figure

`figures/layer_sweep_accuracy.{png,pdf}` replaces a plotly bar chart with a **y-axis starting
at 0.86**, no uncertainty, no n, and default styling. Now two panels off
`results/layer_sweep.csv`: (a) the full 0–1 range with the embedding space as its own tick,
ruled off from the blocks (§4.1's input anchor), and the majority-class null as gray dashed
(§3); (b) the same series zoomed, where the layer differences live.

A truncated axis is honest for marks with intervals and dishonest for bars, so the bars are
gone and the marks carry ±1 SD over five seeds with the per-seed fits drawn behind the mean.
Two things the figure says out loud rather than hiding: the band is **optimisation noise
only** — games, split and activations are fixed — and at SD ≈ 5 × 10⁻⁵ it is smaller than the
markers. Sampling noise over games is not measured; a claim that needs it wants a resample
over the 200 games, and that is a different figure.

Colours and markers come from `paper_style.PROBE_STYLE["linear"]`, `REFERENCE` and `NULL`; no
literals. Checked in grayscale (it is already black plus one gray) and at final width.

## Not done

- **Nothing downstream was re-run**, per the brief. The eight cached TPR probe fits under
  `$SPADE_BRACKET_DIR/tpr_probes/` were left in place; `train()` writes a `labels` fingerprint
  that `load()` never reads, so they will be silently reused against the corrected data.
- **Stale mentions of the defect elsewhere were left alone**, because they are true of the runs
  they describe: `analysis/bracket_flip/README.md` ~line 195,
  `analysis/bracket_flip/reports/DATA_REPORT.md` ~line 130,
  `src/spade/legality_probe/spaces.py` ~line 66, and the 0.9896 quoted in
  `docs/bracket_flip_design.md` and `docs/violation_divergence_design.md`. `docs/claims_status.md`
  now points at this note for which side of the rebuild a number is on.
- **`bracket_flip/events.py:parity_mismatch` still derives its own parity from `move_indices`**
  — deliberately: it is the diagnostic that measures the desync, not a labeller.
- **9 inches wide, not a paper column.** `claims_status.md` already carries "nothing in the
  source repo is drawn at final printed width" as an open defect; this figure follows the
  repo's README width rather than fixing that separately.
- **The 200 games and the 8,000/2,000 positional split are unchanged.** A random split would
  be better and would also change every downstream row index, so it is not this change.
- `PROCESS.md` untouched, as instructed. Nothing pushed.
