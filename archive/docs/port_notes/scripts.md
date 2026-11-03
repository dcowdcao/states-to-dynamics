# Port notes — `scripts/`

All 37 live scripts from the source repo's `shell/` are ported. `bash -n` passes on every
one; nothing was run, since this port had no cluster access. Verification is syntax plus a
read against the source.

## What changed everywhere

**Anchoring.** The source had four conventions; all became `cd "${SPADE_REPO:?}"`. The
eleven scripts that did `cd $HOME/Spade_SAE_Research` were not copied forward — that path
died in the 2026-07 reorg, and under `set -e` those jobs aborted at the `cd` while their
submitter reported success. `REPO="${SLURM_SUBMIT_DIR}"` and the `$(dirname)`-relative
drivers are gone the same way. Two consequences worth knowing: `SPADE_REPO` must be set in
the submitting shell (sbatch propagates the environment by default), and a job submitted
without it now fails immediately and loudly, which is the point.

**Invocation.** `python src/<pkg>/<file>.py` → `python -m spade.<pkg>.<module>`, using the
renames in ARCHITECTURE.md's code table. `python analysis/<study>/make_figures.py` and the
nine `analysis/geometry/*` calls stay as file paths — those are scripts, not modules.

**Log paths.** Every `#SBATCH --output=../../logs/...` became `logs/...`. The directive
resolves against the *submitting* directory, so under the new "submit from the repo root"
convention `../../logs` points two levels above the repo: unwritable, and the job dies at
launch with no log. See "Log directories must exist" below.

**Artifact dirs.** `autoencoders*` → `checkpoints/<experiment>`, `results/karvonen*` →
`results/<experiment>`, per ARCHITECTURE.md. Netscratch paths and every `SPADE_*_DIR` name
are byte-identical to the source — real data is at those exact names.

## Naming

Bash drivers keep `.sh`; sbatch job scripts use `.sbatch`. The skeleton already named the
`main/` jobs `submit_{train,extract,eval}.sbatch` and `multiseed/run_multiseed.sh` already
referenced them, so the split is the target tree's own; it is applied to the other 26 job
scripts for consistency. Reversing it is a rename plus the seven `sbatch <path>` lines in
the four drivers.

Two files were renamed beyond the extension:

- `shell/slurm/submit_karvonen_eval.sh` → `main/submit_eval.sbatch`, mirroring
  `run_karvonen_eval.py` → `evaluation/run_eval.py` (skeleton's choice).
- `shell/cosine_gamma_range/submit_probe.sh` → `sweeps/submit_cosine_gamma_probe.sbatch`.
  "submit_probe" says nothing once the file sits in `sweeps/` next to nine other probes and
  sweeps.

## Deliberate weirdness kept

- **`--export` is comma-delimited.** `run_cosine_sweep_multiseed.sh` exports
  `COSINE_GAMMAS` / `COSINE_SWEEP` into the submitting shell and passes `--export=ALL`,
  because `--export=COSINE_GAMMAS=10,20,50` is parsed as three variables and only the first
  value survives. Kept verbatim, comment and all.
- **Three HF locking strategies.** `HF_USE_SOFTFILELOCK=1` (rules, straightness,
  manifold_capture, bracket sae/flip/intervene, multiseed train + eval, revival, gamma
  probe), a node-local `HF_DATASETS_CACHE` (violation_divergence, cosine sweep eval), and
  the `~/.cache/huggingface/token` block (batch_topk eval, cosine_spade eval). Softfilelock
  is not honoured by the installed datasets/filelock, so the scripts that actually hit the
  concurrent-flock failure moved to a node-local cache and the rest never did. Each script
  keeps its own choice.
- **`submit_bracket_sae_pca`** logs to `logs/bracket/`, not `logs/bracket_flip/` like its
  four siblings, and is the only bracket job that does not `module load cuda`. Both are as
  in the source. The cuda omission is probably harmless (it only runs a matplotlib figure
  script) and the log dir is probably a typo, but neither was changed.
- **Soft-fail merges.** `merge_baseline_results.py` skips a missing CSV and prints a
  warning rather than raising, which is the inline code's behaviour. House style would
  crash here; this is a port, so it was not changed.

## The two fixes asked for

**`clean_smoke_test.sh` now requires `--force`.** It deletes the same six trees it always
did — `checkpoints/main/*`, `data/activations/*`, `results/*`, `figures/*`, `wandb/*`,
`logs/*` — but it was never smoke-test-scoped, so unguarded in a real checkout it took
production checkpoints and result CSVs with it. Note the delete set is asymmetric and was
inherited that way: checkpoints are scoped to the `main` tree (the old `autoencoders/*`)
while results, figures and logs are wiped whole.

**`submit_perf_game_diag.sbatch` takes the checkpoint as `$1`**, defaulting to
`checkpoints/spade_sweep/trainer_3`. That index is only the λ=1e-3, γ=1e-1 point for one
exact grid (λ ∈ {1e-3,1e-4} × γ ∈ {1e-4,1e-3,1e-2,1e-1}) under one loop order
(for-lam-for-gam). Against any other sweep it silently diagnoses the wrong SAE.

## The inline python

`submit_spade_only_eval.sh` and `submit_spade_sweep_eval.sh` each carried a `python -c`
pandas merge. Both bodies implement the same rule — take the non-SpaDE rows of the baseline
CSV, concatenate the sweep's own CSV, write `combined_results.csv` beside it — differing
only in the results dir. They are now one module,
`scripts/slurm/sweeps/merge_baseline_results.py`, called with `--results_dir`.

It lives under `scripts/` because the port of `scripts/` runs ahead of the port of `src/`.
Its real home is `src/spade/evaluation/merge_baseline_results.py`, beside
`merge_revival_results.py`, which does the same job for the revival run. Moving it is a
one-line change in each of the two callers.

**The filter is wrong and was kept.** `~sae_type.str.contains("SpaDE", case=False)` also
matches `CosineSpaDE`, so the "baseline" half of every combined CSV silently drops the
CosineSpaDE row along with the SpaDE ones. Harmless for the sweep this was written for;
wrong the moment a combined CSV is expected to carry CosineSpaDE as a baseline. Worth a
decision before these two jobs are run again.
*Fixed since — see `docs/finish_notes/flagged-defects.md`.*

## Things to know before submitting

**Log directories must exist.** `logs/` is gitignored and SLURM does not create the parent
of an `--output` path — it fails the job at launch, and the log that would tell you so is
the thing that failed. The four bash drivers `mkdir -p` what they submit; the hand-submitted
study and diagnostic jobs cannot, because their own `mkdir -p logs/<study>` runs after the
job has already started. `scripts/slurm/make_log_dirs.sh` is new and creates the full set;
run it once per checkout.

**Submit from the repo root.** `#SBATCH --output` is relative to the submitting directory,
so `cd "$SPADE_REPO" && sbatch scripts/slurm/studies/rules/submit_rules.sbatch`. Submitting
from elsewhere still runs correctly — the job body `cd`s to `SPADE_REPO` — but the logs land
under wherever you submitted from.

**`submit_cosine_gamma_probe.sbatch` gained `--output`/`--error`.** The source had neither,
so its logs went to `slurm-%j.out` in the submit dir. It now writes to `logs/training/`
like its neighbours. This is the one place a log path was added rather than corrected.

## Judgement calls to check

**`train_karvonen_matched.py` → which entrypoint.** Three cases:

- No filter, the full multi-type run (`main/submit_train.sbatch`, `run_pipeline.sh`,
  `smoke_test.sh`) → a loop over the eight `spade.training.train_<type>` entrypoints. This
  is the form the skeleton spelled out in three places. **It is not semantically identical
  to the source**, which trained every type in a single `trainSAE()` call over one shared
  activation stream. Eight sequential calls re-stream the data eight times, and whether
  `trainer_N` indices come out in the same order is a property of the new entrypoints, not
  of these scripts. `checkpoints/main/trainer_3` must still be TopK (MIGRATION.md wave 3
  verifies L0 = 80.00 against it).
- `--trainer_filter <X>Trainer` → `spade.training.train_<type>`. The flag is dropped since
  the module *is* the filter.
- Any grid — `--spade_sweep`, `--cosine_spade_sweep`, `--spade_pairs` →
  `spade.training.sweep`, flags unchanged. The multiseed train job in particular *must*
  stay one call: it trains 14 configs device-parallel across 4 GPUs and its eval array
  indexes `trainer_0..13` by array task id.

**`autoencoders_spade_v2` / `results/karvonen_spade_v2`** are not in ARCHITECTURE.md's
artifact table. They were mapped by the general rule to `checkpoints/spade_v2` and
`results/spade_v2`. If those two dirs are meant to be dead, `run_spade_only.sh` and its two
jobs are dead with them.

**`SPADE_LINEAR_PROBE_ACTIVATIONS` default changed.** The source defaulted it to
`$HOME/Spade_Research_David_Sumedh_Ba/analysis/linear_probe/activations` — an absolute
pointer at the old checkout, since the dir is gitignored. It now defaults to
`${SPADE_REPO}/analysis/linear_probe/activations`. This makes the anchoring rule hold, but
it only works once the 220 MB of probe activations are hand-copied into this tree
(MIGRATION.md, "Things git will not move for you"). Until then, submit
`submit_bracket_intervene.sbatch` with the variable set explicitly.

**`spade.training.sweep_loss_balance`** takes `--sweep_dir`, inherited from
`src/cosine_gamma_range/loss_balance.py`. The gamma probe passes it; confirm the flag
survives the module's port.

## Not carried over

`shell/slurm/submit_legality.sh` — already deleted upstream, replaced by the two `rules`
scripts. Nothing else in `shell/` was dropped.
