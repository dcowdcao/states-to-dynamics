# Finish note — the archive zone

Zone: `archive/`. Before this it held `spade_sweep_collapsed/` and a README listing
candidates. `ARCHITECTURE.md` routed six things here; four landed, two are gated on open
questions.

## Structure

One rule: **an entry is named after the live thing it was split from.** The residue of
`results/lambda_sweep/` is `archive/lambda_sweep/`; superseded `docs/` is `archive/docs/`.
No grouping by reason — a `negative_results/` or `superseded/` tier would be exactly the
abstract naming CLAUDE.md forbids, and the reason is one column of the README table
instead. `spade_sweep_collapsed/` keeps its name, which already carries both the source dir
and the reason.

## What landed

| here | from the source repo |
|---|---|
| `spade_sweep_8config/` | `8_spade_configs_updated/autoencoders_spade_sweep/` — 8 configs + standardization params |
| `lambda_sweep/` | `lambda_sweep_results/autoencoders_spade_sweep/` — 4 configs + standardization params |
| `main_mac_smoke_eval/` | `gamma_sweep_results/karvonen/smoke_eval_trainer_{0..5}.log` |
| `Wandb_Export.csv` | `results/Wandb_Export.csv` |
| `docs/Refactor_Cleanup_Progress.md` | `docs/Refactor_Cleanup_Progress.md`, whole |

All byte-identical to the source except the archived doc, which gained a six-line header.
No manifest: unlike the weights in `manifests/checkpoints/`, every file here is text that
git tracks and hashes itself.

The three sweep generations reused one save-dir name, `autoencoders_spade_sweep/`, so each
generation's configs are the only record of which λ/γ pair produced which `trainer_N` row
in the matching live CSV. That is why the configs are worth keeping at all, and the README
says so. `gamma_sweep_results/` carries no configs — that generation kept only logs.

## Refactor_Cleanup_Progress.md landed whole

`docs/port_notes/docs.md` extracted the λ/γ canonicalization fact from it and did not port
the file. The extraction does not supersede the file. Its last section is the only record
anywhere of the deleted `Run_4_9_weight_init` run (217 files, reachable at commit
`7de4353`, the likely source of the VE 0.995 claim), of the unresolved 0.995-vs-0.43
discrepancy for the matching config, and of which session prose (4/7, 4/18, 4/21, 4/29)
exists only in the progress PDF. `grep -r Run_4_9` over this repo returns nothing.
Deciding which further facts deserve extraction is the owner's call; keeping 32 KB verbatim
costs nothing and forecloses none of it. The header note follows the `docs/session_logs/`
convention — read as a record of its date, nothing below runs as written.

`archive/docs/` rather than `ARCHITECTURE.md`'s `archive/old_md/`: `old_md` names a file
format, not a purpose, and reusing the name of the source repo's own `archive/old_md/`
would read as though that directory (question 5) had been carried in. `ARCHITECTURE.md`'s
row is updated.

## Left out, and why

- **Every image.** The three sweep `Figure_4_*.png`, `best_spade_lam1e-07_gam5e-03/` (a
  single PNG) and the six `W&B Results/` chart exports, ~8 MB. **Question 10.** One rule
  applied to all of them rather than case-by-case, so the owner executes it in one copy.
  The slots are named in `archive/README.md`.
- **The source repo's `archive/`** — `runs/Run_{2,3}_Results/` (126 MB), `old_md/`,
  `gpu_runs_plans/`, `run_notes/`. **Question 5.** Its markdown joins `archive/docs/` if
  it comes.
- **The four collapsed configs stay as they are.** **Question 2**, untouched.
- **SLURM stdout/stderr from the sweep evals**, 1.2 MB. `ARCHITECTURE.md` routed them here;
  they are tqdm progress spam over numbers the eval CSVs already carry, against MIGRATION's
  own rule that logs are transient. Row updated. The six Mac `.log` files are the exception
  and did land — see below.

## For the owner

- **`archive/main_mac_smoke_eval/` is evidence for question 3, and possibly 15.** Six
  evaluations of `checkpoints/main/trainer_0–5` run on a Mac at `device: mps`, N=520, from
  `/Users/dcao/Downloads/...` — the same machine and configuration as the Mac smoke
  `config.json`s question 3 is about. They disagree with the committed rows in
  `results/main/all_results.csv` for all six SAEs, by a lot: trainer_0 L0 2780 vs 339,
  trainer_2 VE 0.439 vs 0.982, trainer_5 L0 776.65 / VE 0.2665 / fidelity −0.2541 against
  the committed 15.43 / 0.352 / 0.597. The protocols differ (N=520 vs 59000), so this is
  not a clean third measurement — but trainer_5 now has three recorded L0 values from three
  evaluations of one weight file, and question 15 is about exactly that.
- **The three sweep CSVs these configs describe are not in this repo.**
  `results/{spade_sweep_8config,lambda_sweep,gamma_sweep_lam1e-07}/` hold a `.gitkeep` and
  nothing else, as do `results/main/`, `multiseed/` and six more. `results/README.md` calls
  them live inputs and `spade.multiseed.aggregate_seeds` hard-codes them. That is a gap in
  the results zone, not this one, but it is what makes these archived configs currently
  unpairable.
- **`../Spade_Research_David_Sumedh_Ba/CLAUDE.md` has an uncommitted modification** that
  predates this session. The source repo is supposed to be frozen.
