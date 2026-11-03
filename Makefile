# The paper: `make setup`, `make data`, `make figures` (alias `make run`), then `make verify`.
# REPRODUCE.md has the per-figure commands, inputs and the compute-* recompute targets.
# Everything runs out of ./.venv. SPADE_REPO is what the shell scripts cd to.

# Must be 3.11; setup checks. Where python3 is another version: make setup PYTHON=/path/to/python3.11
PYTHON ?= python3
VENV   := .venv
PY     := $(VENV)/bin/python
# Shell steps call `python`, which must be the venv's.
VENV_PATH := PATH="$(CURDIR)/$(VENV)/bin:$$PATH"
STAGE  := $(PY) scripts/stage_figures.py

DETECTOR := analysis/violation_divergence/illegal_move_detector
FORK     := analysis/violation_divergence/fork_and_board/make_figures.py
DISTANCE := analysis/violation_divergence/divergence_by_space/make_figures.py

export SPADE_REPO = $(CURDIR)

FIGURES := fig01 fig02 fig03 fig04 fig05 fig06 fig07 fig08 \
           fig09 fig10 fig11 fig12 fig13 fig14 fig15 fig16

COMPUTE := $(addprefix compute-,$(FIGURES)) compute-branches compute-rollout-tree \
           compute-straightness compute-probes compute-roster

.PHONY: setup data figures run verify test $(FIGURES) $(COMPUTE)
# Redraws run on one thread: a multithreaded BLAS reduction is not deterministic run to run,
# and Fig 10's PNG then moves by a few pixels.
$(FIGURES): export OMP_NUM_THREADS := 1
$(FIGURES): export MKL_NUM_THREADS := 1
$(FIGURES): export OPENBLAS_NUM_THREADS := 1
# Some figures rewrite a results/ CSV that another asserts against.
.NOTPARALLEL:

setup:              ## fetch external/, create .venv, install requirements.lock, then spade editable
	@$(PYTHON) -c 'import sys; v = "%d.%d" % sys.version_info[:2]; sys.exit(v != "3.11" and "$(PYTHON) is Python " + v + ", setup needs 3.11: make setup PYTHON=/path/to/python3.11")'
	bash scripts/fetch_external/fetch.sh
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.lock
	$(PY) -m pip install -e . --no-deps
	$(PY) -c "import spade.common.paths as p; p.karvonen_root(); p.tpr_root(); print('spade ready at', p.repo_root())"

data:               ## the data release into $SPADE_DATA_ROOT (sha256-checked), then Karvonen's SAEs
	$(PY) scripts/fetch_release_data/fetch.py
	$(VENV_PATH) bash scripts/download_karvonen_saes/download.sh

figures: $(FIGURES) ## redraw all 16 paper figures on a CPU into figures/
run: figures

verify:             ## figures/ and results/ against manifests/{figures,results}.tsv
	$(PY) scripts/verify_outputs/verify.py

test:               ## fast self-tests (no GPU, no weights)
	$(PY) -m spade.validation.final_validation
	$(PY) -m spade.othello.test_taxonomy
	$(PY) -m spade.violation_divergence.test_divergence
	$(PY) -m spade.violation_divergence.test_curvature_plane

# One figure: its PNG, its camera-ready PDF, then figures/figNN_*.
fig01:              ## streams 607 games from the HF corpus for panel (a)
	$(PY) -m spade.figures.figure_rules_and_generation
	$(PY) -m spade.figures.figure_rules_and_generation --camera_ready
	$(STAGE) $@
fig02:
	$(PY) analysis/probe_directions_and_detection/make_figure.py
	$(PY) analysis/probe_directions_and_detection/make_figure.py --camera-ready
	$(STAGE) $@
fig03:
	$(PY) $(DETECTOR)/figure_overview.py
	$(PY) $(DETECTOR)/figure_overview.py --camera-ready
	$(STAGE) $@
fig04:
	$(PY) $(DETECTOR)/figure_auroc_l0_matched.py
	$(PY) $(DETECTOR)/figure_sae_overview.py
	$(PY) $(DETECTOR)/figure_sae_overview.py --camera-ready
	$(STAGE) $@
fig05:
	$(PY) analysis/board_probe_sweep/layer_sweep.py
	$(PY) analysis/board_probe_sweep/layer_sweep.py --camera_ready
	$(STAGE) $@
fig06:
	$(PY) -m spade.l0_operating_point.plot_pareto
	$(PY) -m spade.l0_operating_point.combine_figure
	$(STAGE) $@
fig07:
	$(PY) analysis/bracket_flip/make_figures.py --only fig1
	$(PY) analysis/bracket_flip/make_figures.py --only fig1 --camera-ready
	$(STAGE) $@
fig08:
	$(PY) analysis/violation_divergence/changepoint_localization/quantity_ablation.py --l0-matched
	$(PY) analysis/violation_divergence/changepoint_localization/quantity_ablation.py --l0-matched --camera-ready
	$(STAGE) $@
fig09:
	$(PY) $(FORK) --l0-matched
	$(PY) $(FORK) --l0-matched --camera-ready
	$(STAGE) $@
fig10:
	$(PY) $(FORK) --l0-matched --variant rollout
	$(PY) $(FORK) --l0-matched --variant rollout --camera-ready
	$(STAGE) $@
fig11:
	$(PY) $(FORK) --l0-matched --plane r21
	$(PY) $(FORK) --l0-matched --plane r21 --camera-ready
	$(STAGE) $@
fig12:
	$(PY) $(FORK) --l0-matched --plane r21 --variant rollout
	$(PY) $(FORK) --l0-matched --plane r21 --variant rollout --camera-ready
	$(STAGE) $@
fig13:
	$(PY) $(DISTANCE) --n_games 100 --l0-matched
	$(PY) $(DISTANCE) --n_games 100 --l0-matched --camera-ready
	$(STAGE) $@
fig14:
	$(PY) $(DISTANCE) --n_games 100 --l0-matched --raw
	$(PY) $(DISTANCE) --n_games 100 --l0-matched --raw --camera-ready
	$(STAGE) $@
fig15:
	$(PY) $(DISTANCE) --n_games 100 --l0-matched --plane r21
	$(PY) $(DISTANCE) --n_games 100 --l0-matched --plane r21 --camera-ready
	$(STAGE) $@
fig16:
	$(PY) $(DISTANCE) --n_games 100 --l0-matched --plane r21 --raw
	$(PY) $(DISTANCE) --n_games 100 --l0-matched --plane r21 --raw --camera-ready
	$(STAGE) $@

# Recompute from public sources (the HF model and game corpus, Karvonen's HF zips, WTHOR) into
# $SPADE_DATA_ROOT, then redraw. "GPU" marks a chain with a GPU step. A step whose output exists
# checks it, refuses, or keeps it (REPRODUCE.md): recompute into an empty data root.
M := $(PY) -m spade

compute-branches:     ## GPU: the branch tree (CPU, ~32 min), then its activations
	$(VENV_PATH) bash scripts/build_branch_tree.sh
	$(M).violation_divergence.extract_acts
compute-rollout-tree: compute-branches ## GPU: rollout branches and their activations; not bit-reproducible
	$(M).violation_divergence.rollout_branches
	$(M).violation_divergence.extract_acts --variant rollout
compute-straightness: ## GPU: the five game classes (downloads WTHOR), trajectories, raw geometry
	$(M).straightness.build_games
	$(M).straightness.extract_traj
	$(M).straightness.compute_geometry --spaces resid_post_5
compute-probes:       ## GPU: the probe corpus, then board and TPR probes at layers 0-7
	$(M).probes.build_corpus
	$(M).probes.train --arm board
	$(M).probes.train --arm tpr
	$(M).probes.tpr_weight_decay
compute-roster:       ## Karvonen's five SAEs; our three are trained weights from the release
	$(VENV_PATH) bash scripts/download_karvonen_saes/download.sh

compute-fig01:        ## GPU: the 1000-game rollout bank and its legality scan
	$(M).othello.rollout_bank --n_games 1000
	$(M).generation_legality.rollout_scan
	$(M).generation_legality.aggregate --rollout-only
	$(MAKE) fig01
compute-fig02: compute-branches compute-probes ## GPU
	$(M).move_legality.fit_set
	$(M).move_legality.extract_acts
	$(M).move_legality.readout --spaces raw
	$(PY) analysis/move_legality/detection_auroc.py --space raw
	$(M).probe_geometry.embed_directions
	$(MAKE) fig02
compute-fig03: compute-straightness compute-branches ## GPU
	$(PY) $(DETECTOR)/extract_geometry.py
	$(PY) $(DETECTOR)/eval_games.py
	$(PY) $(DETECTOR)/select_game.py
	$(PY) $(DETECTOR)/figure_auroc.py
	$(PY) $(DETECTOR)/figure_histograms.py
	$(MAKE) fig03
compute-fig04: compute-fig03 compute-roster ## GPU
	$(PY) $(DETECTOR)/extract_geometry.py --l0-matched
	$(PY) analysis/straightness/l0_matched_profile.py
	$(MAKE) fig04
compute-fig05: compute-probes ## GPU
	$(PY) analysis/board_probe_sweep/measure.py
	$(MAKE) fig05
compute-fig06:        ## CPU: the L0 roster from the released eval tables (training: REPRODUCE.md)
	$(M).l0_operating_point.aggregate
	$(M).l0_operating_point.pick
	$(MAKE) fig06
compute-fig07: compute-probes ## GPU
	$(M).bracket_flip.events
	$(M).bracket_flip.extract_acts
	$(M).bracket_flip.events --variant field --n_events 4000
	$(M).bracket_flip.extract_acts --variant field
	$(M).bracket_flip.probe_space
	$(MAKE) fig07
compute-fig08: compute-fig04 ## GPU
	$(PY) $(DETECTOR)/extract_geometry.py --l0-matched --quantities
	$(MAKE) fig08
compute-fig09 compute-fig11: compute-branches compute-roster ## GPU
	$(PY) analysis/violation_divergence/fork_and_board/game_acts.py
	$(MAKE) $(subst compute-,,$@)
compute-fig10 compute-fig12: compute-rollout-tree compute-roster ## GPU
	$(PY) analysis/violation_divergence/fork_and_board/game_acts.py --variant rollout
	$(MAKE) $(subst compute-,,$@)
# The by-space curves are measured on a CPU on purpose: all eight spaces on one device.
compute-fig13 compute-fig14: compute-branches compute-roster ## GPU
	CUDA_VISIBLE_DEVICES= $(PY) analysis/violation_divergence/divergence_by_space/measure.py --n_games 100 --l0-matched
	$(MAKE) $(subst compute-,,$@)
compute-fig15 compute-fig16: compute-branches compute-roster ## GPU
	CUDA_VISIBLE_DEVICES= $(PY) analysis/violation_divergence/divergence_by_space/measure.py --n_games 100 --l0-matched --plane r21
	$(MAKE) $(subst compute-,,$@)
