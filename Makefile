# Two-command interface (CLAUDE.md setup-docs target): `make setup`, `make run`.
# Everything runs out of ./.venv. SPADE_REPO is what the shell scripts cd to.

PYTHON ?= python3.11
VENV   := .venv
PY     := $(VENV)/bin/python

export SPADE_REPO = $(CURDIR)

.PHONY: setup run test

setup:              ## fetch external/, create .venv and install spade editable with the pinned deps
	bash scripts/fetch_external/fetch.sh
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .
	$(PY) -c "import spade.common.paths as p; p.karvonen_root(); p.tpr_root(); print('spade ready at', p.repo_root())"

run: export PATH := $(CURDIR)/$(VENV)/bin:$(PATH)
run:                ## full local pipeline: train -> extract -> eval -> figures
	bash scripts/run_pipeline.sh all

test:               ## fast self-tests (no GPU, no weights)
	$(PY) -m spade.validation.final_validation
