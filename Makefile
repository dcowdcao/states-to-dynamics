# Two-command interface (CLAUDE.md setup-docs target): `make setup`, `make run`.
# Everything runs out of ./.venv. SPADE_REPO is what the shell scripts cd to.

# Must be 3.11; setup checks. Where python3 is another version: make setup PYTHON=/path/to/python3.11
PYTHON ?= python3
VENV   := .venv
PY     := $(VENV)/bin/python

export SPADE_REPO = $(CURDIR)

.PHONY: setup run test

setup:              ## fetch external/, create .venv and install spade editable with the pinned deps
	@$(PYTHON) -c 'import sys; v = "%d.%d" % sys.version_info[:2]; sys.exit(v != "3.11" and "$(PYTHON) is Python " + v + ", setup needs 3.11: make setup PYTHON=/path/to/python3.11")'
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
