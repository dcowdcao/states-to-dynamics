# `make setup`, then `make test`.
# Everything runs out of ./.venv. SPADE_REPO is what the shell scripts cd to.

# Must be 3.11; setup checks. Where python3 is another version: make setup PYTHON=/path/to/python3.11
PYTHON ?= python3
VENV   := .venv
PY     := $(VENV)/bin/python
# Shell steps call `python`, which must be the venv's.
VENV_PATH := PATH="$(CURDIR)/$(VENV)/bin:$$PATH"

export SPADE_REPO = $(CURDIR)

.PHONY: setup data test

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

test:               ## fast self-tests (no GPU, no weights)
	$(PY) -m spade.validation.final_validation
