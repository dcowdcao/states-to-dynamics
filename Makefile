# Two-command interface (CLAUDE.md setup-docs target): `make setup`, `make run`.
# Skeleton — fill in once the code is ported.

.PHONY: setup run test clean

setup:              ## create env, install package editable, vendor the external forks
	# python -m venv .venv && . .venv/bin/activate
	# pip install -e .            # editable: paths anchor on the repo root marker
	# ensure external/{Karvonen_Research,tpr_othello} are present, pinned, and complete
	#   (their own .gitignore hides files circuits needs at runtime — see external/README.md)

run:                ## full local pipeline: extract -> train -> eval -> figures
	bash scripts/run_pipeline.sh all

test:               ## fast self-tests (no GPU)
	python -m spade.validation.final_validation

clean:              ## remove generated outputs (keeps checkpoints)
	# rm -rf results/* figures/* data/activations/* logs/*
