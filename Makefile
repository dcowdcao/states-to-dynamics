# Two-command interface (CLAUDE.md setup-docs target): `make setup`, `make run`.
# Skeleton — fill in once the code is ported.

.PHONY: setup run test clean

setup:              ## create venv, install package, vendor Karvonen_Research
	# python -m venv .venv && . .venv/bin/activate
	# pip install -e .
	# ensure external/Karvonen_Research is present (git submodule / rsync)

run:                ## full local pipeline: extract -> train -> eval -> figures
	bash scripts/run_pipeline.sh all

test:               ## fast self-tests (no GPU)
	python -m spade.validation.final_validation

clean:              ## remove generated outputs (keeps checkpoints)
	# rm -rf results/* figures/* data/activations/* logs/*
