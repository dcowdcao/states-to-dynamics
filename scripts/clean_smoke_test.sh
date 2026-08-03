#!/bin/bash
# Delete every artifact a smoke-test run produces.
#
# Usage: SPADE_REPO=<repo root> bash scripts/clean_smoke_test.sh --force
#
# The delete set is not smoke-test-scoped — it wipes the whole results, figures
# and log trees plus the main checkpoint tree — so --force is required rather
# than defaulted. The original ran unguarded and would take production
# checkpoints and result CSVs with it in any real checkout.

set -e

cd "${SPADE_REPO:?}"

if [ "${1:-}" != "--force" ]; then
    echo "refusing to run without --force." >&2
    echo "this deletes, under ${SPADE_REPO}:" >&2
    echo "  checkpoints/main/*  data/activations/*  results/*  figures/*  wandb/*  logs/*" >&2
    exit 1
fi

echo "Cleaning smoke test artifacts from: ${SPADE_REPO}"

rm -rf checkpoints/main/*
rm -rf data/activations/*
rm -rf results/*
rm -rf figures/*
rm -rf wandb/*
rm -rf logs/*

echo "Done."
