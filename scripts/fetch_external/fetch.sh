#!/usr/bin/env bash
# Clone each third-party dependency into external/ at its pinned commit and apply our patch.
# A tree that is already there is left alone, so a checkout that vendors them keeps its copy.
#
#   SPADE_REPO=<repo root> bash scripts/fetch_external/fetch.sh
set -euo pipefail

patches=$(cd "$(dirname "$0")" && pwd)/patches
cd "${SPADE_REPO:?}"
mkdir -p external

fetch() {  # url sha dest patch
    git init -q "$3"
    git -C "$3" fetch -q --depth 1 "$1" "$2"
    git -C "$3" checkout -q --detach FETCH_HEAD
    git -C "$3" apply --whitespace=nowarn "$patches/$4"
}

stage=$(mktemp -d external/.fetch.XXXXXX)
trap 'rm -rf "$stage"' EXIT

if [ ! -e external/Karvonen_Research ]; then
    fetch https://github.com/adamkarvonen/SAE_BoardGameEval.git \
        7187d4a19ffa62157eb6a19c0b2c3cd6d151dcef "$stage/Karvonen_Research" Karvonen_Research.patch
    fetch https://github.com/saprmarks/dictionary_learning.git \
        43421f5934a1476cb3f32f0b9e1b5d14b84540a1 "$stage/Karvonen_Research/circuits/dictionary_learning" \
        dictionary_learning.patch
    mv "$stage/Karvonen_Research" external/
    echo "fetched external/Karvonen_Research"
fi

if [ ! -e external/tpr_othello ]; then
    fetch https://github.com/ajyl/tpr_othello.git \
        cc90cead6cf66a8d3e25ceec6c0951112066fbdf "$stage/tpr_othello" tpr_othello.patch
    mv "$stage/tpr_othello" external/
    echo "fetched external/tpr_othello"
fi
