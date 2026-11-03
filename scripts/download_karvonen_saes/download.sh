#!/usr/bin/env bash
# Download Karvonen's two OthelloGPT layer-5 SAE zips at a pinned Hugging Face revision,
# check them against zips.sha256, and extract only the trainer dirs the paper's roster reads.
#
#   SPADE_REPO=<repo root> bash scripts/download_karvonen_saes/download.sh
#
# Extracts into $SPADE_DATA_ROOT/karvonen_saes, where spade.l0_operating_point.checkpoints
# reads them; extracted_files.tsv holds each extracted file's sha256, relative to that root.
# SPADE_KARVONEN_ZIPS  where the zips are read from, and downloaded to when absent
#                      (default <that root>/zips; 9.3 + 3.0 GB, kept afterwards)
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
cd "${SPADE_REPO:?}"

HF_REVISION=f997cba04444628deda521daab4c8b9a48735af6
dest="$(python -m spade.common.study)/karvonen_saes"
zips=${SPADE_KARVONEN_ZIPS:-$dest/zips}
mkdir -p "$dest" "$zips"

while read -r _ zip; do
    [ -f "$zips/$zip" ] && continue
    curl -fL -o "$zips/$zip.partial" \
        "https://huggingface.co/adamkarvonen/othello_saes/resolve/$HF_REVISION/$zip"
    mv "$zips/$zip.partial" "$zips/$zip"
done < "$here/zips.sha256"
(cd "$zips" && sha256sum -c "$here/zips.sha256")

extract() {  # zip, then the trainer dirs to take from it
    local zip=$1 patterns=()
    shift
    for d in "$@"; do patterns+=("${zip%.zip}/$d/*"); done
    unzip -q -o "$zips/$zip" "${patterns[@]}" -d "$dest"
}

# Both Gated dirs: the roster picks trainer32, and the published L0-matched figures encoded trainer32-v2.
extract othello-trained_model-layer_5-2024-05-23.zip \
    othello-trained_model-layer_5-gated/trainer32 \
    othello-trained_model-layer_5-gated/trainer32-v2 \
    othello-trained_model-layer_5-gated_anneal/trainer25 \
    othello-trained_model-layer_5-p_anneal/trainer22 \
    othello-trained_model-layer_5-standard/trainer25-v2
extract othello-trained_model-layer_5-2024-07-10_topk.zip \
    othello-trained_model-layer_5-top_k/trainer_61
echo "Karvonen roster SAEs in $dest"
