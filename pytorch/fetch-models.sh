#!/usr/bin/env bash
# Download the published value nets into the model repository Triton serves
# (data/strategy/default/models), from the same Hugging Face revision the
# code downloads from (DefaultMLModelsURL in config/config.go).
#
#   pytorch/fetch-models.sh              # every published net
#   pytorch/fetch-models.sh macondo-tf   # just these
#
# Needs only curl. The CPU path downloads a missing net by itself; Triton
# reads the folder directly, so a Triton setup runs this once, then
# pytorch/build-engines.sh. Each model.onnx gets a model.onnx.source file
# naming the URL it came from, as the CPU path writes, so a later revision
# replaces it. Files already downloaded from this revision are skipped.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${MODELS:-$REPO/data/strategy/default/models}
URL=${MACONDO_ML_MODELS_URL:-$(sed -n 's/^const DefaultMLModelsURL = "\(.*\)"$/\1/p' "$REPO/config/config.go")}
[ -n "$URL" ] || { echo "no DefaultMLModelsURL in config/config.go" >&2; exit 1; }
NAMES=("$@")
[ ${#NAMES[@]} -gt 0 ] || NAMES=(macondo-tf macondo-tf-csw macondo-tf-selfplay)

for name in "${NAMES[@]}"; do
    dir=$MODELS/$name
    src=$URL/$name/1/model.onnx
    if [ "$(cat "$dir/1/model.onnx.source" 2>/dev/null)" = "$src" ]; then
        echo "$name: up to date"
        continue
    fi
    mkdir -p "$dir/1"
    curl -fsSL -o "$dir/config.pbtxt" "$URL/$name/config.pbtxt"
    curl -fSL --progress-bar -o "$dir/1/model.onnx.tmp" "$src"
    mv "$dir/1/model.onnx.tmp" "$dir/1/model.onnx"
    printf '%s' "$src" > "$dir/1/model.onnx.source"
    echo "$name: downloaded"
done
echo "now build the engines: pytorch/build-engines.sh"
