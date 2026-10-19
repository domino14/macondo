#!/usr/bin/env bash
# Build the TensorRT engines (model.plan) that Triton serves, from the
# model.onnx files in the model repository, inside the Triton container
# itself so the TensorRT version always matches the server.
#
#   pytorch/build-engines.sh              # build missing or stale engines
#   FORCE=1 pytorch/build-engines.sh      # rebuild them all
#   pytorch/build-engines.sh macondo-tf   # just these models
#
# A .plan only loads on the TensorRT version and GPU generation that built
# it, so engines are not committed: run this once after cloning, and again
# after changing GPUs or the Triton image. Then (re)load the models in
# Triton (pytorch/triton-models.sh load NAME, or restart the server).
#
# Engines are fp16. TensorRT 11 takes precision from the ONNX graph, so each
# model.onnx (fp32, also what the CPU path runs) is first converted to an
# fp16 copy (onnx_fp16.py, the same step onnx-to-tensorrt.py does).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${MODELS:-$REPO/data/strategy/default/models}
IMAGE=${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:26.08-py3}
MAX_BATCH=128 # game.MLMaxBatch

todo=()
for cfg in "$MODELS"/*/config.pbtxt; do
    dir=$(dirname "$cfg")
    name=$(basename "$dir")
    if [ $# -gt 0 ] && [[ " $* " != *" $name "* ]]; then continue; fi
    grep -q 'platform: *"tensorrt_plan"' "$cfg" || continue
    for onnx in "$dir"/*/model.onnx; do
        [ -f "$onnx" ] || continue
        plan=$(dirname "$onnx")/model.plan
        if [ -n "${FORCE:-}" ] || [ ! -f "$plan" ] || [ "$onnx" -nt "$plan" ]; then
            todo+=("${onnx#"$MODELS"/}")
        fi
    done
done
if [ ${#todo[@]} -eq 0 ]; then
    echo "all engines are up to date (FORCE=1 to rebuild)"
    exit 0
fi
echo "building ${#todo[@]} engine(s) with $IMAGE: ${todo[*]}"

docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint bash \
    -v "$MODELS":/models -v "$REPO/pytorch/onnx_fp16.py":/opt/onnx_fp16.py:ro \
    "$IMAGE" -c '
set -euo pipefail
pip install -q --disable-pip-version-check --target /tmp/py onnx==1.22.0 onnxconverter-common==1.16.0 \
    > /tmp/pip.log 2>&1 || { cat /tmp/pip.log; exit 1; }
export PYTHONPATH=/tmp/py
max='"$MAX_BATCH"'
for rel in "$@"; do
    dir=/models/$(dirname "$rel")
    echo "== $rel"
    python3 -W ignore /opt/onnx_fp16.py "/models/$rel" /tmp/fp16.onnx
    shapes() { echo "board:$1x85x15x15,scalars:$1x72"; }
    trtexec --onnx=/tmp/fp16.onnx --saveEngine="$dir/model.plan.tmp" --skipInference \
        --minShapes="$(shapes 1)" --optShapes="$(shapes $((max / 2)))" --maxShapes="$(shapes $max)" \
        --memPoolSize=workspace:1024M > /tmp/trtexec.log 2>&1 \
        || { tail -20 /tmp/trtexec.log; exit 1; }
    mv "$dir/model.plan.tmp" "$dir/model.plan"
    echo "   wrote $(dirname "$rel")/model.plan"
done
' _ "${todo[@]}"
echo "done; load or reload the models in Triton to use the new engines"
