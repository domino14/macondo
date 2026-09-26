#!/usr/bin/env bash
# Export a checkpoint to ONNX + TensorRT and install it as the next version
# of a Triton model.
#
#   ./copy_model.sh                          # CNN: best.pt -> models/macondo-nn/<n>/
#   ./copy_model.sh macondo-nn-tf best-tf.pt # transformer, served side by side
#
# Select the model from Go with MACONDO_TRITON_MODEL_NAME=<name>.
set -e

MODEL_NAME="${1:-macondo-nn}"
CKPT="${2:-best.pt}"

# Activate the virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "venv/bin/activate not found!"
    exit 1
fi

# Backup the checkpoint
BAK="${CKPT%.pt}-bak.pt"
cp "$CKPT" "$BAK"

# Export the model
python export.py --ckpt "$BAK" --out "$MODEL_NAME.onnx"
python onnx-to-tensorrt.py --onnx "$MODEL_NAME.onnx" --output "$MODEL_NAME.engine"

MODELS_ROOT="../data/strategy/default/models"
MODEL_DIR="$MODELS_ROOT/$MODEL_NAME"
mkdir -p "$MODEL_DIR"

# Triton's config.pbtxt is per model name, not per version, and it lists
# the outputs. Every version under a name must therefore expose the same
# outputs: a model with new heads goes under a new name.
OUTPUTS=$(python -c 'import sys, onnx; print(" ".join(o.name for o in onnx.load(sys.argv[1]).graph.output))' "$MODEL_NAME.onnx")
if [ -f "$MODEL_DIR/config.pbtxt" ]; then
    EXISTING=$(python -c '
import re, sys
cfg = open(sys.argv[1]).read()
body = cfg[cfg.index("output ["):]
body = body[:body.index("]")]
print(" ".join(re.findall(r"name: \"([^\"]+)\"", body)))
' "$MODEL_DIR/config.pbtxt")
    if [ "$EXISTING" != "$OUTPUTS" ]; then
        echo "config.pbtxt for $MODEL_NAME lists outputs [$EXISTING] but the export has [$OUTPUTS]." >&2
        echo "Use a new model name for this export." >&2
        exit 1
    fi
else
    python - "$MODEL_NAME" "$MODEL_DIR/config.pbtxt" $OUTPUTS <<'PY'
import sys
name, path, *outputs = sys.argv[1:]
out_blocks = ",\n".join(
    f'  {{\n    name: "{o}"\n    data_type: TYPE_FP32\n    dims: [ -1 ]\n  }}' for o in outputs
)
open(path, "w").write(f'''name: "{name}"
platform: "tensorrt_plan"
max_batch_size: 0
input [
  {{
    name: "board"
    data_type: TYPE_FP32
    dims: [ -1, 85, 15, 15 ]
  }},
  {{
    name: "scalars"
    data_type: TYPE_FP32
    dims: [ -1, 72 ]
  }}
]
output [
{out_blocks}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator : [ {{ name : "tensorrt" }} ]
  }}
}}
''')
PY
    echo "Created $MODEL_DIR/config.pbtxt with outputs: $OUTPUTS"
fi

# Find the next unused version number
n=1
while [ -d "$MODEL_DIR/$n" ]; do
    n=$((n+1))
done

# Create the new directory
mkdir -p "$MODEL_DIR/$n"

# Move the exported ONNX model
mv "$MODEL_NAME.onnx" "$MODEL_DIR/$n/model.onnx"
mv "$MODEL_NAME.engine" "$MODEL_DIR/$n/model.plan"

echo "Model exported to $MODEL_DIR/$n/model.plan"
