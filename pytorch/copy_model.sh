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

# A new model name needs its own config.pbtxt; the interface (inputs,
# outputs, dims) is identical for every architecture, only the name changes.
if [ ! -f "$MODEL_DIR/config.pbtxt" ]; then
    sed "s/^name: .*/name: \"$MODEL_NAME\"/" "$MODELS_ROOT/macondo-nn/config.pbtxt" \
        > "$MODEL_DIR/config.pbtxt"
    echo "Created $MODEL_DIR/config.pbtxt"
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
