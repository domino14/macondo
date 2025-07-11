#!/usr/bin/env bash
set -e

# Activate the virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "venv/bin/activate not found!"
    exit 1
fi

# Backup the best.pt file
cp best.pt best-bak.pt

# Export the model
python export.py
python onnx-to-tensorrt.py --onnx macondo-nn.onnx

# Find the next unused version number
MODEL_DIR="../data/strategy/default/models/macondo-nn"
n=1
while [ -d "$MODEL_DIR/$n" ]; do
    n=$((n+1))
done

# Create the new directory
mkdir -p "$MODEL_DIR/$n"

# Move the exported ONNX model
mv macondo-nn.onnx "$MODEL_DIR/$n/model.onnx"
mv scrabble_model.engine "$MODEL_DIR/$n/model.plan"

echo "Model exported to $MODEL_DIR/$n/model.plan"