Training code for the value nets (CNN and transformer) with PyTorch.


#### Installation

On my Linux box with NVIDIA gfx card (your setup may vary). The venv is
Python 3.14; `tensorrt` must match the TensorRT shipped in the Triton
container you run (26.08 -> TensorRT 11.2.1), since a `.plan` engine only
loads on the exact TensorRT version that built it.

```
uv venv --python 3.14 venv && source venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy onnx onnxruntime "tensorrt==11.2.1.2" pycuda polygraphy torchviz matplotlib
```

#### Running pipeline

- Go to `cmd/mlproducer` and do `go build`
- Collect lots of data with autoplay and copy /tmp/autoplay.txt somewhere
- Activate your venv then `cat /tmp/autoplay.txt | ../cmd/mlproducer/mlproducer | python3 training.py`
- You can pipe in `pv -br` after the call to `mlproducer` to see what the byte rate is. I see like 700 MiB on my computer. It can probably be made faster with more efficient `mlproducer` code.
- Wait a long time (maybe not that long, depends on how many games you used).
- Profit!

#### Convert

export.py - export to onnx
onnx-to-tensorrt.py - make sure the python tensorrt version matches whatever the triton container expects (see the NVIDIA frameworks support matrix). this part is a pain in the ass. When bumping the container, rebuild every `model.plan` from its `model.onnx`.
#### Architectures

`training.py --arch cnn` (default) is the ResNet; `--arch transformer` trains
`transformer_model.py` on the same inputs (see `--help` for sizes, `--ckpt`,
`--csv`). Checkpoints record their architecture, so `export.py --ckpt X.pt`
rebuilds the right model. `copy_model.sh <model-name> <ckpt>` exports to ONNX +
TensorRT and installs the next version under
`data/strategy/default/models/<model-name>/`, creating a `config.pbtxt` for a
new name. Pick the model from Go with `MACONDO_TRITON_MODEL_NAME`.
`export-tester.py` checks ONNX vs PyTorch parity on real frames.

#### Heads, weights, epochs

Five heads (value, spread, wdl, opp_bingo, opp_score). `--w-<head>` sets a
fixed loss weight; `--aux-share F` instead rebalances the auxiliary weights
at every validation so each head's measured pull on the trunk is F times
the value head's (recommended; `head_grads.py` shows the same measurement
for a checkpoint). The checkpoint is chosen on value val loss only.

`--epochs N --cache frames.bin` caches every stdin frame (bit-packed, 2.7 KB
each) during epoch 1 and reads epochs 2..N from it; `--from-cache frames.bin`
trains from an existing cache (first `--val-size` rows are validation).
`--total-steps` is the cosine length and the stop point; `--snapshot-every`
keeps intermediate checkpoints.

`run-gen1.sh` is the unattended generation-1 pipeline (rollout labels ->
5 epochs -> deploy -> paired match); `watch-tf-heads.sh` is the deploy +
match step on its own; `pairs-stats.py` summarizes a game-pairs CSV.
