# macondo

A crossword board game solver. It may be the best one in the world (so far).

Current master build status:

![Build status](https://github.com/domino14/macondo/actions/workflows/build-and-deploy-bot.yml/badge.svg)

# What is a crossword board game?

A crossword board game is a board game where you take turns creating crosswords
with one or more players. Some examples are:

- Scrabble™️ Brand Crossword Game
- Words with Friends
- Lexulous
- Yahoo! Literati (defunct)

# How to use Macondo:

See the manual and information here:

https://domino14.github.io/macondo

# protoc

To generate pb files, run this in the macondo directory:

`go generate`

Make sure you have done

`go install google.golang.org/protobuf/cmd/protoc-gen-go@latest`

# Creating a new release

(Notes mostly for myself)

Tag the release; i.e. `git tag vX.Y.Z`, then `git push --tags`. This will kick off a github action that builds and uploads the latest binaries. Then you should generate some release notes manually.

# Using Triton

If you want to use the neural network model, I recommend you use Triton server
rather than the default Go server. It's _much_ faster, but it's not trivial to run locally.
You need Docker and the NVIDIA Container Toolkit (for GPU access from Docker).

1. **Build the engines, once per machine.** Triton runs each model as a
   TensorRT engine (`model.plan`), compiled from the committed `model.onnx`
   for your GPU. Engines are not in git because one only loads on the GPU
   generation and TensorRT version that built it. This builds every missing
   or out-of-date engine inside the same Triton image the server uses, so
   the versions always match (about 20 s per model):

   ```
   pytorch/build-engines.sh
   ```

   Run it again after changing GPUs or the Triton image (set `TRITON_IMAGE`
   if you use a different one than below; `FORCE=1` rebuilds everything).

2. **Start Triton.** If your macondo directory is at `$HOME/code/macondo`:

   ```
   docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002     -v $HOME/code/macondo/data/strategy/default/models/:/models     nvcr.io/nvidia/tritonserver:26.08-py3     tritonserver --model-repository=/models
   ```

3. **Point macondo at it:** set the `MACONDO_TRITON_USE_TRITON` environment
   variable to `true` (and `MACONDO_TRITON_URL` if Triton's gRPC port is not
   `localhost:8001`).

The model is chosen with `triton-model-name` and `triton-model-version` in
the config (or `MACONDO_TRITON_MODEL_NAME` / `MACONDO_TRITON_MODEL_VERSION`).
With `--model-control-mode=explicit`, Triton must also be told to load it
(`--load-model=<name>` or `pytorch/triton-models.sh load <name>`).

## Models

Two value nets ship in `data/strategy/default/models/`. Both are the same
transformer (8 layers, d_model 192, a token per square), scoring each of the
top 50 static-equity candidates by win probability:

| name | trained on | vs HastyBot | head to head |
|---|---|---|---|
| `macondo-tf` (default) | 54M HastyBot self-play games, 6 passes, one position per game per pass | 57.65% ± 0.18 | |
| `macondo-tf-selfplay` | `macondo-tf` fine-tuned on 3M of its own games (mixed 50/50 with 3M HastyBot games) | 57.07% ± 0.18 | beats `macondo-tf` 50.37% ± 0.16 |

Win rates are 100k game pairs, ties counted half, with a 95% interval. The
self-play net exploits HastyBot a little less but beats its parent head to
head. Training runs and matches are logged in `pytorch/RUNS.md` and
`pytorch/experiments.md`.

To pick one for a whole session, set `triton-model-name` (e.g.
`MACONDO_TRITON_MODEL_NAME=macondo-tf-selfplay`). In `autoplay`,
`-tritonmodel1` / `-tritonmodel2` give each player its own model, so the two
can play each other.

Each model folder holds:

- `model.onnx`: the model itself, its graph and full-precision (fp32)
  weights, in the portable ONNX format. This is what is committed. The CPU
  path below runs it directly.
- `model.plan` (built by `pytorch/build-engines.sh`, not committed): the
  TensorRT engine Triton serves, the same net compiled for your GPU in half
  precision (fp16), which is much faster on NVIDIA tensor cores and changes
  the outputs only by rounding (win probabilities by about 0.002).
- `config.pbtxt`: Triton's description of the model's inputs and outputs.

# Running the neural network without a GPU

Without Triton (`triton-use-triton` false), macondo runs the same model's
`model.onnx` on the CPU with [ONNX Runtime](https://github.com/microsoft/onnxruntime).
Slower than a GPU, but enough for `mleval` and the occasional game.

1. Download ONNX Runtime 1.29 for your platform from its
   [releases page](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0)
   (e.g. `onnxruntime-linux-x64-1.29.0.tgz`) and unpack it anywhere. On
   macOS, `brew install onnxruntime` also works if it is 1.29.
2. Point macondo at the library, in the config or the environment:
   `onnxruntime-lib: /path/to/onnxruntime-linux-x64-1.29.0/lib/libonnxruntime.so`
   (`MACONDO_ONNXRUNTIME_LIB`). Optional: `onnx-threads` (`MACONDO_ONNX_THREADS`),
   threads per evaluation; ONNX Runtime picks one per core by default.
3. Pick the model as above (`triton-model-name`); macondo loads
   `data/strategy/default/models/<name>/<version, default 1>/model.onnx`.

This needs a cgo build (the default when building from source on the
target machine).

### Attributions

Wolges-awsm is Copyright (C) 2020-2022 Andy Kurnia and released under the MIT license. It can be found at https://github.com/andy-k/wolges-awsm/. Macondo interfaces with it as a server.

KLV and KWG are Andy Kurnia's leave and word graph formats. They are small and fast! See more info at https://github.com/andy-k/wolges

Some of the code for the endgame solver was influenced by the MIT-licensed Chess solver Blunder. See code at https://github.com/algerbrex/blunder
