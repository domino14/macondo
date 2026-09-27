#!/usr/bin/env python3
"""Dump a checkpoint's weights as one raw little-endian float32 blob plus a
JSON manifest, and a parity fixture (real input frames with the ONNX
outputs), for a from-scratch reimplementation (Metal, C, ...).

    python export-raw.py --ckpt best-tf-nwl23s.pt --onnx model.onnx --out DIR

Writes DIR/weights.f32, DIR/manifest.json, DIR/parity/{board,scalars,value,spread}.f32
and DIR/parity/fixture.json. Frames come from --frames (the producer's
length-prefixed format; only the leading 19,125 + 72 floats are used).
"""
import argparse
import hashlib
import json
import os
import struct

import numpy as np
import onnxruntime as ort
import torch

from training import C, H, W, N_PLANE, N_SCAL
from export import load_net


def read_frames(path, n):
    boards, scalars = [], []
    with open(path, "rb") as f:
        while len(boards) < n:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            (nb,) = struct.unpack("<I", hdr)
            vec = np.frombuffer(f.read(nb), dtype=np.float32)
            boards.append(vec[:N_PLANE].reshape(C, H, W))
            scalars.append(vec[N_PLANE : N_PLANE + N_SCAL])
    return np.stack(boards), np.stack(scalars)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--frames", default="calibrate.bin")
    p.add_argument("--n-frames", type=int, default=64)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "parity"), exist_ok=True)

    net, arch, hparams = load_net(args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = net.state_dict()
    tensors = []
    offset = 0
    with open(os.path.join(args.out, "weights.f32"), "wb") as blob:
        for name, t in sd.items():
            a = t.detach().cpu().to(torch.float32).contiguous().numpy()
            blob.write(a.tobytes())
            tensors.append({"name": name, "shape": list(a.shape), "offset_floats": offset, "count": a.size})
            offset += a.size
    manifest = {
        "model": "macondo-nn-tf-nwl23s v1",
        "checkpoint": os.path.basename(args.ckpt),
        "checkpoint_step": ckpt.get("step"),
        "arch": arch,
        "hparams": hparams,
        "primary": ckpt.get("primary"),
        "dtype": "float32 little-endian, row-major (C order), one blob",
        "total_floats": offset,
        "inputs": {"board": [C, H, W], "scalars": [N_SCAL]},
        "served_outputs": {"value": "P(win) - P(loss) from softmax(heads.wdl) [loss, draw, win]",
                           "spread": "tanh(heads.spread)"},
        "tensors": tensors,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    # Parity fixture: real frames through the ONNX graph.
    boards, scalars = read_frames(args.frames, args.n_frames)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    value, spread = sess.run(["value", "spread"], {"board": boards, "scalars": scalars})
    pd = os.path.join(args.out, "parity")
    for nm, a in (("board", boards), ("scalars", scalars), ("value", value), ("spread", spread)):
        with open(os.path.join(pd, nm + ".f32"), "wb") as f:
            f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    # torch reference too, so a port can be checked against either
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(boards), torch.from_numpy(scalars))
        pw = torch.softmax(out["wdl"], 1)
        tv = (pw[:, 2] - pw[:, 0]).numpy()
    fixture = {
        "frames": int(boards.shape[0]),
        "board": {"file": "board.f32", "shape": [int(boards.shape[0]), C, H, W]},
        "scalars": {"file": "scalars.f32", "shape": [int(boards.shape[0]), N_SCAL]},
        "value": {"file": "value.f32", "shape": [int(boards.shape[0])], "source": "onnxruntime fp32"},
        "spread": {"file": "spread.f32", "shape": [int(boards.shape[0])], "source": "onnxruntime fp32"},
        "max_abs_diff_torch_vs_onnx_value": float(np.abs(tv - value).max()),
        "value_range": [float(value.min()), float(value.max())],
        "tolerance_note": "fp32 ports should match within ~1e-5; fp16 within ~2e-3 (the TensorRT engine is at 1.75e-3)",
    }
    with open(os.path.join(pd, "fixture.json"), "w") as f:
        json.dump(fixture, f, indent=1)
    md5 = {}
    for root, _, files in os.walk(args.out):
        for fn in files:
            path = os.path.join(root, fn)
            md5[os.path.relpath(path, args.out)] = hashlib.md5(open(path, "rb").read()).hexdigest()
    with open(os.path.join(args.out, "MD5SUMS"), "w") as f:
        for k in sorted(md5):
            f.write(f"{md5[k]}  {k}\n")
    print(f"{len(tensors)} tensors, {offset:,} floats; parity {boards.shape[0]} frames, "
          f"torch-vs-onnx max diff {fixture['max_abs_diff_torch_vs_onnx_value']:.2e}")


if __name__ == "__main__":
    main()
