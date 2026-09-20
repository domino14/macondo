"""
Parity check: PyTorch checkpoint vs exported ONNX model on real frames.

    python export-tester.py --ckpt best-tf.pt --onnx macondo-nn-tf.onnx \
        --frames calibrate.bin --batch-sizes 1,7,50,128 [--engine macondo-nn-tf.engine]

Frames are in mlproducer format: [uint32 len | payload], payload being
float32 planes + scalars (+ optional targets). Exits non-zero if the max
absolute difference exceeds --tol.
"""

import argparse
import struct
import sys

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
            (n_bytes,) = struct.unpack("<I", hdr)
            payload = f.read(n_bytes)
            if len(payload) != n_bytes:
                break
            vec = np.frombuffer(payload, dtype=np.float32)
            boards.append(vec[:N_PLANE].reshape(C, H, W))
            scalars.append(vec[N_PLANE : N_PLANE + N_SCAL])
    return np.stack(boards), np.stack(scalars)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="best-bak.pt")
    p.add_argument("--onnx", default="macondo-nn.onnx")
    p.add_argument("--frames", default="calibrate.bin")
    p.add_argument("--batch-sizes", default="1,7,50,128")
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument(
        "--engine", help="TensorRT .plan/.engine to compare too (fp16 tol 2e-2)"
    )
    p.add_argument("--engine-tol", type=float, default=2e-2)
    args = p.parse_args()

    sizes = [int(s) for s in args.batch_sizes.split(",")]
    boards, scalars = read_frames(args.frames, max(sizes))
    print(f"read {len(boards)} frames from {args.frames}")

    net, arch, hparams = load_net(args.ckpt)
    print(f"torch model: arch={arch} hparams={hparams}")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    print("onnx inputs:", [i.name for i in sess.get_inputs()])
    print("onnx outputs:", [o.name for o in sess.get_outputs()])

    trt_runner = None
    if args.engine:
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        trt_runner = TrtRunner(EngineFromBytes(open(args.engine, "rb").read()))
        trt_runner.activate()

    outputs = [o.name for o in sess.get_outputs()]
    worst, worst_trt = 0.0, 0.0
    for bs in sizes:
        b = boards[:bs]
        s = scalars[:bs]
        with torch.no_grad():
            refs = net(torch.from_numpy(b), torch.from_numpy(s))
        outs = sess.run(outputs, {"board": b, "scalars": s})
        touts = trt_runner.infer({"board": b, "scalars": s}) if trt_runner else None
        for name, out in zip(outputs, outs):
            ref = refs[name].numpy()
            assert out.shape == ref.shape, (name, out.shape, ref.shape)
            diff = float(np.abs(out - ref).max())
            worst = max(worst, diff)
            line = (
                f"batch {bs:4d} {name:>7}: max|onnx-torch|={diff:.2e}  "
                f"torch[0]={ref[0]:+.5f} onnx[0]={out[0]:+.5f}"
            )
            if touts is not None:
                tout = touts[name]
                tdiff = float(np.abs(tout.reshape(ref.shape) - ref).max())
                worst_trt = max(worst_trt, tdiff)
                line += f"  max|trt-torch|={tdiff:.2e} trt[0]={float(tout.flat[0]):+.5f}"
            print(line)

    if trt_runner is not None:
        trt_runner.deactivate()

    ok = True
    if worst > args.tol:
        print(f"FAIL: onnx max diff {worst:.2e} > tol {args.tol:.0e}")
        ok = False
    else:
        print(f"OK: onnx max diff {worst:.2e} <= tol {args.tol:.0e}")
    if trt_runner is not None:
        if worst_trt > args.engine_tol:
            print(f"FAIL: engine max diff {worst_trt:.2e} > tol {args.engine_tol:.0e}")
            ok = False
        else:
            print(f"OK: engine max diff {worst_trt:.2e} <= tol {args.engine_tol:.0e}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
