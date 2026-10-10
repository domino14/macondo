"""
Per-head trunk gradient norms for an existing checkpoint.

    python head_grads.py --ckpt best-tf-heads.pt --frames val-slice.bin [--n 512]

Reports how hard each head pulls on the trunk, unweighted, and the ratio
to the value head. Multiply a ratio by that head's --w-<head> to get its
actual share of the trunk gradient in training.
"""
import argparse, struct

import numpy as np
import torch

from training import (
    C, H, W, N_PLANE, N_SCAL, N_TARGETS, TARGETS, DEFAULT_WEIGHTS,
    build_model, load_state_dict_compat, head_grad_norms,
)


def read_frames(path, n):
    rows = []
    with open(path, "rb") as f:
        while len(rows) < n:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            (n_bytes,) = struct.unpack("<I", hdr)
            payload = f.read(n_bytes)
            vec = np.frombuffer(payload, dtype=np.float32)
            if len(vec) != N_PLANE + N_SCAL + N_TARGETS:
                raise SystemExit(f"frame has {len(vec)} floats; need {N_TARGETS} targets")
            rows.append(vec)
    rows = torch.from_numpy(np.stack(rows))
    return (
        rows[:, :N_PLANE].view(-1, C, H, W),
        rows[:, N_PLANE : N_PLANE + N_SCAL],
        rows[:, N_PLANE + N_SCAL :],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--frames", required=True, help="mlproducer output with 5 targets")
    p.add_argument("--n", type=int, default=512)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    net = build_model(ckpt.get("arch", "cnn"), ckpt.get("hparams", {"ch": 96, "blocks": 10}))
    load_state_dict_compat(net, ckpt["model"])
    weights = ckpt.get("weights", DEFAULT_WEIGHTS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    board, scalars, targets = (t.to(device) for t in read_frames(args.frames, args.n))
    gn = head_grad_norms(net, board, scalars, targets)
    ref = gn["value"]
    print(f"{args.ckpt}: step {ckpt.get('step')}, {board.shape[0]} positions")
    print(f"{'head':>10} {'grad norm':>10} {'vs value':>9} {'weight':>7} {'weighted share':>15}")
    for k in TARGETS:
        w = weights.get(k, 0.0)
        print(f"{k:>10} {gn[k]:10.4g} {gn[k]/ref:9.2f} {w:7.2f} {w*gn[k]/ref:15.2f}")


if __name__ == "__main__":
    main()
