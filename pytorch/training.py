#!/usr/bin/env python3
"""
Training pipeline for César's Scrabble CNN
-----------------------------------------
• stdin  : rows of float32  (space-separated, newline-terminated)
           [ 18 675 board floats | 56 scalars | 1 target ]
• output : checkpoints best.pt and training log
"""

import io, os, sys, struct, time
from typing import Iterator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS  – keep these in sync with Go producer
# ─────────────────────────────────────────────────────────────────────────────
C = 83  # planes
H = W = 15
N_PLANE = C * H * W  # 18 675
N_SCAL = 56  # 27 rack + 27 unseen + tilesRem + spread
ROW_LEN = N_PLANE + N_SCAL + 1  # +1 target
DTYPE = np.float32


# ─────────────────────────────────────────────────────────────────────────────
# 2.  STREAMING DATASET
# ─────────────────────────────────────────────────────────────────────────────
class StdinFloatStream(IterableDataset):
    """Reads rows of ASCII floats from sys.stdin."""

    def __iter__(self):
        stdin = io.BufferedReader(sys.stdin.buffer, buffer_size=1 << 20)
        for line in stdin:
            # Slightly faster than str.split  ➜ 1.3 M rec/s on Ryzen 9
            arr = np.fromstring(line, dtype=DTYPE, sep=" ")
            if arr.size != ROW_LEN:
                continue  # skip malformed
            board = torch.from_numpy(arr[:N_PLANE]).view(C, H, W)
            scal = torch.from_numpy(arr[N_PLANE : N_PLANE + N_SCAL])
            target = torch.tensor(arr[-1], dtype=torch.float32)
            yield board, scal, target


class StdinBinDataset(IterableDataset):
    def __iter__(self):
        buf = sys.stdin.buffer  # already a BufferedReader
        while True:
            len_hdr = buf.read(4)
            if not len_hdr:
                break
            (n_bytes,) = struct.unpack("<I", len_hdr)
            payload = buf.read(n_bytes)
            if len(payload) != n_bytes:
                break
            vec = np.frombuffer(payload, dtype=np.float32)
            board = torch.from_numpy(vec[:N_PLANE]).view(C, H, W)
            scal = torch.from_numpy(vec[N_PLANE : N_PLANE + N_SCAL])
            y = torch.tensor(vec[-1])
            yield board, scal, y


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODEL  –  6-block, 64-channel ResNet
# ─────────────────────────────────────────────────────────────────────────────
class ResidBlock(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + x)


class ScrabbleValueNet(nn.Module):
    def __init__(self, planes=C, scalars=N_SCAL, ch=64, blocks=6):
        super().__init__()
        self.in_conv = nn.Conv2d(planes, ch, 3, padding=1, bias=False)
        self.in_bn = nn.BatchNorm2d(ch)
        self.res = nn.Sequential(*[ResidBlock(ch) for _ in range(blocks)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(ch + scalars, 128)
        self.fc_out = nn.Linear(128, 1)  # Δ-spread_k

    def forward(self, board, scalars):
        x = F.relu(self.in_bn(self.in_conv(board)))  # (B, ch, 15, 15)
        x = self.res(x)
        x = self.gap(x).flatten(1)  # (B, ch)
        x = torch.cat([x, scalars], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc_out(x).squeeze(1)  # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("CUDA available:", torch.cuda.is_available())
    print("PyTorch version:", torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = StdinBinDataset()
    loader = DataLoader(ds, batch_size=2048, num_workers=0, pin_memory=True)

    net = ScrabbleValueNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    running, step, best_loss = 0.0, 0, float("inf")
    t0 = time.time()

    for board, scal, y in loader:  # endless stream
        if torch.any(y.abs() > 1.0001):
            bad = y[y.abs() > 1].tolist()[:10]
            raise ValueError(f"Detected un-scaled targets: {bad[:3]} …")
        board = board.to(device, non_blocking=True)
        scal = scal.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if step == 0:  # or any small step
            print("target  min/max:", y.min().item(), y.max().item())
            print("first 5 targets:", y[:5].tolist())
            print("avg |target|:", y.abs().mean().item())
            sys.stdout.flush()

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            pred = net(board, scal)
            loss = F.smooth_l1_loss(pred, y)  # Huber

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        running += loss.item()
        step += 1

        # Every 100 batches print & checkpoint
        if step % 100 == 0:
            avg = running / 100
            running = 0.0
            elapsed = time.time() - t0
            print(
                f"{step:>7}  loss={avg:.4f}  {step*loader.batch_size/elapsed:,.0f} pos/s"
            )

            if avg < best_loss:
                torch.save({"step": step, "model": net.state_dict()}, "best.pt")
                best_loss = avg
                print("  ✓ checkpointed (best so far)")

    print(f"Training complete after {step} steps.")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Total time: {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # autotune kernels
    main()
