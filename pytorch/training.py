#!/usr/bin/env python3
"""
Streaming trainer + validation logger for the Scrabble CNN
----------------------------------------------------------
• stdin  : binary frames [len | 18 675 floats | 58 scalars | 1 target]
• output : best.pt (checkpoint)  &  loss_log.csv
"""

import io, struct, sys, time, csv
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# ── constants (sync with Go) ─────────────────────────────────────────
C, H, W = 83, 15, 15
N_PLANE = C * H * W  # 18 675
N_SCAL = 58
ROW_FLOATS = N_PLANE + N_SCAL + 1  # +1 target
DTYPE = np.float32

VAL_SIZE = 50_000  # hold-out vectors ≈ 25 batches @ 2048
VAL_EVERY = 500  # training steps between evals
CSV_PATH = "loss_log.csv"


# ── binary dataset on stdin ──────────────────────────────────────────
class StdinBinDataset(IterableDataset):
    def __iter__(self):
        buf = sys.stdin.buffer
        while True:
            hdr = buf.read(4)
            if not hdr:
                break
            (n_bytes,) = struct.unpack("<I", hdr)
            payload = buf.read(n_bytes)
            if len(payload) != n_bytes:
                break
            vec = np.frombuffer(payload, dtype=DTYPE, count=ROW_FLOATS)
            board = torch.from_numpy(vec[:N_PLANE]).view(C, H, W)
            scalars = torch.from_numpy(vec[N_PLANE : N_PLANE + N_SCAL])
            target = torch.tensor(vec[-1], dtype=torch.float32)
            yield board, scalars, target


# ── model (unchanged) ───────────────────────────────────────────────
class ResidBlock(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch)
        self.c2, self.b2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + x)


class ScrabbleValueNet(nn.Module):
    def __init__(self, planes=C, scalars=N_SCAL, ch=64, blocks=6):
        super().__init__()
        self.in_conv = nn.Conv2d(planes, ch, 3, 1, 1, bias=False)
        self.in_bn = nn.BatchNorm2d(ch)
        self.res = nn.Sequential(*[ResidBlock(ch) for _ in range(blocks)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(ch + scalars, 128)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, board, scalars):
        x = F.relu(self.in_bn(self.in_conv(board)))
        x = self.res(x)
        x = self.gap(x).flatten(1)
        x = torch.cat([x, scalars], 1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc_out(x)).squeeze(1)


# ── helper to evaluate on validation tensors ────────────────────────
@torch.no_grad()
def validate(net: nn.Module, val_tensors: List[torch.Tensor], device) -> float:
    boards, scalars, targets = val_tensors
    bs = 4096
    total, n = 0.0, 0
    for i in range(0, len(targets), bs):
        b = boards[i : i + bs].to(device)
        s = scalars[i : i + bs].to(device)
        y = targets[i : i + bs].to(device)
        with autocast(enabled=torch.cuda.is_available()):
            pred = net(b, s)
            total += F.smooth_l1_loss(pred, y, reduction="sum").item()
        n += y.numel()
    return total / n


# ── main training routine ───────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = StdinBinDataset()
    iter_ds = iter(ds)  # manual iterator to skim val
    print("collecting validation set …", file=sys.stderr)
    boards, scalars, targets = [], [], []
    for _ in range(VAL_SIZE):
        try:
            b, s, y = next(iter_ds)
        except StopIteration:
            break
        boards.append(b)
        scalars.append(s)
        targets.append(y)
    val_tensors = [torch.stack(boards), torch.stack(scalars), torch.stack(targets)]
    print(f"validation set size: {len(targets)}", file=sys.stderr)

    # DataLoader continues from current iterator position
    loader = DataLoader(iter_ds, batch_size=2048, num_workers=0, pin_memory=True)

    net = ScrabbleValueNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val, running, step = float("inf"), 0.0, 0
    t0 = time.time()
    csv_fh = open(CSV_PATH, "a", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(["step", "train_loss", "val_loss"])

    for board, scal, y in loader:
        board, scal, y = board.to(device), scal.to(device), y.to(device)

        with autocast(enabled=torch.cuda.is_available()):
            pred = net(board, scal)
            loss = F.smooth_l1_loss(pred, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        running += loss.item()
        step += 1

        # ── periodic validation ────────────────────────────────────
        if step % VAL_EVERY == 0:
            train_avg = running / VAL_EVERY
            running = 0.0
            val_loss = validate(net, val_tensors, device)
            csv_writer.writerow([step, f"{train_avg:.6f}", f"{val_loss:.6f}"])
            csv_fh.flush()

            elapsed = time.time() - t0
            print(
                f"{step:>7}  train={train_avg:.4f}  val={val_loss:.4f}  "
                f"{step*loader.batch_size/elapsed:,.0f} pos/s"
            )

            if val_loss < best_val:
                torch.save({"step": step, "model": net.state_dict()}, "best.pt")
                best_val = val_loss
                print("  ✓ checkpointed (best validation)")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
