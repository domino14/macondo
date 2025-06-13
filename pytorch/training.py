#!/usr/bin/env python3
"""
Streaming trainer for César's Scrabble CNN
-----------------------------------------
stdin  : binary frames  [len | 18 675 board | 58 scalars | 1 target]
output : best.pt  +  loss_log.csv  (train & val loss)
"""

import io, struct, sys, time, csv, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import GradScaler, autocast

# ── feature sizes ────────────────────────────────────────────────────
C, H, W = 83, 15, 15
N_PLANE = C * H * W  # 18 675
N_SCAL = 56
ROW_FLOATS = N_PLANE + N_SCAL + 1
DTYPE = np.float32

VAL_SIZE = 50_000  # vectors for validation  (~25 batches)
VAL_EVERY = 500  # train steps between val checks
CSV_PATH = "loss_log.csv"


# ────────────────────────────────────────────────────────────────────
class StdinBinDataset(IterableDataset):
    """Infinite stream of binary records from stdin."""

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


# ────────────────────────────────────────────────────────────────────
class SkipIterable(IterableDataset):
    """Wrap another iterable but skip the first `n_skip` records each epoch."""

    def __init__(self, base_iterable, n_skip):
        super().__init__()
        self.base = base_iterable
        self.n_skip = n_skip

    def __iter__(self):
        return itertools.islice(self.base.__iter__(), self.n_skip, None)


# ────────────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(net, tensors, device, batch=4096):
    boards, scalars, targets = tensors
    total, n = 0.0, 0
    for i in range(0, len(targets), batch):
        b = boards[i : i + batch].to(device)
        s = scalars[i : i + batch].to(device)
        y = targets[i : i + batch].to(device)
        with autocast(device.type, enabled=(device.type in ("cuda", "mps"))):
            pred = net(b, s)
            total += F.smooth_l1_loss(pred, y, reduction="sum").item()
        n += y.numel()
    return total / n


# ────────────────────────────────────────────────────────────────────
def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    stream = StdinBinDataset()

    # ---- collect validation set -----------------------------------
    val_boards, val_scals, val_targets = [], [], []
    it = stream.__iter__()
    for _ in range(VAL_SIZE):
        try:
            b, s, y = next(it)
        except StopIteration:
            break
        val_boards.append(b)
        val_scals.append(s)
        val_targets.append(y)
    val_tensors = [
        torch.stack(val_boards),
        torch.stack(val_scals),
        torch.stack(val_targets),
    ]
    print(f"Validation set: {len(val_targets)} positions", file=sys.stderr)

    # ---- training loader (skip validation slice) ------------------
    train_ds = SkipIterable(stream, n_skip=VAL_SIZE)
    loader = DataLoader(train_ds, batch_size=2048, num_workers=0, pin_memory=True)

    net = ScrabbleValueNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type in ("cuda", "mps")))

    best_val, running, step, t0 = float("inf"), 0.0, 0, time.time()
    csv_fh = open(CSV_PATH, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(["step", "train_loss", "val_loss"])

    for board, scal, y in loader:
        board, scal, y = board.to(device), scal.to(device), y.to(device)

        with autocast(device.type, enabled=(device.type in ("cuda", "mps"))):
            pred = net(board, scal)
            loss = F.smooth_l1_loss(pred, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        running += loss.item()
        step += 1

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
