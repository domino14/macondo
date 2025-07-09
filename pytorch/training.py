#!/usr/bin/env python3
"""
Streaming trainer for César's Scrabble CNN
-----------------------------------------
stdin  : binary frames  [len | 18 675 board | 58 scalars | 1 target]
output : best.pt  +  loss_log.csv  (train & val loss)
"""

import io, struct, sys, time, csv, os, tempfile
from multiprocessing import Queue
from threading import Thread
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import GradScaler, autocast

# ── feature sizes ────────────────────────────────────────────────────
C, H, W = 85, 15, 15
N_PLANE = C * H * W  # 19_575
N_SCAL = 72
N_TARGETS = 4  # [win, points, bingo_prob, opp_score]
ROW_FLOATS = N_PLANE + N_SCAL + N_TARGETS
DTYPE = np.float32

VAL_SIZE = 150_000  # vectors for validation  (~75 batches)
VAL_EVERY = 500  # train steps between val checks
CSV_PATH = "loss_log.csv"


# ────────────────────────────────────────────────────────────────────
def producer(val_q, train_q, val_size, num_workers):
    """Read from stdin and push to validation and training queues."""

    def _shutdown_queues():
        """Send sentinel values to terminate all consumer processes."""
        val_q.put(None)
        for _ in range(num_workers):
            train_q.put(None)

    buf = sys.stdin.buffer
    # Validation data
    for _ in range(val_size):
        try:
            hdr = buf.read(4)
            if not hdr:
                _shutdown_queues()
                return
            (n_bytes,) = struct.unpack("<I", hdr)
            payload = buf.read(n_bytes)
            if len(payload) != n_bytes:
                _shutdown_queues()
                return
            val_q.put(payload)
        except (IOError, struct.error):
            _shutdown_queues()
            return
    val_q.put(None)  # Sentinel for validation queue

    # Training data
    while True:
        try:
            hdr = buf.read(4)
            if not hdr:
                break
            (n_bytes,) = struct.unpack("<I", hdr)
            payload = buf.read(n_bytes)
            if len(payload) != n_bytes:
                break
            train_q.put(payload)
        except (IOError, struct.error):
            break

    for _ in range(num_workers):
        train_q.put(None)


class QueueDataset(IterableDataset):
    """An iterable dataset that pulls from a multiprocessing queue."""

    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.worker_sentinel_received = False

    def __iter__(self):
        while True:
            if self.worker_sentinel_received:
                break

            payload = self.queue.get(timeout=60)  # Add timeout to avoid hanging
            if payload is None:
                self.worker_sentinel_received = True
                break

            vec = np.frombuffer(payload, dtype=DTYPE, count=ROW_FLOATS).copy()
            board = torch.from_numpy(vec[:N_PLANE]).view(C, H, W)
            scalars = torch.from_numpy(vec[N_PLANE : N_PLANE + N_SCAL])
            target_start = N_PLANE + N_SCAL
            # Extract multiple targets
            targets = {
                "value": torch.tensor(vec[target_start], dtype=torch.float32),
                "total_game_points": torch.tensor(
                    vec[target_start + 1], dtype=torch.float32
                ),
                "opp_bingo_prob": torch.tensor(
                    vec[target_start + 2], dtype=torch.float32
                ),
                "opp_score": torch.tensor(vec[target_start + 3], dtype=torch.float32),
            }

            yield board, scalars, targets
            del payload, vec


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

        # Original value (-1 to 1). The value of the move. (-1 = strong loss, 1 = strong win)
        self.value_head = nn.Linear(128, 1)
        self.total_points_head = nn.Linear(128, 1)  # Total points scored
        self.opp_bingo_prob_head = nn.Linear(128, 1)  # Bingo probability
        self.opp_score_head = nn.Linear(128, 1)  # Opponent score

    def forward(self, board, scalars):
        x = F.relu(self.in_bn(self.in_conv(board)))
        x = self.res(x)
        x = self.gap(x).flatten(1)
        x = torch.cat([x, scalars], 1)
        x = F.relu(self.fc1(x))
        value = torch.tanh(self.value_head(x)).squeeze(1)  # -1 to 1
        # total_game_points = self.total_points_head(x).squeeze(
        #     1
        # )  # No activation if pre-scaled
        # opp_bingo_prob = self.opp_bingo_prob_head(x).squeeze(1)
        # # Opponent score: use ReLU to ensure non-negative
        # opp_score = F.relu(self.opp_score_head(x)).squeeze(1)

        return {
            "value": value,
            # "total_game_points": total_game_points,
            # "opp_bingo_prob": opp_bingo_prob,
            # "opp_score": opp_score,
        }


# ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(net, tensors, device, batch=4096):
    boards, scalars, targets_dict = tensors
    total, n = 0.0, 0
    for i in range(0, len(list(targets_dict.values())[0]), batch):
        b = boards[i : i + batch].to(device)
        s = scalars[i : i + batch].to(device)

        # Move all targets to device
        batch_targets = {
            k: v[i : i + batch].to(device) for k, v in targets_dict.items()
        }

        with autocast(device.type, enabled=(device.type in ("cuda", "mps"))):
            pred = net(b, s)
            loss, _ = compute_loss(pred, batch_targets)
            total += loss.item() * b.size(0)
        n += b.size(0)
    return total / n


def write_validation_to_file(val_ds, C, H, W, N_SCAL, DTYPE):
    val_file = tempfile.NamedTemporaryFile(delete=False)
    val_count = 0
    for b, s, targets in val_ds:
        b_bytes = b.numpy().astype(DTYPE).tobytes()
        s_bytes = s.numpy().astype(DTYPE).tobytes()

        # Pack all targets into a single array
        targets_array = np.array(
            [
                targets["value"].numpy(),
                # targets["total_game_points"].numpy(),
                # targets["opp_bingo_prob"].numpy(),
                # targets["opp_score"].numpy(),
            ],
            dtype=DTYPE,
        )

        t_bytes = targets_array.tobytes()
        val_file.write(struct.pack("<III", len(b_bytes), len(s_bytes), len(t_bytes)))
        val_file.write(b_bytes)
        val_file.write(s_bytes)
        val_file.write(t_bytes)
        val_count += 1
    val_file.close()
    return val_file.name, val_count


@torch.no_grad()
def validate_streaming(net, val_filename, val_count, device, batch=1024):
    net.eval()
    total, n = 0.0, 0
    with open(val_filename, "rb") as f:
        batch_boards, batch_scalars, batch_targets = [], [], []
        for _ in range(val_count):
            b_size, s_size, t_size = struct.unpack("<III", f.read(12))
            b_bytes = f.read(b_size)
            s_bytes = f.read(s_size)
            t_bytes = f.read(t_size)
            b = torch.from_numpy(np.frombuffer(b_bytes, dtype=DTYPE).reshape(C, H, W))
            s = torch.from_numpy(np.frombuffer(s_bytes, dtype=DTYPE))
            t = np.frombuffer(t_bytes, dtype=DTYPE)

            # Unpack targets
            targets = {
                "value": torch.tensor(t[0], dtype=torch.float32),
                # "total_game_points": torch.tensor(t[1], dtype=torch.float32),
                # "opp_bingo_prob": torch.tensor(t[2], dtype=torch.float32),
                # "opp_score": torch.tensor(t[3], dtype=torch.float32),
            }

            batch_boards.append(b)
            batch_scalars.append(s)
            batch_targets.append(targets)

            if len(batch_boards) == batch:
                b_tensor = torch.stack(batch_boards).to(device)
                s_tensor = torch.stack(batch_scalars).to(device)

                # Stack targets into dict of tensors
                stacked_targets = {
                    k: torch.stack([d[k] for d in batch_targets]).to(device)
                    for k in batch_targets[0]
                }

                with autocast(device.type, enabled=(device.type in ("cuda", "mps"))):
                    pred = net(b_tensor, s_tensor)
                    loss, _ = compute_loss(pred, stacked_targets)
                    total += loss.item() * b_tensor.size(0)

                n += b_tensor.size(0)
                batch_boards, batch_scalars, batch_targets = [], [], []

        if batch_boards:
            b_tensor = torch.stack(batch_boards).to(device)
            s_tensor = torch.stack(batch_scalars).to(device)

            # Stack targets into dict of tensors
            stacked_targets = {
                k: torch.stack([d[k] for d in batch_targets]).to(device)
                for k in batch_targets[0]
            }

            with autocast(device.type, enabled=(device.type in ("cuda", "mps"))):
                pred = net(b_tensor, s_tensor)
                loss, _ = compute_loss(pred, stacked_targets)
                total += loss.item() * b_tensor.size(0)
                n += b_tensor.size(0)
    net.train()
    return total / n


def compute_loss(predictions, targets, target_weights=None):
    """
    Compute weighted loss across multiple prediction heads.

    Args:
        predictions: Dictionary of model predictions
        targets: Dictionary of target values
        target_weights: Optional dictionary of weights for each loss component

    Returns:
        total_loss: Combined loss value
        loss_dict: Dictionary of individual losses for logging
    """
    if target_weights is None:
        # Default weights
        target_weights = {
            "value": 1.0,
            # "total_game_points": 0.25,
            # "opp_bingo_prob": 0.5,
            # "opp_score": 0.25,
        }

    # Unpack predictions
    pred_value = predictions["value"]
    # pred_points = predictions["total_game_points"]
    # pred_bingo_prob = predictions["opp_bingo_prob"]
    # pred_opp_score = predictions["opp_score"]

    # Unpack targets
    target_value = targets["value"]
    # target_points = targets["total_game_points"]
    # target_bingo_prob = targets["opp_bingo_prob"]
    # target_opp_score = targets["opp_score"]

    # Calculate individual losses
    value_loss = F.smooth_l1_loss(pred_value, target_value)
    # points_loss = F.smooth_l1_loss(pred_points, target_points)

    # # For binary classification, use BCE loss
    # bingo_loss = F.binary_cross_entropy_with_logits(pred_bingo_prob, target_bingo_prob)

    # # For score prediction
    # opp_score_loss = F.smooth_l1_loss(pred_opp_score, target_opp_score)

    # Combine losses with weights
    total_loss = (
        target_weights["value"]
        * value_loss
        # + target_weights["total_game_points"] * points_loss
        # + target_weights["opp_bingo_prob"] * bingo_loss
        # + target_weights["opp_score"] * opp_score_loss
    )

    return total_loss, {
        "value_loss": value_loss.item(),
        # "points_loss": points_loss.item(),
        # "bingo_loss": bingo_loss.item(),
        # "opp_score_loss": opp_score_loss.item(),
        "total_loss": total_loss.item(),
    }


# ────────────────────────────────────────────────────────────────────
def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    val_q = Queue()
    train_q = Queue(maxsize=2048)

    num_workers = os.cpu_count()
    p = Thread(target=producer, args=(val_q, train_q, VAL_SIZE, num_workers))
    p.daemon = True
    p.start()

    # ---- collect validation set -----------------------------------
    val_ds = QueueDataset(val_q)
    val_file_name, val_count = write_validation_to_file(val_ds, C, H, W, N_SCAL, DTYPE)
    print(f"Validation set: {val_count} positions", file=sys.stderr)

    # ---- training loader ------------------------------------------
    train_ds = QueueDataset(train_q)
    loader = DataLoader(
        train_ds,
        batch_size=2048,
        num_workers=num_workers,
        pin_memory=False,
    )

    net = ScrabbleValueNet(ch=96, blocks=10).to(device)
    #–– Optimiser -----------------------------------------------------------
    base_lr   = 1.0e-3          # peak LR after warm-up
    warm_up   =   2_000         # #steps spent warming up
    t_total   = 250_000         # #scheduler steps before it restarts at 0

    opt = torch.optim.AdamW(net.parameters(), lr=base_lr, weight_decay=1e-4)

    # 1) linear warm-up from 0 → base_lr
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-3, end_factor=1.0, total_iters=warm_up
    )
    # 2) cosine decay down to 0
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=t_total - warm_up, eta_min=0.0
    )
    # 3) chain them: warm-up runs first, then cosine takes over
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[warm_up]
    )



    scaler = GradScaler(enabled=(device.type in ("cuda", "mps")))

    best_val, step, t0 = float("inf"), 0, time.time()
    running = {
        "total": 0.0,
        "value": 0.0,
        # "points": 0.0,
        # "bingo": 0.0,
        # "opp_score": 0.0,
    }
    csv_fh = open(CSV_PATH, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(
        [
            "step",
            "train_loss",
            "val_loss",
            "value_loss",
            # "points_loss",
            # "bingo_loss",
            # "opp_score_loss",
        ]
    )

    try:
        for board, scal, targets in loader:
            board, scal = board.to(device), scal.to(device)
            # Move all targets to device
            targets = {k: v.to(device) for k, v in targets.items()}

            with autocast(device.type, enabled=(device.type in ("cuda", "mps"))):
                pred = net(board, scal)
                loss, loss_dict = compute_loss(pred, targets)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

            # Track all loss components
            running["total"] += loss.item()
            running["value"] += loss_dict["value_loss"]
            # running["points"] += loss_dict["points_loss"]
            # running["bingo"] += loss_dict["bingo_loss"]
            # running["opp_score"] += loss_dict["opp_score_loss"]
            step += 1

            if step % VAL_EVERY == 0:
                train_avg = running["total"] / VAL_EVERY
                # Store current loss components
                val_loss_dict = {
                    "value_loss": running["value"] / VAL_EVERY,
                    # "points_loss": running["points"] / VAL_EVERY,
                    # "bingo_loss": running["bingo"] / VAL_EVERY,
                    # "opp_score_loss": running["opp_score"] / VAL_EVERY,
                }
                # Reset running losses
                running = {
                    "total": 0.0,
                    "value": 0.0,
                    # "points": 0.0,
                    # "bingo": 0.0,
                    # "opp_score": 0.0,
                }
                val_loss = validate_streaming(net, val_file_name, val_count, device)
                # Include loss components in CSV
                csv_writer.writerow(
                    [
                        step,
                        f"{train_avg:.6f}",
                        f"{val_loss:.6f}",
                        f"{val_loss_dict.get('value_loss', 0.0):.6f}",
                        # f"{val_loss_dict.get('points_loss', 0.0):.6f}",
                        # f"{val_loss_dict.get('bingo_loss', 0.0):.6f}",
                        # f"{val_loss_dict.get('opp_score_loss', 0.0):.6f}",
                    ]
                )
                csv_fh.flush()

                elapsed = time.time() - t0
                print(
                    f"{step:>7}  train={train_avg:.4f}  val={val_loss:.4f}  "
                    f"v_loss={val_loss_dict.get('value_loss', 0.0):.4f}  "
                    # f"p_loss={val_loss_dict.get('points_loss', 0.0):.4f}  "
                    # f"b_loss={val_loss_dict.get('bingo_loss', 0.0):.4f}  "
                    # f"o_loss={val_loss_dict.get('opp_score_loss', 0.0):.4f}  "
                    f"{step*loader.batch_size/elapsed:,.0f} pos/s"
                )

                if val_loss < best_val:
                    torch.save({"step": step, "model": net.state_dict()}, "best.pt")
                    best_val = val_loss
                    print("  ✓ checkpointed (best validation)")

        # Print total training time
        total_time = time.time() - t0
        print(
            f"Total training time: {total_time:.1f} seconds ({total_time/60:.2f} min)"
        )

    finally:
        csv_fh.close()
        os.unlink(val_file_name)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
