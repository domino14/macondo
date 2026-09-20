#!/usr/bin/env python3
"""
Streaming trainer for Macondo's value nets (CNN and transformer)
----------------------------------------------------------------
stdin  : binary frames  [len | 19 125 board | 72 scalars | 5 targets]
output : best.pt  +  loss_log.csv  (train & val loss, per head)

Targets, in frame order (see cmd/mlproducer/game_assembler.go):
    value     bogowin after the horizon, in [-1, 1]      (smooth-L1, tanh)
    spread    spread change over the horizon, tanh-scaled (smooth-L1, tanh)
    wdl       final game result for the mover: -1, 0, 1  (cross-entropy)
    opp_bingo opponent bingos next turn: 0 or 1          (BCE with logits)
    opp_score opponent's next score / 300                (smooth-L1)

The bot ranks moves on `value`; the rest are auxiliary heads that
regularize the trunk (KataGo §4.1). The checkpoint is chosen on the value
head's validation loss alone, so runs with different head weights stay
comparable.
"""

import argparse, struct, sys, time, csv, os, signal, tempfile
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
N_PLANE = C * H * W  # 19_125
N_SCAL = 72
TARGETS = ["value", "spread", "wdl", "opp_bingo", "opp_score"]
N_TARGETS = len(TARGETS)
ROW_FLOATS = N_PLANE + N_SCAL + N_TARGETS
DTYPE = np.float32

VAL_SIZE = 150_000  # vectors for validation  (~75 batches)
VAL_EVERY = 500  # train steps between val checks
CSV_PATH = "loss_log.csv"

DEFAULT_WEIGHTS = {
    "value": 1.0,
    "spread": 0.5,
    "wdl": 0.25,
    "opp_bingo": 0.1,
    "opp_score": 0.1,
}


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


def unpack_frame(payload):
    """One frame -> (board (C,H,W), scalars (N_SCAL,), targets (N_TARGETS,))."""
    vec = np.frombuffer(payload, dtype=DTYPE, count=ROW_FLOATS).copy()
    board = torch.from_numpy(vec[:N_PLANE]).view(C, H, W)
    scalars = torch.from_numpy(vec[N_PLANE : N_PLANE + N_SCAL])
    targets = torch.from_numpy(vec[N_PLANE + N_SCAL :])
    return board, scalars, targets


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

            yield unpack_frame(payload)
            del payload


# ────────────────────────────────────────────────────────────────────
class Heads(nn.Module):
    """Output heads shared by every trunk. Takes the 128-wide hidden vector."""

    def __init__(self, hidden=128):
        super().__init__()
        self.value = nn.Linear(hidden, 1)
        self.spread = nn.Linear(hidden, 1)
        self.wdl = nn.Linear(hidden, 3)  # loss / draw / win logits
        self.opp_bingo = nn.Linear(hidden, 1)  # logit
        self.opp_score = nn.Linear(hidden, 1)

    def forward(self, h):
        return {
            "value": torch.tanh(self.value(h)).squeeze(1),
            "spread": torch.tanh(self.spread(h)).squeeze(1),
            "wdl": self.wdl(h),
            "opp_bingo": self.opp_bingo(h).squeeze(1),
            "opp_score": self.opp_score(h).squeeze(1),
        }


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
        self.heads = Heads(128)

    def forward(self, board, scalars):
        x = F.relu(self.in_bn(self.in_conv(board)))
        x = self.res(x)
        x = self.gap(x).flatten(1)
        x = torch.cat([x, scalars], 1)
        x = F.relu(self.fc1(x))
        return self.heads(x)


def build_model(arch, hparams):
    """Build a model from an arch name and its hyperparameter dict.

    `hparams` is what gets stored in the checkpoint so export.py can rebuild
    the exact architecture without hardcoding anything.
    """
    if arch == "cnn":
        return ScrabbleValueNet(ch=hparams["ch"], blocks=hparams["blocks"])
    if arch == "transformer":
        from transformer_model import ScrabbleTransformerNet  # lazy: no import cycle

        return ScrabbleTransformerNet(**hparams)
    raise ValueError(f"unknown arch {arch!r}")


def load_state_dict_compat(net, state):
    """Load a checkpoint saved before the Heads module existed.

    Old checkpoints have `value_head.*` (and the never-trained
    `total_points_head`, `opp_bingo_prob_head`, `opp_score_head`); the value
    head is remapped and the rest are dropped, leaving the new heads at
    their random init.
    """
    remapped = {}
    for k, v in state.items():
        if k.startswith("value_head."):
            k = "heads.value." + k[len("value_head.") :]
        elif k.split(".")[0] in (
            "total_points_head",
            "opp_bingo_prob_head",
            "opp_score_head",
        ):
            continue
        remapped[k] = v
    missing, unexpected = net.load_state_dict(remapped, strict=False)
    if unexpected:
        raise KeyError(f"unexpected keys in checkpoint: {unexpected}")
    return missing


ARCH_DEFAULTS = {
    "cnn": dict(batch_size=2048, accum=1, lr=1.0e-3, grad_clip=0.0, amp_dtype="fp16"),
    "transformer": dict(
        batch_size=256, accum=8, lr=3.0e-4, grad_clip=1.0, amp_dtype="bf16"
    ),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Streaming trainer for Macondo value nets")
    p.add_argument("--arch", choices=list(ARCH_DEFAULTS), default="cnn")
    # cnn
    p.add_argument("--ch", type=int, default=96)
    p.add_argument("--blocks", type=int, default=10)
    # transformer
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    # loss weights per head; 0 disables a head's gradient (it is still logged)
    for name, w in DEFAULT_WEIGHTS.items():
        p.add_argument(f"--w-{name.replace('_', '-')}", type=float, default=w)
    # optimisation; None means "use the per-arch default"
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--accum", type=int, default=None, help="gradient accumulation steps"
    )
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None, help="0 disables")
    p.add_argument("--amp-dtype", choices=["fp16", "bf16"], default=None)
    p.add_argument("--warmup", type=int, default=2_000)
    p.add_argument(
        "--total-steps",
        type=int,
        default=250_000,
        help="length of the cosine schedule; training stops here too",
    )
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # io
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--csv", default=CSV_PATH)
    p.add_argument("--val-size", type=int, default=VAL_SIZE)
    p.add_argument("--val-every", type=int, default=VAL_EVERY)
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="also save the current model every N steps as <ckpt>-stepN.pt",
    )
    args = p.parse_args(argv)

    for k, v in ARCH_DEFAULTS[args.arch].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    if args.arch == "cnn":
        args.hparams = dict(ch=args.ch, blocks=args.blocks)
    else:
        args.hparams = dict(
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            ff_mult=args.ff_mult,
            dropout=args.dropout,
        )
    args.weights = {name: getattr(args, f"w_{name}") for name in TARGETS}
    return args


# ────────────────────────────────────────────────────────────────────
def compute_loss(pred, targets, weights):
    """Per-head losses and their weighted sum.

    targets: (B, N_TARGETS) in frame order. Returns (total, {head: loss})
    where the per-head losses are detached tensors for accumulation.
    """
    losses = {
        "value": F.smooth_l1_loss(pred["value"], targets[:, 0]),
        "spread": F.smooth_l1_loss(pred["spread"], targets[:, 1]),
        # -1/0/1 -> class 0/1/2
        "wdl": F.cross_entropy(pred["wdl"], (targets[:, 2].round() + 1).long()),
        "opp_bingo": F.binary_cross_entropy_with_logits(
            pred["opp_bingo"], targets[:, 3]
        ),
        "opp_score": F.smooth_l1_loss(pred["opp_score"], targets[:, 4]),
    }
    total = sum(weights[k] * losses[k] for k in TARGETS if weights[k] > 0)
    return total, {k: v.detach() for k, v in losses.items()}


def write_validation_to_file(val_ds, directory):
    # ~77 KB per position, so 150k positions is 11.5 GB: keep it on real
    # disk next to the checkpoint, not in a tmpfs /tmp.
    val_file = tempfile.NamedTemporaryFile(
        prefix="val-", suffix=".bin", dir=directory, delete=False
    )
    val_count = 0
    for b, s, t in val_ds:
        val_file.write(b.numpy().astype(DTYPE).tobytes())
        val_file.write(s.numpy().astype(DTYPE).tobytes())
        val_file.write(t.numpy().astype(DTYPE).tobytes())
        val_count += 1
    val_file.close()
    return val_file.name, val_count


@torch.no_grad()
def validate_streaming(
    net, val_filename, val_count, device, weights, batch=1024, amp_dtype=torch.float16
):
    """Mean loss per head (and weighted total) over the validation file."""
    net.eval()
    sums = {k: torch.zeros((), device=device) for k in ["total", *TARGETS]}
    n = 0
    row_bytes = ROW_FLOATS * 4
    with open(val_filename, "rb") as f:
        while n < val_count:
            k = min(batch, val_count - n)
            raw = np.frombuffer(f.read(row_bytes * k), dtype=DTYPE).reshape(
                k, ROW_FLOATS
            )
            rows = torch.from_numpy(raw.copy()).to(device)
            b = rows[:, :N_PLANE].view(k, C, H, W)
            s = rows[:, N_PLANE : N_PLANE + N_SCAL]
            t = rows[:, N_PLANE + N_SCAL :]
            with autocast(
                device.type, dtype=amp_dtype, enabled=(device.type in ("cuda", "mps"))
            ):
                pred = net(b, s)
                total, losses = compute_loss(pred, t, weights)
            sums["total"] += total.detach() * k
            for name, l in losses.items():
                sums[name] += l * k
            n += k
    net.train()
    return {name: (v / n).item() for name, v in sums.items()}


# ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress lines survive a pipe/log
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    print(f"config: {vars(args)}", file=sys.stderr)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    val_q = Queue()
    train_q = Queue(maxsize=2048)

    num_workers = os.cpu_count()
    p = Thread(target=producer, args=(val_q, train_q, args.val_size, num_workers))
    p.daemon = True
    p.start()

    # ---- collect validation set -----------------------------------
    val_ds = QueueDataset(val_q)
    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    val_file_name, val_count = write_validation_to_file(val_ds, ckpt_dir)
    # Delete the validation file if we are killed, not only on a clean exit.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    print(f"Validation set: {val_count} positions", file=sys.stderr)

    # ---- training loader ------------------------------------------
    train_ds = QueueDataset(train_q)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )

    net = build_model(args.arch, args.hparams).to(device)
    print(
        f"{args.arch} params: {sum(p.numel() for p in net.parameters()):,}",
        file=sys.stderr,
    )
    # –– Optimiser -----------------------------------------------------------
    base_lr = args.lr  # peak LR after warm-up
    warm_up = args.warmup  # #steps spent warming up
    t_total = args.total_steps  # #scheduler steps before it restarts at 0

    opt = torch.optim.AdamW(
        net.parameters(), lr=base_lr, weight_decay=args.weight_decay
    )

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

    # bf16 has fp32's exponent range, so loss scaling is unnecessary; with
    # enabled=False every scaler call below is a no-op passthrough.
    scaler = GradScaler(
        enabled=(device.type in ("cuda", "mps")) and amp_dtype == torch.float16
    )

    def zero_running():
        return {k: torch.zeros((), device=device) for k in ["total", *TARGETS]}

    best_val, step, micro, t0 = float("inf"), 0, 0, time.time()
    running = zero_running()
    csv_fh = open(args.csv, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(
        ["step", "train_loss", "val_loss"]
        + [f"train_{k}" for k in TARGETS]
        + [f"val_{k}" for k in TARGETS]
    )

    try:
        for board, scal, targets in loader:
            board, scal = board.to(device), scal.to(device)
            targets = targets.to(device)

            with autocast(
                device.type, dtype=amp_dtype, enabled=(device.type in ("cuda", "mps"))
            ):
                pred = net(board, scal)
                loss, losses = compute_loss(pred, targets, args.weights)

            # Track all loss components (per micro-batch), without syncing
            running["total"] += loss.detach()
            for k, l in losses.items():
                running[k] += l

            scaler.scale(loss / args.accum if args.accum > 1 else loss).backward()
            micro += 1
            if micro % args.accum:
                continue  # accumulate; not yet an optimiser step
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % args.val_every == 0:
                n_micro = args.val_every * args.accum
                train = {k: (v / n_micro).item() for k, v in running.items()}
                running = zero_running()
                val = validate_streaming(
                    net,
                    val_file_name,
                    val_count,
                    device,
                    args.weights,
                    amp_dtype=amp_dtype,
                )
                if step == args.val_every and device.type == "cuda":
                    print(
                        f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                        file=sys.stderr,
                    )
                csv_writer.writerow(
                    [step, f"{train['total']:.6f}", f"{val['total']:.6f}"]
                    + [f"{train[k]:.6f}" for k in TARGETS]
                    + [f"{val[k]:.6f}" for k in TARGETS]
                )
                csv_fh.flush()

                elapsed = time.time() - t0
                heads = "  ".join(f"{k}={val[k]:.4f}" for k in TARGETS)
                print(
                    f"{step:>7}  train={train['total']:.4f}  val={val['total']:.4f}  "
                    f"[{heads}]  "
                    f"{step*args.batch_size*args.accum/elapsed:,.0f} pos/s"
                )

                # Checkpoint on the value head alone: it is what the bot
                # ranks on, and it keeps runs with different head weights
                # comparable.
                if val["value"] < best_val:
                    torch.save(
                        {
                            "step": step,
                            "model": net.state_dict(),
                            "arch": args.arch,
                            "hparams": args.hparams,
                            "weights": args.weights,
                            "val": val,
                        },
                        args.ckpt,
                    )
                    best_val = val["value"]
                    print("  ✓ checkpointed (best validation value loss)")

                if args.snapshot_every and step % args.snapshot_every == 0:
                    stem, ext = os.path.splitext(args.ckpt)
                    torch.save(
                        {
                            "step": step,
                            "model": net.state_dict(),
                            "arch": args.arch,
                            "hparams": args.hparams,
                            "weights": args.weights,
                            "val": val,
                        },
                        f"{stem}-step{step}{ext}",
                    )

            if step >= args.total_steps:
                # The cosine schedule is at zero; CosineAnnealingLR would
                # climb back up from here, so this is the end of the run.
                print(f"reached --total-steps {args.total_steps}; stopping")
                break

        # Print total training time
        total_time = time.time() - t0
        print(
            f"Total training time: {total_time:.1f} seconds ({total_time/60:.2f} min)"
        )

    finally:
        csv_fh.close()
        os.unlink(val_file_name)
        sys.stdout.flush()
        # Leave without waiting on the loader workers or the stdin producer
        # thread, which may be blocked mid-stream if we stopped early.
        os._exit(0)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
