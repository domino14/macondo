#!/usr/bin/env python
"""Does board energy add to a Bogowin-style win probability?

For held-out mid-game positions (frame caches: board after the mover's play,
mover's side, opponent to move; label = the mover's true game result), fit
P(win | unseen tiles, spread) as a table, then the same table split by the
board's Ising energy tercile (energy ranked among boards with the same
number of tiles, so it is shape, not tile count). Compare on the other half
of the positions: log loss, the spread slope of a logistic fit (flatter =
more variance), and calibration in the tails.

    python ising_bogowin.py --frames temp=val-stream.bin open=val-streamopen.bin --models ~/data/ising
"""
import argparse, os
import numpy as np
from ising_fit import BANDS, N, load_occupancy
from training import CACHE_ROW_BYTES, PACKED_BYTES, N_SCAL

UNSEEN = [(8, 14), (15, 25), (26, 40), (41, 55), (56, 70), (71, 93)]
SPREAD_EDGES = np.arange(-155, 160, 10)


def load_labels(path, n):
    sc = np.empty((n, 2), np.float32)
    y = np.empty(n, np.float32)
    with open(path, "rb") as f:
        for a in range(0, n, 20000):
            b = min(n, a + 20000)
            rows = np.frombuffer(f.read((b - a) * CACHE_ROW_BYTES), np.uint8).reshape(-1, CACHE_ROW_BYTES)
            s = rows[:, PACKED_BYTES:PACKED_BYTES + N_SCAL * 4].copy().view(np.float32)
            t = rows[:, PACKED_BYTES + N_SCAL * 4:].copy().view(np.float32)
            sc[a:b, 0] = s[:, 70] * 100                                    # unseen tiles
            sc[a:b, 1] = 130 * np.arctanh(np.clip(s[:, 71], -0.999999, 0.999999))  # spread after the move
            y[a:b] = (t[:, 2] + 1) / 2                                      # win 1, draw 0.5, loss 0
    return sc, y


def energies(occ, npz):
    tiles = occ.sum(1)
    E = np.full(len(occ), np.nan)
    for lo, hi in BANDS:
        key = f"{lo:02d}-{hi:02d}"
        if f"h_{key}" not in npz:
            continue
        idx = np.flatnonzero((tiles >= lo) & (tiles <= hi))
        h, W = npz[f"h_{key}"], npz[f"W_{key}"]
        for a in range(0, len(idx), 50000):
            j = idx[a:a + 50000]
            s = occ[j].astype(np.float64) * 2 - 1
            E[j] = -(s @ h + 0.5 * np.einsum("ij,ij->i", s @ W, s))
    # rank within boards of the same tile count -> tercile 0 (low) .. 2 (high)
    terc = np.full(len(occ), -1)
    for t in np.unique(tiles):
        j = np.flatnonzero((tiles == t) & ~np.isnan(E))
        if len(j) < 30:
            continue
        r = np.argsort(np.argsort(E[j])) / len(j)
        terc[j] = np.minimum(2, (r * 3).astype(int))
    return E, terc


def table(ub, sb, y, extra=None, k=None, alpha=1.0):
    """Smoothed win-rate table over (unseen bin, spread bin[, extra])."""
    nb, ns = len(UNSEEN), len(SPREAD_EDGES) + 1
    ne = k or 1
    key = (ub * ns + sb) * ne + (extra if extra is not None else 0)
    wins = np.bincount(key, weights=y, minlength=nb * ns * ne)
    cnt = np.bincount(key, minlength=nb * ns * ne)
    return wins, cnt, key


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logistic_slope(x, y, iters=30):
    a, b = 0.0, 0.0
    for _ in range(iters):
        z = a + b * x
        p = 1 / (1 + np.exp(-z))
        w = p * (1 - p) + 1e-9
        g = np.array([np.sum(y - p), np.sum((y - p) * x)])
        H = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
        a, b = np.array([a, b]) + np.linalg.solve(H, g)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--models", required=True)
    args = ap.parse_args()
    for spec in args.frames:
        name, path = spec.split("=", 1)
        occ = load_occupancy(path)
        sc, y = load_labels(path, len(occ))
        E, terc = energies(occ, np.load(os.path.join(os.path.expanduser(args.models), f"{name}.npz")))
        ub = np.full(len(y), -1)
        for i, (lo, hi) in enumerate(UNSEEN):
            ub[(sc[:, 0] >= lo - 0.5) & (sc[:, 0] <= hi + 0.5)] = i
        ok = (ub >= 0) & (terc >= 0)
        ub, sb, y2, t2, spread = ub[ok], np.digitize(sc[ok, 1], SPREAD_EDGES), y[ok], terc[ok], sc[ok, 1]
        rng = np.random.default_rng(7)
        tr = rng.random(len(y2)) < 0.5
        te = ~tr
        print(f"\n== {name}: {ok.sum():,} positions (unseen 8..93), half to fit the tables, half to score them")
        # A. held-out log loss
        w0, c0, k0 = table(ub, sb, y2)
        w0t, c0t = np.bincount(k0[tr], y2[tr], len(c0)), np.bincount(k0[tr], minlength=len(c0))
        p0 = (w0t[k0] + 1) / (c0t[k0] + 2)
        w1, c1, k1 = table(ub, sb, y2, t2, 3)
        w1t, c1t = np.bincount(k1[tr], y2[tr], len(c1)), np.bincount(k1[tr], minlength=len(c1))
        p1 = (w1t[k1] + 1) / (c1t[k1] + 2)
        l0, l1 = logloss(p0[te], y2[te]), logloss(p1[te], y2[te])
        # paired bootstrap SE of the difference
        d = (-(y2[te] * np.log(np.clip(p1[te], 1e-4, 1)) + (1 - y2[te]) * np.log(np.clip(1 - p1[te], 1e-4, 1)))
             + (y2[te] * np.log(np.clip(p0[te], 1e-4, 1)) + (1 - y2[te]) * np.log(np.clip(1 - p0[te], 1e-4, 1))))
        print(f"held-out log loss: spread+unseen {l0:.5f}   + energy tercile {l1:.5f}   "
              f"change {l1 - l0:+.5f} (SE {d.std() / np.sqrt(len(d)):.5f})")
        # B. spread slope by energy tercile (flatter = more variance)
        print("logistic slope of P(win) per 100 points of spread, by unseen tiles and energy tercile:")
        print("  unseen      low E   mid E   high E    n")
        for i, (lo, hi) in enumerate(UNSEEN):
            bs = []
            for t in range(3):
                m = (ub == i) & (t2 == t) & (np.abs(spread) < 150)
                bs.append(logistic_slope(spread[m] / 100, y2[m]))
            print(f"  {lo:2d}-{hi:2d}     {bs[0]:6.2f}  {bs[1]:6.2f}  {bs[2]:6.2f}   {((ub == i)).sum():,}")
        # C. tails: where the plain table says <15% or >85%, actual win rate by tercile
        print("tails (scored half): mean predicted vs actual win rate, by energy tercile")
        for nm, m in (("table < 15%", p0 < 0.15), ("table > 85%", p0 > 0.85)):
            row = []
            for t in range(3):
                mm = te & m & (t2 == t)
                row.append(f"{t}: pred {p0[mm].mean():.3f} actual {y2[mm].mean():.3f} (n {mm.sum():,})")
            print(f"  {nm}:  " + "   ".join(row))
        print("mean energy-rank effect: win rate by tercile within the same (unseen, spread) cell, average deviation from the table:")
        for t in range(3):
            mm = te & (t2 == t)
            print(f"  tercile {t}: actual - table = {np.mean(y2[mm] - p0[mm]):+.4f}  (n {mm.sum():,})")


if __name__ == "__main__":
    main()
