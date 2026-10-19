#!/usr/bin/env python
"""Null test for ising_bogowin: is the energy effect real signal, or would any
board statistic built the same way show it?

The real energy is compared with control energies computed the same way
(E = -(h.s + 0.5 s'Ws), ranked within boards of the same tile count) from
(a) the fitted parameters with the squares randomly permuted (same values,
geometry destroyed), and (b) random Gaussian fields and couplings of the
same scale, identical in every tile band (so smooth across the game).
Each control is scored with the same three statistics:
  tail gap   lower tail (table < 15%): win rate of high-energy minus low-
             energy tercile; upper tail (> 85%): low minus high. Positive =
             high-energy boards less extreme.
  slope      mean over bag bands of (slope high-E / slope low-E) - 1.
  logloss    held-out change from adding the five energy terms to the
             smooth logistic model (negative = helps).
"""
import os, sys
import numpy as np, torch
from ising_fit import BANDS, load_occupancy
import ising_bogowin as B

DRAWS = int(os.environ.get("DRAWS", 10))


def energy(occ, tiles, params):
    E = np.full(len(occ), np.nan)
    for lo, hi in BANDS:
        if (lo, hi) not in params:
            continue
        h, W = params[(lo, hi)]
        idx = np.flatnonzero((tiles >= lo) & (tiles <= hi))
        for a in range(0, len(idx), 100000):
            j = idx[a:a + 100000]
            s = occ[j].astype(np.float32) * 2 - 1
            E[j] = -(s @ h + 0.5 * np.einsum("ij,ij->i", s @ W, s))
    return E


def rank_stats(E, tiles):
    terc = np.full(len(E), -1); z = np.full(len(E), np.nan)
    for t in np.unique(tiles):
        j = np.flatnonzero((tiles == t) & ~np.isnan(E))
        if len(j) < 30:
            continue
        r = np.argsort(np.argsort(E[j])) / len(j)
        terc[j] = np.minimum(2, (r * 3).astype(int))
        z[j] = (E[j] - E[j].mean()) / (E[j].std() + 1e-9)
    return terc, z


def score(terc, z, ctx):
    ub, sb, y, spread, unseen, tr, p0, ok = ctx
    t2, z2 = terc[ok], np.clip(z[ok], -3, 3)
    te = ~tr
    lo = te & (p0 < 0.15); hi = te & (p0 > 0.85)
    gap_lo = y[lo & (t2 == 2)].mean() - y[lo & (t2 == 0)].mean()
    gap_hi = y[hi & (t2 == 0)].mean() - y[hi & (t2 == 2)].mean()
    ratios = []
    for i in range(len(B.UNSEEN)):
        m = (ub == i) & (np.abs(spread) < 150)
        bl = B.logistic_slope(spread[m & (t2 == 0)] / 100, y[m & (t2 == 0)])
        bh = B.logistic_slope(spread[m & (t2 == 2)] / 100, y[m & (t2 == 2)])
        ratios.append(bh / bl - 1)
    # smooth logistic with / without energy terms
    u = torch.tensor(unseen / 100, dtype=torch.float64); s = torch.tensor(spread / 100, dtype=torch.float64)
    zz = torch.tensor(z2, dtype=torch.float64); yy = torch.tensor(y, dtype=torch.float64); trt = torch.tensor(tr)
    base = [torch.ones_like(u), u, u * u, s, s * u, s * u * u, s * s * s]
    losses = []
    for X in (torch.stack(base, 1), torch.stack(base + [zz, s * zz, s * zz * u, zz * zz, s * zz * zz], 1)):
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")
        def cl():
            opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(X[trt] @ w, yy[trt]); l.backward(); return l
        opt.step(cl)
        with torch.no_grad():
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(X[~trt] @ w, yy[~trt]).item())
    return gap_lo, gap_hi, float(np.mean(ratios)), losses[1] - losses[0]


def main():
    torch.set_num_threads(4)
    for name, path in (("temp", "val-stream.bin"), ("open", "val-streamopen.bin")):
        occ = load_occupancy(path); tiles = occ.sum(1)
        sc, y = B.load_labels(path, len(occ))
        npz = np.load(os.path.expanduser(f"~/data/ising/{name}.npz"))
        real = {(lo, hi): (npz[f"h_{lo:02d}-{hi:02d}"].astype(np.float32), npz[f"W_{lo:02d}-{hi:02d}"].astype(np.float32))
                for lo, hi in BANDS if f"h_{lo:02d}-{hi:02d}" in npz}
        ub = np.full(len(y), -1)
        for i, (lo, hi) in enumerate(B.UNSEEN):
            ub[(sc[:, 0] >= lo - 0.5) & (sc[:, 0] <= hi + 0.5)] = i
        ok = (ub >= 0) & (np.abs(sc[:, 1]) < 300) & (tiles >= 1) & (tiles <= 100)
        ubo, sbo, yo, spo, uno = ub[ok], np.digitize(sc[ok, 1], B.SPREAD_EDGES), y[ok], sc[ok, 1], sc[ok, 0]
        tr = np.random.default_rng(7).random(ok.sum()) < 0.5
        _, _, k0 = B.table(ubo, sbo, yo)
        w0t, c0t = np.bincount(k0[tr], yo[tr], k0.max() + 1), np.bincount(k0[tr], minlength=k0.max() + 1)
        p0 = (w0t[k0] + 1) / (c0t[k0] + 2)
        ctx = (ubo, sbo, yo, spo, uno, tr, p0, ok)
        rng = np.random.default_rng(2026)
        rows = [("real energy", score(*rank_stats(energy(occ, tiles, real), tiles), ctx))]
        h_sd = np.mean([v[0].std() for v in real.values()]); w_sd = np.mean([v[1][np.triu_indices(225, 1)].std() for v in real.values()])
        for d in range(DRAWS):
            perm = rng.permutation(225)
            pp = {k: (h[perm], W[np.ix_(perm, perm)]) for k, (h, W) in real.items()}
            rows.append((f"permuted squares {d}", score(*rank_stats(energy(occ, tiles, pp), tiles), ctx)))
            h = rng.normal(0, h_sd, 225).astype(np.float32); A = rng.normal(0, w_sd, (225, 225)).astype(np.float32)
            Wr = np.triu(A, 1); Wr = Wr + Wr.T
            rows.append((f"random smooth {d}", score(*rank_stats(energy(occ, tiles, {k: (h, Wr) for k in real}), tiles), ctx)))
            print(f"  {name}: draw {d + 1}/{DRAWS} done", file=sys.stderr, flush=True)
        print(f"\n== {name}: {ok.sum():,} positions")
        print(f"  {'energy':22s} {'tail gap <15%':>14s} {'tail gap >85%':>14s} {'slope hi/lo-1':>14s} {'logloss change':>15s}")
        for nm, (gl, gh, sr, dl) in rows:
            print(f"  {nm:22s} {gl:+14.4f} {gh:+14.4f} {sr:+14.3f} {dl:+15.6f}")
        for col, nm in ((0, "tail gap <15%"), (1, "tail gap >85%"), (2, "slope"), (3, "logloss")):
            for kind in ("permuted", "random"):
                v = np.array([r[1][col] for r in rows if r[0].startswith(kind)])
                realv = rows[0][1][col]
                beyond = np.mean(v <= realv) if col == 3 or col == 2 else np.mean(v >= realv)
                print(f"  {nm:14s} vs {kind:8s}: null mean {v.mean():+.5f} sd {v.std():.5f}; real {realv:+.5f}; share of nulls at least as extreme {beyond:.2f}")


if __name__ == "__main__":
    main()
