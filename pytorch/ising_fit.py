#!/usr/bin/env python
"""Pairwise maximum-entropy (Ising) model of board occupancy.

After Witteveen & Bauer, "Statistical mechanics for Scrabble predicts
strategy, entropy and language" (arXiv 2605.00813): each square is a spin
(+1 a tile, -1 empty) and

    E(s) = - sum_{i<j} W_ij s_i s_j - sum_i h_i s_i,     p(s) ~ exp(-E(s))

on connected tile patterns. Fitted by pseudolikelihood restricted, as in
the paper, to the squares whose flip keeps the pattern connected: an empty
square next to a tile, or a tile that is not an articulation point.

Unlike the paper (final boards only) the boards here are mid-game: the
positions in a training frame cache (one random turn per game), fitted
separately per band of tiles on the board, because the use is a move-level
shape bonus during game generation.

    python ising_fit.py --frames temp=val-stream.bin open=val-streamopen.bin --out ~/data/ising

Writes <out>/<name>.npz (h and W per band) and <out>/summary.json, and
prints: held-out pseudolikelihood against the independent-squares model,
the structure of h and W, board-shape statistics per band, and how well
each data set's model explains the other's boards.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

from training import CACHE_ROW_BYTES

H = W_ = 15
N = H * W_
CENTER = 7 * 15 + 7
LETTER_BITS = 26 * N  # planes 0..25 are the letters; a square is occupied if any is set
LETTER_BYTES = (LETTER_BITS + 7) // 8
BANDS = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 100)]
LAYOUT = [
    "=  '   =   '  =", " -   \"   \"   - ", "  -   ' '   -  ", "'  -   '   -  '",
    "    -     -    ", " \"   \"   \"   \" ", "  '   ' '   '  ", "=  '   -   '  =",
    "  '   ' '   '  ", " \"   \"   \"   \" ", "    -     -    ", "'  -   '   -  '",
    "  -   ' '   -  ", " -   \"   \"   - ", "=  '   =   '  =",
]
KIND = {"=": "TW", "-": "DW", '"': "TL", "'": "DL", " ": "plain"}
NB = [[] for _ in range(N)]
for r in range(H):
    for c in range(W_):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W_:
                NB[r * 15 + c].append(rr * 15 + cc)


def load_occupancy(path, limit=0):
    """Frame cache -> uint8 (n, 225), 1 where a tile sits."""
    n = os.path.getsize(path) // CACHE_ROW_BYTES
    if limit:
        n = min(limit, n)
    out = np.empty((n, N), dtype=np.uint8)
    with open(path, "rb") as f:  # plain reads: a memmap of a multi-GB cache trips ulimit -v
        for a in range(0, n, 20000):
            b = min(n, a + 20000)
            rows = np.frombuffer(f.read((b - a) * CACHE_ROW_BYTES), dtype=np.uint8).reshape(-1, CACHE_ROW_BYTES)
            bits = np.unpackbits(rows[:, :LETTER_BYTES], axis=1)[:, :LETTER_BITS]
            out[a:b] = bits.reshape(-1, 26, N).any(axis=1)
    return out


def flippable(occ):
    """Squares whose flip keeps the tiles connected, for one board (uint8[225]).

    Returns None if the board is empty or its tiles are not one connected
    component (never the case for a real position)."""
    tiles = np.flatnonzero(occ).tolist()
    if not tiles:
        return None
    s = set(tiles)
    out = np.zeros(N, dtype=bool)
    # Tarjan's articulation points, iteratively, from the first tile.
    disc, low, art = {}, {}, set()
    root = tiles[0]
    disc[root] = low[root] = 0
    t, root_children = 1, 0
    stack = [(root, -1, iter(NB[root]))]
    while stack:
        v, parent, it = stack[-1]
        advanced = False
        for w in it:
            if w not in s or w == parent:
                continue
            if w in disc:
                low[v] = min(low[v], disc[w])
            else:
                disc[w] = low[w] = t
                t += 1
                stack.append((w, v, iter(NB[w])))
                advanced = True
                break
        if not advanced:
            stack.pop()
            if stack:
                p = stack[-1][0]
                low[p] = min(low[p], low[v])
                if p == root:
                    root_children += 1
                elif low[v] >= disc[p]:
                    art.add(p)
    if root_children > 1:
        art.add(root)
    if len(disc) != len(s):
        return None
    for v in tiles:
        if len(s) > 1 and v not in art:
            out[v] = True
        for w in NB[v]:
            if w not in s:
                out[w] = True
    return out


def flippable_many(occ):
    rows = [flippable(o) for o in occ]
    ok = np.array([r is not None for r in rows])
    m = np.zeros((len(occ), N), dtype=bool)
    for i, r in enumerate(rows):
        if r is not None:
            m[i] = r
    return m, ok


def masks(occ, procs):
    parts = np.array_split(occ, max(1, len(occ) // 2000))
    with Pool(procs) as p:
        res = p.map(flippable_many, parts)
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


def pl_terms(S, M, h, Wm):
    """Sum over flippable squares of log p(s_i | rest), and their count."""
    heff = h + S @ Wm
    ll = -torch.nn.functional.softplus(-2 * S * heff)
    return (ll * M).sum(), M.sum()


def fit(S, M, l2, iters):
    """Pseudolikelihood fit. S: (n, 225) of +-1; M: 1 where the square may flip."""
    h = torch.zeros(N, dtype=torch.float64, requires_grad=True)
    A = torch.zeros(N, N, dtype=torch.float64, requires_grad=True)
    off = 1 - torch.eye(N, dtype=torch.float64)
    opt = torch.optim.LBFGS([h, A], lr=1.0, max_iter=iters, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        Wm = (A + A.T) / 2 * off
        ll, n = pl_terms(S, M, h, Wm)
        loss = -ll / n + l2 * ((Wm**2).sum() / 2 + (h**2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return h.detach().clone(), ((A + A.T) / 2 * off).detach().clone()


def fit_independent(S, M):
    """The no-coupling model under the same restriction: one field per square."""
    up = ((S > 0) * M).sum(0)
    n = M.sum(0)
    p = ((up + 0.5) / (n + 1)).clamp(1e-6, 1 - 1e-6)
    return 0.5 * torch.log(p / (1 - p))


def shape_stats(occ):
    """Board-shape observables, averaged over the boards (uint8 (n, 225))."""
    o = occ.reshape(-1, H, W_).astype(bool)
    n = o.sum((1, 2)).astype(np.float64)
    z = np.zeros_like(o[:, :, :1])
    left = np.concatenate([z, o[:, :, :-1]], 2)
    right = np.concatenate([o[:, :, 1:], z], 2)
    zr = np.zeros_like(o[:, :1, :])
    up = np.concatenate([zr, o[:, :-1, :]], 1)
    down = np.concatenate([o[:, 1:, :], zr], 1)
    hwords = (o & ~left & right).sum((1, 2))
    vwords = (o & ~up & down).sum((1, 2))
    hcells = (o & (left | right)).sum((1, 2))
    vcells = (o & (up | down)).sum((1, 2))
    words = hwords + vwords
    # perimeter: tile edges facing an empty square or the rim
    per = sum((o & ~x).sum((1, 2)) for x in (left, right, up, down))
    frontier = (~o & (left | right | up | down)).sum((1, 2))
    rr, cc = np.mgrid[0:H, 0:W_]
    mr = (o * rr).sum((1, 2)) / n
    mc = (o * cc).sum((1, 2)) / n
    rg = np.sqrt((o * ((rr - mr[:, None, None]) ** 2 + (cc - mc[:, None, None]) ** 2)).sum((1, 2)) / n)
    return {
        "tiles": float(n.mean()),
        "words": float(words.mean()),
        "word_len": float((hcells + vcells).sum() / max(1, words.sum())),
        "perimeter_per_tile": float((per / n).mean()),
        "frontier_per_tile": float((frontier / n).mean()),
        "gyration": float(rg.mean()),
    }


def structure(h, Wm):
    """Mean field per square kind; mean coupling per displacement."""
    h, Wm = h.numpy(), Wm.numpy()
    kinds = {}
    for r in range(H):
        for c in range(W_):
            k = "center" if r * 15 + c == CENTER else KIND[LAYOUT[r][c]]
            kinds.setdefault(k, []).append(h[r * 15 + c])
    out = {"h": {k: float(np.mean(v)) for k, v in kinds.items()}}
    disp = {"horizontal": (0, 1), "vertical": (1, 0), "diagonal": (1, 1), "antidiagonal": (1, -1),
            "two_apart_h": (0, 2), "two_apart_v": (2, 0), "knight": (1, 2)}
    w = {}
    for name, (dr, dc) in disp.items():
        vals = [Wm[r * 15 + c, (r + dr) * 15 + c + dc] for r in range(H) for c in range(W_)
                if 0 <= r + dr < H and 0 <= c + dc < W_]
        w[name] = float(np.mean(vals))
    far = [abs(Wm[i, j]) for i in range(N) for j in range(i + 1, N)
           if max(abs(i // 15 - j // 15), abs(i % 15 - j % 15)) >= 4]
    w["far_abs"] = float(np.mean(far))
    out["W"] = w
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True, help="name=cache.bin ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-band", type=int, default=50000, help="boards per band for the fit (10%% held out)")
    ap.add_argument("--limit", type=int, default=0, help="read only this many frames per cache (smoke test)")
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)
    summary, models, held = {}, {}, {}
    for spec in args.frames:
        name, path = spec.split("=", 1)
        occ = load_occupancy(path, args.limit)
        tiles = occ.sum(1)
        print(f"\n== {name}: {len(occ):,} boards from {path}", flush=True)
        summary[name] = {"boards": int(len(occ)), "bands": {}}
        arrays = {}
        for lo, hi in BANDS:
            band = f"{lo:02d}-{hi:02d}"
            idx = np.flatnonzero((tiles >= lo) & (tiles <= hi))
            if len(idx) < 2000:
                print(f"band {band}: {len(idx)} boards, skipped")
                continue
            stats = shape_stats(occ[idx])
            pick = rng.permutation(idx)[: args.per_band]
            m, ok = masks(occ[pick], args.procs)
            pick, m = pick[ok], m[ok]
            S = torch.from_numpy(occ[pick].astype(np.float64) * 2 - 1)
            M = torch.from_numpy(m.astype(np.float64))
            # A square that never changes in the band carries no information
            # about its own field (the centre is always covered): leave it out.
            const = (S.min(0).values == S.max(0).values)
            M[:, const] = 0
            nval = max(1, len(pick) // 10)
            Sv, Mv, St, Mt = S[:nval], M[:nval], S[nval:], M[nval:]
            h, Wm = fit(St, Mt, args.l2, args.iters)
            h0 = fit_independent(St, Mt)
            with torch.no_grad():
                ll, n = pl_terms(Sv, Mv, h, Wm)
                ll0, _ = pl_terms(Sv, Mv, h0, torch.zeros(N, N, dtype=torch.float64))
                llt, nt = pl_terms(St, Mt, h, Wm)
            st = structure(h, Wm)
            row = {"boards": int(len(idx)), "fit_boards": int(len(pick) - nval), "disconnected": int((~ok).sum()),
                   "flippable_per_board": float(M.sum() / len(pick)),
                   "pl_val": float(ll / n), "pl_train": float(llt / nt), "pl_independent": float(ll0 / n),
                   "shape": stats, **st}
            summary[name]["bands"][band] = row
            models[(name, band)] = (h, Wm)
            held[(name, band)] = (Sv, Mv)
            arrays[f"h_{band}"], arrays[f"W_{band}"] = h.numpy(), Wm.numpy()
            print(f"band {band}: {len(idx):>7,} boards  flippable/board {row['flippable_per_board']:.1f}  "
                  f"log p per flippable square: pairwise {row['pl_val']:.4f} (train {row['pl_train']:.4f})  "
                  f"independent {row['pl_independent']:.4f}", flush=True)
            print(f"   h by square: " + "  ".join(f"{k} {v:+.2f}" for k, v in st["h"].items()))
            print(f"   W by displacement: " + "  ".join(f"{k} {v:+.3f}" for k, v in st["W"].items()))
            print(f"   shape: " + "  ".join(f"{k} {v:.3f}" for k, v in stats.items()), flush=True)
        np.savez(os.path.join(out_dir, f"{name}.npz"), bands=np.array(BANDS), **arrays)
    # How well does each data set's model explain the other's held-out boards?
    names = list(summary)
    cross = {}
    if len(names) > 1:
        print("\n== cross: log p per flippable square of B's held-out boards under A's model")
        for (b, band), (Sv, Mv) in held.items():
            for a in names:
                if (a, band) not in models:
                    continue
                h, Wm = models[(a, band)]
                with torch.no_grad():
                    ll, n = pl_terms(Sv, Mv, h, Wm)
                cross[f"{band} model={a} boards={b}"] = float(ll / n)
        for band in sorted({k[1] for k in held}):
            print(f"band {band}: " + "  ".join(
                f"{a}->{b} {cross.get(f'{band} model={a} boards={b}', float('nan')):.4f}" for a in names for b in names))
    summary["cross"] = cross
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {out_dir}/summary.json and " + ", ".join(f"{n}.npz" for n in names))


if __name__ == "__main__":
    main()
