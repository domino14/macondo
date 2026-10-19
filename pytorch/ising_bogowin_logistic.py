#!/usr/bin/env python
"""Smooth logistic version of ising_bogowin.py: P(win) from unseen tiles and spread, with and without board-energy terms; held-out log loss."""
import sys, os, numpy as np, torch
sys.path.insert(0, "/home/cesar/code/macondo/pytorch")
from ising_fit import load_occupancy
import ising_bogowin as B
for name, path in (("temp", "val-stream.bin"), ("open", "val-streamopen.bin")):
    occ = load_occupancy(path); sc, y = B.load_labels(path, len(occ))
    E, terc = B.energies(occ, np.load(os.path.expanduser(f"~/data/ising/{name}.npz")))
    tiles = occ.sum(1); z = np.full(len(E), np.nan)
    for t in np.unique(tiles):
        j = np.flatnonzero((tiles == t) & ~np.isnan(E))
        if len(j) >= 30: z[j] = (E[j] - E[j].mean()) / (E[j].std() + 1e-9)
    ok = (sc[:, 0] >= 8) & ~np.isnan(z) & (np.abs(sc[:, 1]) < 300)
    u = torch.tensor(sc[ok, 0] / 100, dtype=torch.float64); s = torch.tensor(sc[ok, 1] / 100, dtype=torch.float64)
    zz = torch.tensor(np.clip(z[ok], -3, 3), dtype=torch.float64); yy = torch.tensor(y[ok], dtype=torch.float64)
    tr = torch.tensor(np.random.default_rng(7).random(ok.sum()) < 0.5)
    def feats(withE):
        # slope and intercept as smooth (quadratic) functions of unseen tiles; with energy: slope and intercept also shift with energy
        base = [torch.ones_like(u), u, u * u, s, s * u, s * u * u, s * s * s]
        if withE: base += [zz, s * zz, s * zz * u, zz * zz, s * zz * zz]
        return torch.stack(base, 1)
    out = []
    for withE in (False, True):
        X = feats(withE); w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")
        def cl():
            opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(X[tr] @ w, yy[tr]); l.backward(); return l
        opt.step(cl)
        with torch.no_grad():
            per = torch.nn.functional.binary_cross_entropy_with_logits(X[~tr] @ w, yy[~tr], reduction="none")
        out.append((per, w.detach()))
    d = (out[1][0] - out[0][0]).numpy()
    print(f"{name}: {ok.sum():,} positions; held-out log loss smooth logistic {out[0][0].mean():.5f}, + energy terms {out[1][0].mean():.5f}, change {d.mean():+.5f} (SE {d.std()/np.sqrt(len(d)):.5f})")
    w = out[1][1].numpy(); print(f"   energy coefficients: intercept {w[7]:+.4f}  spread*E {w[8]:+.4f}  spread*E*unseen {w[9]:+.4f}  E^2 {w[10]:+.4f}  spread*E^2 {w[11]:+.4f}")
