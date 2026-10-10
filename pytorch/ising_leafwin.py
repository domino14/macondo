#!/usr/bin/env python
"""Fit the sim's end-of-line win model from our games, with and without
board energy, and export it for the Go simmer (montecarlo leaf win model).

Convention, as the sim's table lookup: P(the player ON TURN wins | their
spread, unseen tiles[, board energy z]). The frame caches hold the board
after the mover's play with the opponent on turn and the mover's result, so
the on-turn player's spread is minus the mover's and their result is the
complement.

Features (u = unseen/100, s = on-turn spread/100, z = energy z-score among
boards with the same tile count, clipped to +/-3):
  base   1, u, u^2, s, s u, s u^2, s^3
  energy z, s z, s z u, z^2, s z^2
Writes JSON: coefficients for both models and, per tile count, the mean and
SD of E used for z.

    python ising_leafwin.py --frames val-streamopen.bin --npz ~/data/ising/open.npz \
        --out ../data/strategy/default/ising/leafwin-v1.json
"""
import argparse, json, os
import numpy as np, torch
from ising_fit import load_occupancy
import ising_bogowin as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    occ = load_occupancy(a.frames); tiles = occ.sum(1)
    sc, y = B.load_labels(a.frames, len(occ))
    E, _ = B.energies(occ, np.load(os.path.expanduser(a.npz)))
    stats = {}
    z = np.full(len(E), np.nan)
    for t in np.unique(tiles):
        j = np.flatnonzero((tiles == t) & ~np.isnan(E))
        if len(j) < 30:
            continue
        mu, sd = float(E[j].mean()), float(E[j].std() + 1e-9)
        stats[int(t)] = [mu, sd]
        z[j] = (E[j] - mu) / sd
    ok = (sc[:, 0] >= 1) & ~np.isnan(z) & (np.abs(sc[:, 1]) < 300)
    u = torch.tensor(sc[ok, 0] / 100, dtype=torch.float64)
    s = torch.tensor(-sc[ok, 1] / 100, dtype=torch.float64)       # on-turn player's spread
    zz = torch.tensor(np.clip(z[ok], -3, 3), dtype=torch.float64)
    yy = torch.tensor(1 - y[ok], dtype=torch.float64)              # on-turn player's result
    tr = torch.tensor(np.random.default_rng(7).random(ok.sum()) < 0.5)
    base = [torch.ones_like(u), u, u * u, s, s * u, s * u * u, s * s * s]
    out = {"features_base": ["1", "u", "u^2", "s", "s*u", "s*u^2", "s^3"],
           "features_energy": ["z", "s*z", "s*z*u", "z^2", "s*z^2"],
           "u": "unseen tiles / 100", "s": "on-turn player's spread / 100",
           "z": "(E - mean[tiles]) / sd[tiles], clipped to +/-3", "energy_stats": stats}
    for name, X in (("base", torch.stack(base, 1)), ("energy", torch.stack(base + [zz, s * zz, s * zz * u, zz * zz, s * zz * zz], 1))):
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=300, line_search_fn="strong_wolfe")
        def cl():
            opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(X[tr] @ w, yy[tr]); l.backward(); return l
        opt.step(cl)
        with torch.no_grad():
            ll = torch.nn.functional.binary_cross_entropy_with_logits(X[~tr] @ w, yy[~tr]).item()
        out[f"coef_{name}"] = w.detach().tolist(); out[f"heldout_logloss_{name}"] = ll
        print(f"{name}: held-out log loss {ll:.5f}  coef {np.round(w.detach().numpy(), 4).tolist()}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
