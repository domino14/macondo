#!/usr/bin/env python
"""Export fitted Ising parameters (ising_fit.py .npz) to the binary format the
Go package boardenergy reads, plus a parity fixture of real boards.

Format, little-endian: "ISNG", uint32 version (1), uint32 band count; per
band: uint32 lo, uint32 hi (tiles on the board), 225 float32 h, 225x225
float32 W (symmetric, zero diagonal, row-major).
Fixture (--fixture): uint32 n; per board: 225 uint8 occupancy, float64 E
under the band model of its tile count.

    python ising_export.py --npz ~/data/ising/open.npz --out ../data/strategy/default/ising/open-v1.bin \
        --fixture ../boardenergy/testdata/fixture.bin --frames val-streamopen.bin
"""
import argparse, os, struct
import numpy as np
from ising_fit import BANDS, load_occupancy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixture")
    ap.add_argument("--frames")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    z = np.load(os.path.expanduser(a.npz))
    bands = [(lo, hi) for lo, hi in BANDS if f"h_{lo:02d}-{hi:02d}" in z]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(b"ISNG" + struct.pack("<II", 1, len(bands)))
        for lo, hi in bands:
            h = z[f"h_{lo:02d}-{hi:02d}"].astype("<f4"); W = z[f"W_{lo:02d}-{hi:02d}"].astype("<f4")
            assert h.shape == (225,) and W.shape == (225, 225)
            f.write(struct.pack("<II", lo, hi)); f.write(h.tobytes()); f.write(W.tobytes())
    print(f"wrote {a.out}: {len(bands)} bands")
    if a.fixture:
        occ = load_occupancy(a.frames, 20000)
        rng = np.random.default_rng(3)
        pick = rng.choice(len(occ), a.n, replace=False)
        os.makedirs(os.path.dirname(os.path.abspath(a.fixture)), exist_ok=True)
        with open(a.fixture, "wb") as f:
            f.write(struct.pack("<I", a.n))
            for i in pick:
                o = occ[i]; t = int(o.sum())
                lo, hi = next(b for b in bands if b[0] <= t <= b[1])
                h = z[f"h_{lo:02d}-{hi:02d}"].astype(np.float64); W = z[f"W_{lo:02d}-{hi:02d}"].astype(np.float64)
                s = o.astype(np.float64) * 2 - 1
                E = -(s @ h + 0.5 * s @ W @ s)
                f.write(o.astype(np.uint8).tobytes()); f.write(struct.pack("<d", E))
        print(f"wrote {a.fixture}: {a.n} boards")


if __name__ == "__main__":
    main()
