"""Feature sizes, which depend on the alphabet (Go: game/mllayout.go).

With L letters (not counting the blank) a position is 3L+7 board planes of
15x15 and 2L+20 scalars:

    planes   0..L-1 letters, L blank, L+1..2L horizontal cross-checks,
             2L+1..3L vertical cross-checks, 3L+1..3L+4 bonus squares,
             3L+5..3L+6 history
    scalars  0..L rack (blank first), L+1..2L+1 unseen, then 18 more

English and French have 26 letters (85 planes, 72 scalars), German 29 (94,
78). Set MACONDO_LETTERS for another alphabet; every script that reads
frames or builds a model takes the sizes from here, and checkpoints record
the alphabet they were trained on.
"""

import os

LETTERS = int(os.environ.get("MACONDO_LETTERS", "26"))
H, W = 15, 15
C = 3 * LETTERS + 7
N_SCAL = 2 * LETTERS + 20
N_TILE_TYPES = LETTERS + 1  # blank + the letters: rack[0:L+1], unseen[L+1:2L+2]

# Diagonal transpose: the one board symmetry that keeps words readable. It
# swaps the horizontal cross-check planes with the vertical ones and keeps
# every other plane in place.
_HOR = list(range(LETTERS + 1, 2 * LETTERS + 1))
_VER = list(range(2 * LETTERS + 1, 3 * LETTERS + 1))
TRANSPOSE_PERM = list(range(LETTERS + 1)) + _VER + _HOR + list(range(3 * LETTERS + 1, C))
assert sorted(TRANSPOSE_PERM) == list(range(C))


def check_letters(ckpt, path):
    """Refuse a checkpoint trained on another alphabet (old ones: English)."""
    letters = ckpt.get("letters", 26)
    if letters != LETTERS:
        raise SystemExit(
            f"{path} was trained on a {letters}-letter alphabet; set MACONDO_LETTERS={letters}"
        )
