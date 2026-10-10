"""Paired-match summary for an autoplay game-pairs CSV (games-<exp>.txt)."""
import csv, math, sys
from collections import defaultdict

# Several files pool their pairs (separate runs of the same matchup; game
# IDs are seeds, so they do not collide). The "+/-" printed is the 95%
# interval half-width, 1.96 standard errors of the per-pair win share.
paths = sys.argv[1:]
path = paths[0] if len(paths) == 1 else "+".join(paths)
rows = [r for p in paths for r in csv.DictReader(open(p))]
cols = [c for c in rows[0] if c.endswith("_score")]
a_col, b_col = cols[0], cols[1]
pairs = defaultdict(list)
n = 0
spread = 0.0
for r in rows:
    a, b = int(r[a_col]), int(r[b_col])
    n += 1
    spread += a - b
    pairs[r["gameID"]].append(1.0 if a > b else 0.5 if a == b else 0.0)
full = [v for v in pairs.values() if len(v) == 2]
ps = [sum(v) / 2 for v in full]
m = sum(ps) / len(ps)
sd = math.sqrt(sum((p - m) ** 2 for p in ps) / (len(ps) - 1))
ci = 1.96 * sd / math.sqrt(len(ps))
swept = sum(1 for v in full if sum(v) == 2)
lost = sum(1 for v in full if sum(v) == 0)
print(
    f"{path}: games={n} pairs={len(full)} {a_col[:-6]} paired win rate "
    f"{100*m:.2f}% +/- {100*ci:.2f}  swept {100*swept/len(full):.1f}%  "
    f"lost {100*lost/len(full):.1f}%  spread {spread/n:+.1f}/game  "
    f"(+/- is the 95% interval; SE {100*sd/math.sqrt(len(ps)):.2f})"
)
