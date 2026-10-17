#!/usr/bin/env python3
"""Summarises decidelab output: what each chooser gains over `none`, by bag size.

A chooser's gain in a position is the judge's win % for its move minus the
judge's for none's move (0 when they chose the same). none2 is the noise floor,
infer is inference, oracle the ceiling (a sim that knows the true leave).
Intervals are 95% bootstrap intervals over positions.

    analyze.py decidelab.jsonl [more.jsonl ...]
"""
import json
import random
import sys

BUCKETS = [(2, 7), (8, 20), (21, 50), (51, 100)]
ARMS = ["none2", "infer", "oracle"]


def load(paths):
    recs = []
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("err") or "none" not in r["choices"]:
                    continue
                recs.append(r)
    return recs


def gain(r, arm):
    c, n = r["choices"].get(arm), r["choices"]["none"]
    if c is None:
        return None
    if c["move"] == n["move"]:
        return 0.0
    return 100 * (c["judgeWin"] - n["judgeWin"])


def changed(r, arm):
    c = r["choices"].get(arm)
    return c is not None and c["move"] != r["choices"]["none"]["move"]


def ci(xs, n=2000):
    if not xs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(1)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return sum(xs) / len(xs), means[int(0.025 * n)], means[int(0.975 * n)]


def table(recs, label):
    print(f"\n{label}: {len(recs)} positions")
    print(f"{'bag':>8} {'n':>5}  " + "  ".join(f"{a + ' changed':>15} {a + ' gain (pts/turn)':>30}" for a in ARMS))
    for lo, hi in BUCKETS + [(0, 100)]:
        sub = [r for r in recs if lo <= r["bag"] <= hi]
        if not sub:
            continue
        cells = []
        for a in ARMS:
            g = [x for x in (gain(r, a) for r in sub) if x is not None]
            if not g:
                cells.append(f"{'-':>15} {'-':>30}")
                continue
            m, l, h = ci(g)
            ch = 100 * sum(changed(r, a) for r in sub) / len(sub)
            cells.append(f"{ch:>14.1f}% {f'{m:+.3f} [{l:+.3f}, {h:+.3f}]':>30}")
        name = "all" if (lo, hi) == (0, 100) else f"{lo}-{hi}"
        print(f"{name:>8} {len(sub):>5}  " + "  ".join(cells))


def ranks(recs):
    print("\nstatic rank of the chosen move (1 = top static play; 0 = beyond 1000)")
    for a in ["none", "infer", "oracle"]:
        rs = [r["choices"][a]["staticRank"] for r in recs if a in r["choices"]]
        if not rs:
            continue
        n = len(rs)
        frac = lambda f: 100 * sum(1 for x in rs if f(x)) / n
        print(f"  {a:>6}: top1 {frac(lambda x: x == 1):5.1f}%  2-10 {frac(lambda x: 2 <= x <= 10):5.1f}%  "
              f"11-50 {frac(lambda x: 11 <= x <= 50):5.1f}%  51-100 {frac(lambda x: 51 <= x <= 100):5.2f}%  "
              f">100 {frac(lambda x: x > 100 or x == 0):5.2f}%  max {max(rs)}")


def main():
    recs = load(sys.argv[1:])
    table(recs, "all positions")
    table([r for r in recs if r["exchange"]], "opponent exchanged")
    table([r for r in recs if not r["exchange"]], "opponent played")
    ranks(recs)


if __name__ == "__main__":
    main()
