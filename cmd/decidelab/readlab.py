#!/usr/bin/env python3
"""How close does each read get to the oracle? Summarises decidelab output run
with read grading (always on) and -agree.

Per inferring arm and bag bucket:
  exact-leave figures   lift bits, the true leave's rank (median, share in top 5)
  partial credit        soft lifts (overlap and value kernels), expected tile
                        overlap and leave-value distance, each against the prior
  decisions (-agree)    regret, in win points, of the moves leaves drawn from the
                        posterior lead to, priced against the true leave; the same
                        for the prior and for the truth itself (sim-noise floor).
                        closed = (prior - post) / (prior - truth): the share of the
                        way from knowing nothing to knowing the leave.
  gain                  the judge's verdict on the arm's own pick against none's

Then, for each arm, how well each read metric predicts the position's decision
regret gain and judge gain (Spearman over positions).

    readlab.py decidelab.jsonl [more.jsonl ...]
"""
import json
import math
import random
import statistics as st
import sys

BUCKETS = [(2, 7), (8, 20), (21, 50), (51, 100)]


def load(paths):
    recs = []
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("err") and "none" in r["choices"]:
                    recs.append(r)
    return recs


def ci(xs, n=2000):
    if not xs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(1)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return sum(xs) / len(xs), means[int(0.025 * n)], means[int(0.975 * n)]


def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def gain(r, arm):
    c, n = r["choices"][arm], r["choices"]["none"]
    return 0.0 if c["move"] == n["move"] else 100 * (c["judgeWin"] - n["judgeWin"])


def arms_in(recs):
    seen = {a for r in recs for a, c in r["choices"].items() if c.get("read")}
    return (["infer"] if "infer" in seen else []) + sorted(seen - {"infer"})


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def summary(recs, arm):
    print(f"\n== {arm}")
    print(f"{'bag':>7} {'n':>5} {'ms':>6} {'lift':>6} {'medRk':>6} {'top5':>5} {'softO':>6} {'softV':>6} "
          f"{'ovl+':>6} {'vdist-':>6} | {'nAg':>4} {'Rtruth':>6} {'Rprior':>6} {'Rpost':>6} {'closed':>7} "
          f"{'prior-post [95%]':>20} | {'gain [95%]':>22}")
    for lo, hi in BUCKETS + [(0, 100)]:
        sub = [r for r in recs if lo <= r["bag"] <= hi and arm in r["choices"]]
        if not sub:
            continue
        rd = [r["choices"][arm].get("read") or {} for r in sub]
        comp = [x for x in rd if x.get("complete")]
        lift = mean([x.get("liftBits") for x in rd])
        rks = [x["rank"] for x in rd if x.get("rank")]
        top5 = 100 * sum(1 for k in rks if k <= 5) / len(rd) if rd else float("nan")
        ag = [r["agree"] for r in sub if r.get("agree") and arm in r["agree"]["arms"]]
        rt = mean([a["truthRegret"] for a in ag])
        rp = mean([a["priorRegret"] for a in ag])
        rq = mean([a["arms"][arm]["postRegret"] for a in ag])
        closed = (rp - rq) / (rp - rt) if ag and rp != rt else float("nan")
        m, l, h = ci([a["priorRegret"] - a["arms"][arm]["postRegret"] for a in ag])
        gm, gl, gh = ci([gain(r, arm) for r in sub])
        name = "all" if (lo, hi) == (0, 100) else f"{lo}-{hi}"
        print(f"{name:>7} {len(sub):>5} {mean([r['choices'][arm]['elapsedMs'] for r in sub]):>6.0f} "
              f"{lift:>6.2f} {st.median(rks) if rks else float('nan'):>6.0f} {top5:>4.1f}% "
              f"{mean([x['softLiftOverlap'] for x in comp]):>6.2f} {mean([x['softLiftValue'] for x in comp]):>6.2f} "
              f"{mean([x['overlap'] - x['overlapPrior'] for x in comp]):>+6.3f} "
              f"{mean([x['valueDistPrior'] - x['valueDist'] for x in comp]):>+6.2f} | "
              f"{len(ag):>4} {rt:>6.2f} {rp:>6.2f} {rq:>6.2f} {100 * closed:>6.0f}% "
              f"{f'{m:+.2f} [{l:+.2f}, {h:+.2f}]':>20} | {f'{gm:+.3f} [{gl:+.3f}, {gh:+.3f}]':>22}")


def predictors(recs, arm):
    rows = []
    for r in recs:
        rd = (r["choices"].get(arm) or {}).get("read") or {}
        if not rd.get("complete"):
            continue
        ag = (r.get("agree") or {}).get("arms", {}).get(arm)
        rows.append({
            "lift": rd.get("liftBits", -10.0) if rd.get("liftBits") is not None else -10.0,
            "softO": rd["softLiftOverlap"], "softV": rd["softLiftValue"],
            "ovl+": rd["overlap"] - rd["overlapPrior"], "vdist-": rd["valueDistPrior"] - rd["valueDist"],
            "regretGain": (r["agree"]["priorRegret"] - ag["postRegret"]) if ag else None,
            "judgeGain": gain(r, arm),
        })
    if len(rows) < 10:
        return
    print(f"\n{arm}: Spearman of each read metric with the position's decision outcome ({len(rows)} positions)")
    for target in ["regretGain", "judgeGain"]:
        sub = [x for x in rows if x[target] is not None]
        cells = "  ".join(f"{k} {spearman([x[k] for x in sub], [x[target] for x in sub]):+.3f}"
                          for k in ["lift", "softO", "softV", "ovl+", "vdist-"])
        print(f"  {target:>10} (n={len(sub)}): {cells}")


def main():
    recs = load(sys.argv[1:])
    print(f"{len(recs)} positions")
    for a in arms_in(recs):
        summary(recs, a)
    for a in arms_in(recs):
        predictors(recs, a)


if __name__ == "__main__":
    main()
