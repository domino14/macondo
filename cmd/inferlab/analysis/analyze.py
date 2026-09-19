#!/usr/bin/env python3
"""Read inferlab JSONL and report on the proposal and the imputation.

Three questions, one per section:

  1. Does the replay reproduce the run it came from?
  2. Per refine round, does the model's prediction for a leaf match what the
     mini-sim finds, and does the proposal point at leaves worth measuring?
  3. Does the run's outcome depend on whether the true leave got measured?
"""
import json, math, sys
from collections import defaultdict


def load(path):
    """Records, with a null lift (the leave was ruled out) read as -inf."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        for k in ("liftBits", "loggedLiftBits"):
            if r.get(k) is None:
                r[k] = float("-inf")
        out.append(r)
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def stderr(xs):
    xs = list(xs)
    n = len(xs)
    if n < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def ranks(xs):
    """Average ranks, ties shared — what Spearman needs."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def spearman(xs, ys):
    if len(xs) < 3:
        return float("nan")
    return pearson(ranks(xs), ranks(ys))


# A measured likelihood of exactly 0 is real evidence, not missing data, so it
# stays in. log(0) is floored well below any observed positive value rather than
# dropped, which would quietly remove the very cases that matter most.
FLOOR = 1e-30
def lg(v):
    return math.log10(max(v, FLOOR))


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main(path):
    recs = [r for r in load(path) if not r.get("err")]
    bad = [r for r in load(path) if r.get("err")]
    print(f"{len(recs)} positions replayed" + (f", {len(bad)} errored" if bad else ""))
    if not recs:
        return
    v = {r.get("variant", "?") for r in recs}
    p = {r.get("proposal", "?") for r in recs}
    print(f"variant(s): {', '.join(sorted(v))}   proposal(s): {', '.join(sorted(p))}")

    # ---- 1. does the replay reproduce the run? ----------------------------
    section("Replay against the original run")
    withlog = [r for r in recs if r.get("hasLogged")]
    if withlog:
        finite = [r for r in withlog if math.isfinite(r["liftBits"]) and math.isfinite(r["loggedLiftBits"])]
        nruled = len(withlog) - len(finite)
        if nruled:
            print(f"  {nruled} of {len(withlog)} had the leave ruled out by one side or the other;")
            print("  they have no ratio to average and are left out of the means below.")
        a = [r["liftBits"] for r in finite]
        b = [r["loggedLiftBits"] for r in finite]
        print(f"  replayed lift  {mean(a):+7.3f} ± {1.96*stderr(a):.3f}   median {median(a):+7.3f}")
        print(f"  logged   lift  {mean(b):+7.3f} ± {1.96*stderr(b):.3f}   median {median(b):+7.3f}")
        print(f"  per-position correlation: {pearson(a, b):+.3f} (Pearson), {spearman(a, b):+.3f} (Spearman)")
        print(f"  measured/imputed verdict agrees: "
              f"{sum(1 for r in withlog if r['measured'] == r['loggedMeasured'])}/{len(withlog)}")
        print(f"  ruled out: {sum(1 for r in withlog if not math.isfinite(r['liftBits']))} on replay, "
              f"{sum(1 for r in withlog if not math.isfinite(r['loggedLiftBits']))} in the run")
        print("  (the means should match; single positions need not, and mostly will not)")

    # ---- 2. the proposal, round by round -----------------------------------
    section("Per round: what the model predicted, and what the mini-sim found")
    byround = defaultdict(list)
    for r in recs:
        for d in r.get("draws", []):
            byround[d["Round"]].append(d)
    if not byround:
        print("  no draws recorded — rerun with -trace")
    else:
        base = mean(d["Measured"] for d in byround.get(0, [])) or float("nan")
        print(f"  {'round':<6}{'draws':>7}{'mean w':>12}{'median w':>12}{'vs round 0':>12}"
              f"{'zeros':>8}{'ρ(lhat,w)':>11}{'ρ(q,w)':>9}")
        for rd in sorted(byround):
            ds = byround[rd]
            w = [d["Measured"] for d in ds]
            enrich = mean(w) / base if base and not math.isnan(base) and base > 0 else float("nan")
            zeros = 100 * sum(1 for x in w if x <= 0) / len(w)
            if rd == 0:
                rho_l = rho_q = float("nan")
            else:
                rho_l = spearman([lg(d["Predicted"]) for d in ds], [lg(d["Measured"]) for d in ds])
                rho_q = spearman([lg(d["Q"]) for d in ds], [lg(d["Measured"]) for d in ds])
            print(f"  {rd:<6}{len(ds):>7}{mean(w):>12.3e}{median(w):>12.3e}{enrich:>11.2f}x"
                  f"{zeros:>7.1f}%{rho_l:>11.3f}{rho_q:>9.3f}")
        print("\n  vs round 0 = mean measured likelihood against the blind prior sample.")
        print("  Above 1 means the proposal is finding better leaves than chance would.")
        print("  rho(lhat,w) is the model's out-of-sample calibration: it predicted lhat,")
        print("  the mini-sim then found w. rho(q,w) is what the proposal actually delivered.")

        # pooled, which is the number with the sample size behind it
        allref = [d for rd, ds in byround.items() if rd >= 1 for d in ds]
        if len(allref) >= 3:
            pl = spearman([lg(d["Predicted"]) for d in allref], [lg(d["Measured"]) for d in allref])
            pq = spearman([lg(d["Q"]) for d in allref], [lg(d["Measured"]) for d in allref])
            print(f"\n  pooled over all refine draws (n={len(allref)}): "
                  f"rho(lhat,w) = {pl:+.3f}, rho(q,w) = {pq:+.3f}")

    # ---- 3. did the truth get measured? ------------------------------------
    section("Outcome by whether the true leave was measured")
    for lab, sel in (("measured", lambda r: r["measured"]),
                     ("imputed", lambda r: not r["measured"])):
        s = [r for r in recs if sel(r)]
        if not s:
            continue
        l = [r["liftBits"] for r in s if math.isfinite(r["liftBits"])]
        print(f"  {lab:<9} n={len(s):<5} mean {mean(l):+7.3f} ± {1.96*stderr(l):.3f}   "
              f"median {median(l):+7.3f}   ruled out {sum(1 for r in s if r['ruledOut'])}")
    ro = sum(1 for r in recs if r["ruledOut"])
    print(f"  the true leave got a mini-sim in {sum(1 for r in recs if r['measured'])}/{len(recs)} positions")
    if ro:
        print(f"  and was given no weight at all in {ro}")

    # ---- calibration overfit -----------------------------------------------
    gaps = [r["logCalib"] - r["logCalibInSample"] for r in recs if r.get("rounds")]
    if gaps:
        section("How far the imputation overfits (cross-fitted minus in-sample logCalib)")
        print(f"  median {median(gaps):+.3f} nats ({math.exp(median(gaps)):.2f}x)   "
              f"mean {mean(gaps):+.3f}   max {max(gaps):+.3f}")
        lo = [r for r in recs if r.get("rounds") and
              (r["logCalib"] - r["logCalibInSample"]) <= median(gaps)]
        hi = [r for r in recs if r.get("rounds") and
              (r["logCalib"] - r["logCalibInSample"]) > median(gaps)]
        if lo and hi:
            fl = lambda g: [r["liftBits"] for r in g if math.isfinite(r["liftBits"])]
            print(f"  lift where the gap is small (n={len(lo)}): {mean(fl(lo)):+.3f}")
            print(f"  lift where the gap is large (n={len(hi)}): {mean(fl(hi)):+.3f}")
            print("  (a gap that predicts a bad read would be a runtime warning sign, no oracle needed)")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print("=" * 72)
        print(path)
        print("=" * 72)
        main(path)
