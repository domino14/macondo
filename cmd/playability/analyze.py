#!/usr/bin/env python3
"""Final playability tables from `playability score` output.

Per alphagram (and per word inside it):
  1. Combine estimator A (self-play) and B (conditioned re-deals) by inverse
     variance. A's standard error gets a floor of one unresolved hit (C points
     over G games), so a word never seen in self-play is "about zero, give or
     take one hit", not "exactly zero".
  2. Shrink toward a prior: regress log playability on features (length,
     log probability, anagram count, hooks, power tiles, vowels) over the
     alphagrams that are well measured, then combine each alphagram's own
     log estimate with the prediction, normal-normal. Alphagrams with little
     evidence (most 9+ letter ones) end up near the prior, which leans on
     probability.

Units: equity points per 1000 games (HastyBot static equity).

usage: analyze.py SCORE_DIR [OUT_DIR]
"""
import csv
import math
import os
import sys

import numpy as np

C = 20.0  # points: the size of one unresolved hit, for the A floor
PER = 1000.0


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fnum(s, default=None):
    return float(s) if s not in ("", None) else default


def combine(a, ase, b, bse):
    """Inverse-variance combination; b may be None."""
    if b is None:
        return a, ase, "A"
    wa, wb = 1 / ase**2, 1 / bse**2
    return (a * wa + b * wb) / (wa + wb), math.sqrt(1 / (wa + wb)), "AB"


def b_floor(row, tpg, G):
    """One hit's worth for B: chance the rack qualifies x C / K."""
    k = fnum(row.get("b_k"), 0)
    pm = fnum(row.get("b_pmean"), 0) or 0
    if not k:
        return 0
    return pm * tpg * C / k


def main():
    d = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else d
    meta = dict(l.rstrip("\n").split("\t", 1) for l in open(os.path.join(d, "meta.tsv")))
    G = float(meta["games"])
    tpg = float(meta["turns_per_game"])
    floor_a = C / G
    delta = C / G  # log offset: one hit

    words = read_tsv(os.path.join(d, "words.tsv"))
    alphas = read_tsv(os.path.join(d, "alphas.tsv"))
    by_alpha = {}
    for w in words:
        by_alpha.setdefault(w["alphagram"], []).append(w)

    # Per-word combined values (no shrinkage: within-class shares).
    wval = {}
    for w in words:
        a, ase = fnum(w["a_play"]), math.hypot(fnum(w["a_se"]), floor_a)
        b = fnum(w["b_play"])
        bse = None
        if b is not None:
            bse = max(fnum(w["b_se"]), floor_a * 0.1, 1e-12)
        v, se, src = combine(a, ase, b, bse)
        wval[w["word"]] = (v, se)

    rows = []
    for r in alphas:
        a, ase = fnum(r["a_play"]), math.hypot(fnum(r["a_se"]), floor_a)
        b = fnum(r["b_play"])
        bse = None
        if b is not None:
            bse = max(fnum(r["b_se"]), b_floor(r, tpg, G), 1e-12)
        v, se, src = combine(a, ase, b, bse)
        ws = by_alpha.get(r["alphagram"], [])
        hooks = sum(int(w["fhooks"]) + int(w["bhooks"]) for w in ws)
        rows.append(dict(alpha=r["alphagram"], len=int(r["len"]), combos=float(r["combos"]),
                         words=r["words"].split(","), n=int(r["n"]), a=a, b=b,
                         est=max(v, 0.0), se=se, src=src, hooks=hooks))

    # Features for the prior.
    def feats(x):
        L = x["len"]
        al = x["alpha"]
        f = [1.0]
        f += [1.0 if L == k else 0.0 for k in range(3, 16)]
        lc = math.log(max(x["combos"], 1.0))
        f += [lc, lc * L / 7.0]
        f += [math.log1p(len(x["words"])), math.log1p(x["hooks"])]
        f += [float(al.count(c)) for c in "JQXZKS"]
        f += [sum(al.count(c) for c in "AEIOU") / L]
        return f

    X = np.array([feats(x) for x in rows])
    y = np.array([math.log(x["est"] + delta) for x in rows])
    vy = np.array([(x["se"] / (x["est"] + delta)) ** 2 for x in rows])
    good = np.array([x["est"] > 0 and x["se"] / x["est"] < 0.25 for x in rows])
    print(f"games {G:.0f}; prior fit on {good.sum()} well-measured alphagrams of {len(rows)}")
    beta, *_ = np.linalg.lstsq(X[good], y[good], rcond=None)
    mu = X @ beta
    resid = y[good] - mu[good]
    tau2 = max(float(np.mean(resid**2) - np.mean(vy[good])), 0.05)
    print(f"prior residual sd (log) {math.sqrt(tau2):.2f}")
    post_var = 1 / (1 / vy + 1 / tau2)
    post = post_var * (y / vy + mu / tau2)
    for x, m, pv in zip(rows, post, post_var):
        x["post"] = max(math.exp(m) - delta, 0.0)
        x["lo"] = max(math.exp(m - 1.96 * math.sqrt(pv)) - delta, 0.0)
        x["hi"] = math.exp(m + 1.96 * math.sqrt(pv)) - delta
        x["prior_share"] = pv / tau2  # 1 = all prior, 0 = all data

    # Ranks within length.
    for L in set(x["len"] for x in rows):
        grp = [x for x in rows if x["len"] == L]
        for i, x in enumerate(sorted(grp, key=lambda x: -x["combos"])):
            x["prob_rank"] = i + 1
        for i, x in enumerate(sorted(grp, key=lambda x: -x["post"])):
            x["play_rank"] = i + 1
    rows.sort(key=lambda x: -x["post"])

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "playability.tsv"), "w") as f:
        f.write("rank\talphagram\tlen\tplayability\tlo95\thi95\tprior_share\tprob_rank\tplay_rank_in_len\t"
                "source\tn_best\traw_a\traw_b\twords\n")
        for i, x in enumerate(rows):
            wl = ",".join(f"{w}:{wval[w][0] * PER:.3g}" for w in
                          sorted(x["words"], key=lambda w: -wval[w][0]))
            f.write(f"{i + 1}\t{x['alpha']}\t{x['len']}\t{x['post'] * PER:.4g}\t{x['lo'] * PER:.3g}\t"
                    f"{x['hi'] * PER:.3g}\t{x['prior_share']:.2f}\t{x['prob_rank']}\t{x['play_rank']}\t"
                    f"{x['src']}\t{x['n']}\t{x['a'] * PER:.4g}\t"
                    f"{'' if x['b'] is None else format(x['b'] * PER, '.4g')}\t{wl}\n")
    print(f"wrote {os.path.join(out, 'playability.tsv')}")

    def show(title, items, k):
        print(f"\n== {title}")
        for x in items[:k]:
            wl = " ".join(f"{w}({wval[w][0] * PER:.3g})" for w in
                          sorted(x["words"], key=lambda w: -wval[w][0])[:6])
            print(f"  {x['alpha']:<10} {x['post'] * PER:9.3g}  [{x['lo'] * PER:.3g}, {x['hi'] * PER:.3g}]"
                  f"  prob#{x['prob_rank']:<6} n={x['n']:<7} {x['src']:<2} {wl}")

    show("top alphagrams (equity pts per 1000 games)", rows, 25)
    for L in range(2, 11):
        show(f"length {L}", [x for x in rows if x["len"] == L], 10)


if __name__ == "__main__":
    main()
