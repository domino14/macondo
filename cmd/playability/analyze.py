#!/usr/bin/env python3
"""Final playability tables from `playability score` output.

Per alphagram (and per word inside it):
  1. Take the self-play estimate. Its standard error gets a floor of one
     unresolved hit (C points over G games), so a word never seen in self-play
     is "about zero, give or take one hit", not "exactly zero".
  2. Shrink toward a prior: regress log playability on features (length,
     log probability, anagram count, hooks, power tiles, vowels) over the
     alphagrams that are well measured, then combine each alphagram's own
     log estimate with the prediction, normal-normal. Alphagrams with little
     evidence (most 9+ letter ones) end up near the prior, which leans on
     probability.

The synthetic estimator (`playability synth`) is not used. With 10M games
self-play sees 99.9% of 7-8 letter racks, and the synthetic estimate's
standard error is unreliable when it has few hits, so combining the two
moved some well-measured power-tile racks by thousands of places. For 9+
it misses most of the value: 60% of 9-letter hits are hooks onto an 8
already on the board, which conditioned rack re-deals do not model.

Units: equity points per 1000 games (HastyBot static equity).

usage: analyze.py SCORE_DIR [OUT_DIR]
"""
import collections
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


def main():
    d = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else d
    meta = dict(l.rstrip("\n").split("\t", 1) for l in open(os.path.join(d, "meta.tsv")))
    G = float(meta["games"])
    floor_a = C / G
    delta = C / G  # log offset: one hit

    words = read_tsv(os.path.join(d, "words.tsv"))
    alphas = read_tsv(os.path.join(d, "alphas.tsv"))
    by_alpha = {}
    for w in words:
        by_alpha.setdefault(w["alphagram"], []).append(w)

    # Per-word values (no shrinkage: within-class shares).
    wval = {}
    for w in words:
        wval[w["word"]] = (fnum(w["a_play"]), math.hypot(fnum(w["a_se"]), floor_a))

    rows = []
    for r in alphas:
        a, se = fnum(r["a_play"]), math.hypot(fnum(r["a_se"]), floor_a)
        ws = by_alpha.get(r["alphagram"], [])
        hooks = sum(int(w["fhooks"]) + int(w["bhooks"]) for w in ws)
        rows.append(dict(alpha=r["alphagram"], len=int(r["len"]), combos=float(r["combos"]),
                         words=r["words"].split(","), n=int(r["n"]),
                         est=max(a, 0.0), se=se, hooks=hooks))

    # Features for the prior.
    def feats(x):
        L = x["len"]
        al = x["alpha"]
        f = [1.0]
        # Lengths 3-10 get their own level; 11+ share one with a linear slope,
        # since few long alphagrams are well measured.
        f += [1.0 if L == k else 0.0 for k in range(3, 11)]
        f += [1.0 if L >= 11 else 0.0, max(L - 10, 0)]
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

    # Ranks within length: every alphagram gets a distinct rank 1..N. Ties
    # (rare except among long racks that sit on the prior) go to the more
    # probable rack, then alphabetically.
    for L in set(x["len"] for x in rows):
        grp = [x for x in rows if x["len"] == L]
        for i, x in enumerate(sorted(grp, key=lambda x: (-x["combos"], x["alpha"]))):
            x["prob_rank"] = i + 1
        for i, x in enumerate(sorted(grp, key=lambda x: (-x["post"], -x["combos"], x["alpha"]))):
            x["play_rank"] = i + 1
    rows.sort(key=lambda x: (-x["post"], x["len"], x["play_rank"]))

    # Every lexicon alphagram must be ranked exactly once per length.
    want = collections.Counter(len(a) for a in by_alpha)
    have = collections.Counter(x["len"] for x in rows)
    assert want == have, f"alphagram counts differ: lexicon {want} vs ranked {have}"
    for L, n in have.items():
        assert sorted(x["play_rank"] for x in rows if x["len"] == L) == list(range(1, n + 1))

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "playability.tsv"), "w") as f:
        f.write("rank\talphagram\tlen\tplayability\tlo95\thi95\tprior_share\tprob_rank\tplay_rank_in_len\t"
                "n_best\twords\n")
        for i, x in enumerate(rows):
            wl = ",".join(f"{w}:{wval[w][0] * PER:.3g}" for w in
                          sorted(x["words"], key=lambda w: -wval[w][0]))
            f.write(f"{i + 1}\t{x['alpha']}\t{x['len']}\t{x['post'] * PER:.4g}\t{x['lo'] * PER:.3g}\t"
                    f"{x['hi'] * PER:.3g}\t{x['prior_share']:.2f}\t{x['prob_rank']}\t{x['play_rank']}\t"
                    f"{x['n']}\t{wl}\n")
    print(f"wrote {os.path.join(out, 'playability.tsv')}")

    # The study list: one row per alphagram, by length then playability rank.
    with open(os.path.join(out, "ranks.tsv"), "w") as f:
        f.write("len\trank\talphagram\tplayability\tlo95\thi95\tprior_share\tprob_rank\tn_best\twords\n")
        for x in sorted(rows, key=lambda x: (x["len"], x["play_rank"])):
            ws = ",".join(sorted(x["words"], key=lambda w: -wval[w][0]))
            f.write(f"{x['len']}\t{x['play_rank']}\t{x['alpha']}\t{x['post'] * PER:.4g}\t{x['lo'] * PER:.3g}\t"
                    f"{x['hi'] * PER:.3g}\t{x['prior_share']:.2f}\t{x['prob_rank']}\t{x['n']}\t{ws}\n")
    print(f"wrote {os.path.join(out, 'ranks.tsv')}: " +
          ", ".join(f"{L}s {have[L]}" for L in sorted(have)))

    def show(title, items, k):
        print(f"\n== {title}")
        for x in items[:k]:
            wl = " ".join(f"{w}({wval[w][0] * PER:.3g})" for w in
                          sorted(x["words"], key=lambda w: -wval[w][0])[:6])
            print(f"  {x['alpha']:<10} {x['post'] * PER:9.3g}  [{x['lo'] * PER:.3g}, {x['hi'] * PER:.3g}]"
                  f"  prob#{x['prob_rank']:<6} n={x['n']:<7} {wl}")

    show("top alphagrams (equity pts per 1000 games)", rows, 25)
    for L in range(2, 16):
        show(f"length {L}", [x for x in rows if x["len"] == L], 10)


if __name__ == "__main__":
    main()
