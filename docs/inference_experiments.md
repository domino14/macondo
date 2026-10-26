# Inference experiments

A lab notebook for rack inference (`rangefinder/`, `SIMMING_INFER_BOT*`):
what was run, on which code, what it showed, and what is queued. Newest
entries go at the bottom of the log; the idea queue is at the end.

Conventions:

- **Matches** are inference bot vs the same bot without inference
  (`SIMMING_INFER_BOT_NO_EG` vs `SIMMING_BOT_NO_EG`). The headline is the
  inference bot's **win share** (ties count half) with a **95% interval**.
  Paired runs (game pairs, from Aug 2026) report the paired interval from
  `autoanalyze`; earlier runs are unpaired and report a binomial interval over
  games.
- **Reads** are graded in **bits**: the lift, log2 of posterior / prior
  probability of the opponent's true leave. Replays use `cmd/inferlab` on a
  finished run's turn log; the run's own logged lifts are the baseline, and
  every default that is not under test is pinned (`-tau`, `-lambda`,
  `-value`, `-order`).
- **Decisions** are graded with `cmd/decidelab`: for logged positions the
  sim bot, a second copy of it (noise floor) and the inference bot choose a
  move, and a long sim that knows the opponent's true leave judges them.

## Log

### First runs and the tau sweep (Apr 11-25, 2026)

Code: branch `bayesian-inference-weights`, `c54ca819` for the sweep. Machine:
deb192 (2x EPYC 7K62, 192 threads). Config: 16 games at once, 12 sim threads
per bot, 5-ply sims for both bots, `inferenceMaxEnumeratedLeaves` 500,
inference on a 20 s wall clock. Unpaired games. Data:
`~/data/experiments/deb192/` (configs, games files, turn logs). Writeup:
blog post `2026-04-26-bayesian-inference-scrabble.md`.

The model: one 2-ply mini-sim per candidate leave (top 10 static plays plus
the play made, up to 200 iterations), likelihood = softmax over the log-odds
of win probability at temperature tau. Leave sets of at most 500 were
enumerated; larger ones were sampled from the prior, with a dedup bug that
kept only the first draw of each leave. No imputation: unsampled leaves got
no weight. Exchanges used the old rack sampler. The main sim used the
inferred racks through a sigmoid gate (inferred with probability
0.95 / (1 + exp(-(N - 10) / 5)), random otherwise).

| tau | games | inference bot win % |
|---|---|---|
| 0.03 (sim bot on 6 threads) | 7,236 | 50.59 ± 1.15 |
| 0.04 | 8,970 | 50.23 ± 1.03 |
| **0.05** | **9,725** | **52.00 ± 0.99** |
| 0.06 | 7,293 | 50.31 ± 1.15 |
| 0.07 | 9,489 | 50.14 ± 1.01 |
| 0.1 | 8,043 | 50.44 ± 1.09 |
| oracle (bot told the opponent's true leave) | 7,698 | 53.34 ± 1.11 |

The blog's 51.89% ± 0.77 for tau 0.05 pools in about 6,700 more games whose
file is lost. The oracle row is the ceiling at 5 plies: perfect reads are
worth about 3.3 points of win share.

Re-examined Oct 16 (see below): no artifact found.

### Leave imputation and the tau fit (Jul 24, 2026)

- The sigmoid gate was replaced by a complete posterior over every feasible
  leave: measured leaves keep prior x measured likelihood; unmeasured ones get
  likelihoods imputed from containment-marginal lifts with Möbius inversion
  (`rangefinder/imputation.go`, `docs/leave_imputation.md`). The dedup bug
  was fixed along the way. Calibration moment-matched the same day.
- tau MLE from 2,000 real woogles games, fitted on BestBot's own moves
  (`cmd/taufit`, `docs/development/tau_calibration.md`): tau* = 0.119
  overall, strongly phase dependent (bag 46+: 0.026, 31-45: 0.040, 11-30:
  0.095, 2-10: 0.92).

### Game pairs (Aug 12, 2026)

`a693b93e`: paired autoplay, both games of a pair share a bag, so tile luck
largely cancels (pairing is worth about 3x the games). First paired run
(`infer-v-sim-200`, 400 pairs, budget 200): 50.69%.

### The 5,000-pair run (Sep 2026)

tau 0.05, budget 200 leaves, 5-ply, lambda 10. **50.95% ± 0.70**, +1.63 ±
1.26 points per game (score p = 0.011). Reads +0.30 bits per inference on
average.

- Within a run, read quality does not predict the result: catastrophic reads
  mostly come from exchanges and one-tile plays by an opponent who keeps a
  strong leave, and those games are won *more* often.
- Exchanges: 62% of exchange reads ruled out the true leave, 83% of all
  ruled-out reads. Cause: the old rack sampler, no posterior.

### Replay findings with inferlab (Sep 11-18, 2026)

On six-tile leaves (one-tile plays) from `infer-v-sim-200`, budget 200:

- The likelihood is fine: forcing the true leave into the measured set gives
  +7.0 bits; the problem is the truth never gets measured (coverage).
- Random proposals: +0.26 ± 0.67 (n.s.). Exploration floor: +0.75, tail only.
- Shrinkage (lambda 100-300) +3.1 bits on six-tile leaves, by flattening,
  not by fixing the ordering; it costs -0.26 on three-tile leaves. Reverted to
  lambda 10 everywhere (collaborator: tune shrinkage last).
- Order-4 marginals: negative at any shrinkage (too few observations per
  term at budget 200).
- Leave-value term, value-first with lambda 30: six-tile +2.77 [+2.04,
  +3.52] at fixed tau 0.05, and the first variant to improve the ordering;
  hurts three- to five-tile leaves. Adopted for six-tile leaves only
  (`204ca75c`, PR #531).
- Lesson: a default changed mid-program (the tau schedule) contaminated a
  batch of replays; pin every default that is not under test.

### Tau table, exchange model (Oct 12, 2026)

Branch `claude/tau-table`, PR #537 (merged).

- Tau by bag x tiles played: the tiles-played effect is the bag in disguise.
  A smooth bag curve (`TauFitted`) is the best held-out fit. Replays vs fixed
  0.05: bag 2-7 +3.10 [+1.87, +4.46], bag 8-20 +0.91 [+0.38, +1.50], null
  earlier.
- One-tile plays, tau x2 at bag > 20: +1.52 [+0.99, +2.13].
- Exchanges: a leave-value model, P(kept = L) ∝ prior · exp(β · KLV(L)) with
  no sims (`ExchangeValue`), +0.43 bits vs the old sampler's -1.71; ruled-out
  exchange reads went from 63% to 0%.
- Defaults flipped to all three (`270028a2`).

### Tau-table match (Oct 13-14, 2026)

`infer-v-sim-12500`, Go engine, tau-table defaults, budget 200, 5-ply, 192
threads on deb192. Stopped at 2,194 pairs by a power cut: **51.09% ± 1.02**,
+2.27 ± 1.93 points per game. Not distinguishable from the 5,000-pair run.

### decidelab: do changed moves win? (Oct 14, 2026)

400 positions from the 5,000-pair run. Inference changes the move on 16.2% of
turns, against 14.8% for a second copy of the sim bot (sim noise alone).
Gain +0.08 win points per turn [-0.07, +0.28]; all of it at bag 2-20 (+0.24
to +0.44 per turn), about zero at bag 21+. About 12 sims a game x 0.08 is
about +1 point per game, consistent with the matches' 51%.

### Mini-sim noise on the MAGPIE engine (Oct 14, 2026)

Branch `claude/infer-minisim-magpie`. MAGPIE makes inference 2.8x faster with
the same reads. A 200-rollout mini-sim's log-likelihood has an SD of about
1.5 bits, but quieter mini-sims do not improve reads (1,000 equal rollouts:
+0.00 [-0.23, +0.21] at 5.4x the cost). More leaves do: budget 500 +0.26
[+0.04, +0.55], budget 1000 +0.60 [+0.35, +0.89]. Coverage, not per-leaf
noise, is the limit.

### MAGPIE match, budget 500 (Oct 15-16, 2026)

`infer-v-sim-magpie`, branch `claude/magpie-movegen` (`1169b8c1`), budget
500, mini-sim cap 300, otherwise as the tau-table match. Four launches died
within an hour (deb192 power trips); the fifth, on the owner's default BIOS
settings with boost on, ran 23 h without trouble at ~235 games/h, about 1.5x
the Go engine. Stopped at 2,635 pairs: **50.73% ± 0.87**, +2.55 ± 1.47
points per game. The bigger budget made no measurable difference.

### Why fewer pairs diverge (Oct 16, 2026)

16% of the MAGPIE run's pairs were identical games against 7% in the
tau-table run. The first differing turn of a pair is a turn where the two
bots chose differently in the same position; sims are deterministic (0
changes on turns without inference), so every change is inference's. Rate of
changed moves by read quality:

| lift bits | tau-table (Go) | MAGPIE, budget 500 |
|---|---|---|
| < 0 (wrong read) | 16.0% | 11.0% |
| 0-0.5 | 13.6% | 6.2% |
| 0.5-1 | 15.8% | 8.6% |
| 1-2 | 15.9% | 10.0% |
| 2-4 | 17.8% | 13.2% |
| ≥ 4 | 19.8% | 20.0% |
| all inferring turns | 15.6% | 10.3% |

On the Go engine inference changed the move about as often with or without
information: mostly noise flips of near-tied plays. On MAGPIE the rate rises
with read quality, and strong reads are more common (7.1% of inferring turns
vs 4.7%). Divergent pairs alone: 51.1% in both runs, so the noise flips were
roughly harmless and removing them adds nothing; the better reads do not show
up in wins yet. Scripts: `divergence.py`, `pairvalue.py` (not committed).

### The tau 0.05 spike, re-examined (Oct 16, 2026)

Checks on the April files: every sweep run inferred about 205 racks per turn
on 80% of turns (0.03: 223), so the wall clock gave 0.05 no extra effort;
first move was 50/50 in all runs; the 0.05 edge was steady in all four
quarters of its run (52.0, 51.4, 52.7, 51.9); the only code change during the
sweep (`696d960d`, a sim speedup) came after the 0.05 run. The spike is
about 3.4 standard errors above its neighbours, roughly 1 in 400 by chance
across six tau values, and implausible as a real effect (0.04 and 0.06 barely
change the read). Most likely a fluke. A replication (rebuild `c54ca819`,
same config, ~10,000 unpaired games, ~2 days on deb192) is parked until the
queue below runs dry.

### 2-ply baseline (launched Oct 16, 2026 22:12)

`infer-v-sim-2ply` on deb192: as the MAGPIE match but 2-ply midgame sims for
both bots (`minSimPlies 2`; the late-bag sims and the endgame are unchanged)
and budget 200. Same seed (20261012), 12,500 pairs. The baseline for trying
ideas fast before confirming them at 5 plies. Result pending.

### decidelab ceiling and static ranks (launched Oct 17, 2026 01:13)

decidelab gained an `oracle` chooser, the same sim bot told the opponent's true
leave (what a perfect read would choose), and logs each choice's static rank.
Queued on deb192 (`~/macondo/lab/queue1.sh`, 48 threads beside the match):

- `dl-2ply-oracle`: 2,000 positions from the 2-ply baseline's turn log
  (snapshot of 54,363 lines), 2-ply deciders, budget 200, judge 3,000
  iterations at 5 plies against the true leave.
- `dl-5ply-oracle`: 1,000 positions from the 5,000-pair run, 5-ply deciders.

**2-ply result (Oct 17 02:54, 2,000 positions, 100 min).** Gain over `none`
in win points per turn, 95% bootstrap intervals:

| bag    | n     | none2 (noise) | infer                  | oracle                 |
|--------|-------|---------------|------------------------|------------------------|
| 2-7    | 156   | 0.6% / +0.00  | 9.6% / -0.30 [-0.87, +0.07] | 50.6% / **+2.14** [+1.36, +3.06] |
| 8-20   | 323   | 2.5% / +0.02  | 6.2% / +0.13 [-0.08, +0.36] | 34.7% / **+0.67** [+0.37, +1.02] |
| 21-50  | 725   | 2.2% / +0.00  | 10.6% / -0.00 [-0.10, +0.08] | 25.8% / **+0.33** [+0.24, +0.43] |
| 51+    | 796   | 2.1% / +0.00  | 10.9% / +0.02 [-0.05, +0.07] | 26.6% / **+0.29** [+0.23, +0.36] |
| all    | 2,000 | 2.1% / +0.01  | 9.9% / +0.00 [-0.06, +0.07] | 29.5% / **+0.51** [+0.42, +0.61] |

(cells: share of positions where the move changed / gain)

- At 2 plies, inference gains nothing (+0.00), while a perfect read is worth
  +0.51 points per turn: about +6 points a game at ~12 sims a game. The
  ceiling is large, so there is room to gain; how inference reads and uses
  the leave is the limit.
- The ceiling grows as the bag empties (+0.29 early, +2.14 at bag 2-7), and
  there inference is timid or wrong. At bag 8-20 it changes 6% of moves
  against the oracle's 35%. At bag 2-7 it changes 10%; only 4 of those 15
  changes match the oracle's, and the gain is negative.
- Agreement: when inference changes the move, it picks the oracle's move half
  the time at bag 21+ (82 of 164) and 8-20 (11 of 20). It follows only 21%,
  10% and 5% of the oracle's changes at 21+, 8-20 and 2-7.
- 2-ply sims are quiet: a second copy of the sim bot changes 2.1% of moves
  (the 5-ply run: 14.8%).
- Static rank: the oracle's move ranks 51-100 in 1.8% of positions and the
  inferer's in 0.6%; neither went past 100 (the candidate list is 100 deep).
  A deeper list is worth little for now (idea 5).
- Check: corpus games are unseeded, so the judge never shares the deciding
  sims' draws (`none` and `none2` would otherwise agree everywhere).

**5-ply result (Oct 17 05:00, 1,000 positions from the 5,000-pair run,
122 min).**

| bag    | n     | none2 (noise) | infer                  | oracle                 |
|--------|-------|---------------|------------------------|------------------------|
| 2-7    | 81    | 1.2% / +0.03  | 6.2% / +0.04 [+0.00, +0.10] | 49.4% / **+1.38** [+0.65, +2.22] |
| 8-20   | 171   | 3.5% / +0.04  | 8.2% / +0.10 [+0.01, +0.20] | 43.3% / **+0.78** [+0.54, +1.05] |
| 21-50  | 358   | 3.6% / -0.00  | 9.2% / +0.02 [-0.06, +0.10] | 27.9% / **+0.37** [+0.26, +0.50] |
| 51+    | 390   | 4.4% / +0.03  | 10.3% / +0.02 [-0.05, +0.09] | 24.9% / **+0.35** [+0.25, +0.47] |
| all    | 1,000 | 3.7% / +0.02  | 9.2% / +0.04 [-0.01, +0.08] | 31.1% / **+0.52** [+0.42, +0.62] |

- The ceiling is the same at 5 plies (+0.52 vs +0.51). Inference captures
  about 7% of it (+0.04), against none at 2 plies. Either way it is far from
  the ceiling, in every phase.
- On the MAGPIE engine (BAI sims), 5-ply sim noise is 3.7% changed moves,
  down from 14.8% on the Go engine (Oct 14).
- Static rank: the oracle's move ranks 51-100 in 3.2% of positions (max 99),
  so with perfect reads the candidate list starts to matter; for today's
  reads (1.5%) it does not.

**Gain by read quality (2-ply positions joined to the match's lift bits).**
At bag 21+, weak reads lose and strong ones gain:

| lift     | n   | infer changed | infer gain             | oracle gain |
|----------|-----|---------------|------------------------|-------------|
| < 0      | 312 | 10.6%         | -0.09 [-0.24, +0.03]   | +0.38       |
| 0-1      | 485 | 7.6%          | -0.03 [-0.15, +0.05]   | +0.25       |
| 1-2      | 354 | 13.0%         | +0.06 [-0.00, +0.13]   | +0.36       |
| 2-4      | 279 | 11.5%         | +0.10 [+0.00, +0.24]   | +0.27       |
| 4+       | 91  | 17.6%         | +0.08 [-0.06, +0.22]   | +0.29       |

Inference acts about as often on weak reads as on strong ones, and the weak
ones cost about what the strong ones earn. Lift needs the true leave, so a
gate needs a proxy the bot has: `RangeFinder.InfoBits`, the posterior's KL
from the tile-count prior, now logged by decidelab (`infoBits`) and usable as
a gate (`InferenceMinInfoBits`; decidelab variant key `minbits`).

### Late-bag variants (Oct 17, 2026 05:00-08:30)

`dl-2ply-late-variants`: 1,000 positions with bag <= 20 from the 2-ply
baseline, 2-ply deciders. Gain over `none`, and paired difference from the
default inferer (`infer`: fitted tau, budget 200):

| chooser | changed | gain vs none          | minus infer, 2-7      | minus infer, 8-20     | minus infer, all      |
|---------|---------|-----------------------|-----------------------|-----------------------|-----------------------|
| infer   | 8.5%    | +0.17 [+0.05, +0.30]  |                       |                       |                       |
| sharp (tau x 0.5) | 11.8% | +0.21 [+0.09, +0.35] | +0.07 [-0.06, +0.19] | +0.03 [-0.05, +0.14] | +0.05 [-0.03, +0.13] |
| b1000 (budget 1,000) | 8.4% | +0.19 [+0.07, +0.32] | +0.00 (identical) | +0.04 [-0.04, +0.14] | +0.02 [-0.02, +0.08] |
| **t05 (fixed tau 0.05)** | 18.8% | **+0.38** [+0.18, +0.59] | **+0.38** [+0.02, +0.76] | +0.10 [-0.02, +0.24] | **+0.22** [+0.05, +0.39] |
| oracle  | 38.3%   | +1.10 [+0.81, +1.41]  | +1.64                 | +0.44                 | +0.94                 |

- **A sharp tau late in the game more than doubles inference's gain** there.
  The fitted curve is very flat late (tau 2.5 at bag 2, 0.45 at bag 7, 0.075
  at bag 20), so x 0.5 barely moves it. Fixed 0.05 changes 19% of moves, and
  38% of its changes are the oracle's (71 of 188; default 30 of 85).
- This reverses the Oct 12 replay verdict, where the fitted curve gained
  about 3 bits on the true leave at bag 2-7. Bits measure calibration;
  decisions want a confident read. The fit says the opponent's late plays
  look noisy to a 2-ply mini-sim, which is true, but discounting them that
  much throws away what they do say.
- A bigger budget does nothing late: with few unseen tiles, every leave is
  already enumerated (identical choices at bag 2-7).
- (Bag 2-7 here gave infer +0.28 [+0.08, +0.53] against -0.30 [-0.87,
  +0.07] in the first run on 156 positions: different samples, wide
  intervals.)

### A confidence gate? No (Oct 17, 2026 08:31-10:39)

`dl-2ply-infobits`: 3,000 positions (all bags) from a later snapshot of the
2-ply baseline, with each read's `infoBits`. Overall: infer +0.10 [+0.06,
+0.15] per turn, oracle +0.47 [+0.39, +0.55]. (The first 2,000-position
sample gave infer +0.00 [-0.06, +0.07]; pooled, about +0.06.)

Gain by read strength (infoBits) and the offline gate (gain kept where
infoBits >= threshold, 0 elsewhere):

| bag  | infoBits quartiles | gain by infoBits band | gate 0 / 0.5 / 1 / 2 bits |
|------|--------------------|-----------------------|---------------------------|
| <=20 | 0.13 / 0.30 / 0.71 | +0.03 (<0.25), +0.29, +0.07, +0.30 (1-2), +1.79 (2-4, n=26) | +0.19 / +0.11 / +0.10 / +0.06 |
| 21+  | 0.54 / 1.03 / 1.79 | +0.01 (<0.25), +0.01, +0.06, +0.08, +0.08, +0.33 (4+) | +0.07 / +0.07 / +0.05 / +0.03 |

No band loses: weak reads rarely change the move (3-6%) and then gain about
nothing. Every threshold gives up gain. The losses tied to low lift come
from confident reads that are wrong, which the bot cannot see. Gate dropped;
`InferenceMinInfoBits` stays as an option, default off.

Baseline check at 10:40: `infer-v-sim-2ply` = **50.85% ± 0.70** after 3,899
pairs (spread +1.73 ± 1.16), level with the 5-ply runs.

### Late tau sweep (Oct 17, 2026 10:39-13:44)

`dl-2ply-late-tau`: another 1,000 positions with bag <= 20. Paired
difference from the default inferer, win points per turn:

| fixed tau | 2-7                   | 8-20                  | all                   |
|-----------|-----------------------|-----------------------|-----------------------|
| 0.025     | +0.15 [-0.50, +0.68]  | +0.00 [-0.15, +0.14]  | +0.06 [-0.21, +0.30]  |
| **0.05**  | **+0.36** [+0.09, +0.70] | +0.01 [-0.13, +0.15] | **+0.15** [+0.02, +0.31] |
| 0.1       | +0.25 [-0.03, +0.58]  | -0.00 [-0.12, +0.12]  | +0.10 [-0.03, +0.24]  |
| 0.2       | +0.22 [-0.01, +0.52]  | -0.07 [-0.17, +0.00]  | +0.05 [-0.06, +0.18]  |

0.05 is best again. Pooled with the variants run (2,000 positions): **+0.37
[+0.14, +0.61] at bag 2-7**, +0.05 [-0.04, +0.15] at 8-20, +0.18 [+0.08,
+0.30] overall. 0.025 is no better and much noisier.

New option `inferenceTauCap` (autoplay config) / `InferenceTauCap` (bot) /
`cap` (decidelab): caps the temperature, so a cap of 0.05 keeps the fitted
values early (below 0.05 from bag ~29) and changes only the late reads.

Expected size in a match: about +0.37 x ~1.5 inferring turns at bag 2-7 per
game, plus a little at 8-30: roughly +0.6 win points per game. Too small to
see in two independent 5,000-pair matches (difference SE ~0.45), so the
match reuses the baseline's seed and is compared pair by pair.

Next: `infer-v-sim-2ply-cap05`, the baseline match with `inferenceTauCap
0.05`, after a check that the new binary reproduces the baseline's pairs.
Interim at 2,246 pairs: cap minus baseline **-0.23 ± 0.48** points, so the
+0.6 is already outside the interval.

### How close do reads get to the oracle? (set up Oct 17, 2026)

Where the true leave ranks, from the 2-ply baseline's turn log (18,613
inferring turns; median posterior size 100 leaves at bag 2-7, 3,346 at 51+):

| bag | truth 1st | top 5 | top 20 | median rank | median rank, share of list | lift bits |
|---|---|---|---|---|---|---|
| 2-7 | 7.0% | 25.8% | 56.2% | 17 | 21% | 0.38 |
| 8-20 | 5.6% | 18.3% | 38.2% | 39 | 10% | 0.94 |
| 21-50 | 4.6% | 15.3% | 30.2% | 80 | 6% | 1.03 |
| 51+ | 4.3% | 14.0% | 28.3% | 117 | 4% | 0.90 |

Lift scores only the exact leave: weight on AENRS when the truth is AENST
counts the same as weight on QVWXZ. New measures (`b3d8de19`):

- **Soft lift**: log2(sum post(L)K(L,T) / sum prior(L)K(L,T)), the lift with
  a similarity kernel K(L,T); with K = exact match it is the lift. Overlap
  kernel 2^-(tiles of T missing from L); value kernel exp(-d/3),
  d = |V(L) - V(T)| + 1 point per missing tile (KLV leave values).
- **Decision regret** (`decidelab -agree N`): sample N leaves from the
  posterior and N from the prior, sim each as the opponent's known rack over
  the top 20 plays plus every chooser's pick (300 iterations), and price the
  pick with a 2,000-iteration sim against the true leave. Regret = win points
  given up there. The truth's own short sim gives the noise floor, so
  closed = (prior - posterior) / (prior - truth) is the share of the way from
  no read to a perfect one, in the bot's own currency. Similar leaves score as
  similar exactly when they lead to the same play.
- Mini-sim knobs as decidelab variant keys: `iters` (200), `plays` (10 top
  static plays beside the observed one), `mplies` (2).

`readlab.py` reports all of these by bag, plus how well each tile-level
measure predicts the position's regret gain and judge gain (Spearman), to
pick an everyday score that tracks decisions.

Smoke (12 positions, local): runs end to end. One position: posterior samples
gave up 1.6 win points against 3.3 for prior samples. A truth sim at 300
iterations disagreed with itself in 4 of 6 positions, hence regret rather
than move agreement.

Queued (`lab/queue6.sh`, after cap05): `dl-knobs-late` (1,000 positions,
bag <= 20) and `dl-knobs-early` (1,000, bag >= 21), 2-ply deciders: infer,
iters 50/800, plays 5/20/40, mini-sim plies 1/3, budget 1000, tau 0.05,
oracle; agreement with 16 samples per source, all arms paired.

### Cap match result (Oct 17-18, 2026)

`infer-v-sim-2ply-cap05`, stopped at 4,868 pairs: **50.47% ± 0.62**. Against
the baseline pair by pair (4,862 common pairs, 60.7% identical):
**-0.07 ± 0.33**. The cap was in force (28,564 inferring turns at tau 0.05,
about 2.9 per game). No measurable gain.

That is consistent with decidelab once the right comparison is made: the
cap changes infer into t05 only late, and t05 minus infer is about +0.26
win points per turn at bag 2-7 and about 0 at 8-20 (table below). With
about one such turn per game, the expected gain is about +0.25 per game, at
the top of the interval. The +0.6 estimate above double counted.

### Knob sweep: how close reads get to the oracle (Oct 18, 2026)

`dl-knobs-late` (1,000 positions, bag <= 20) and `dl-knobs-early` (1,000,
bag >= 21), 2-ply deciders, all arms on the same positions, decision
agreement with 16 leaves per source. Regret in win points per turn: what
the moves that leaves drawn from each source lead to give up against the
true leave. Prior 1.29, truth (noise floor) 0.10.

| arm | ms/turn | lift | regret closed 2-7 | 8-20 | 21+ | all | judge gain (all) |
|---|---|---|---|---|---|---|---|
| infer (200 iters, 10 plays, 2 plies, budget 200) | 81k | 0.88 | 15% | 12% | 17% | 15% | +0.10 |
| iters 50 | 40k | 0.58 | 18% | 13% | 14% | 16% | +0.10 |
| iters 800 | 171k | 0.95 | 16% | 12% | 11% | 14% | +0.12 |
| plays 5 | 54k | 0.85 | 16% | 13% | 10% | 14% | +0.12 |
| plays 20 | 131k | 0.82 | 14% | 12% | 10% | 13% | +0.11 |
| plays 40 | 230k | 0.85 | 16% | 12% | 11% | 14% | +0.11 |
| mini-sim 1 ply | 47k | -0.16 | 7% | 0% | 2% | 4% | -0.01 |
| mini-sim 3 plies | 109k | 0.68 | 14% | 9% | 11% | 12% | +0.11 |
| budget 1000 | 207k | 1.24 | 17% | 11% | 14% | 15% | +0.11 |
| **tau 0.05** | 76k | 0.75 | **40%** | **22%** | 10% | **30%** | **+0.17** |

(ms/turn is single-threaded wall time of the whole decision; 21+ pools
21-50 and 51-100.)

- **Reads close about 15% of the gap** between no read and a perfect one.
- **Mini-sim effort does nothing**: iterations 50-800, plays 5-40, 3 plies
  and budget 1,000 all land at 12-18%. Iterations 50 or plays 5 match the
  default at half the time. One-ply mini-sims break the read.
- **Only sharpness moves it**: tau 0.05 doubles the closed share overall and
  reaches 40% at bag 2-7 (regret 2.07 vs 2.85 for infer, 3.32 prior). Early
  (bag 21+) it is the same as fitted, which is already near 0.05 there.
- **Lift points the wrong way late**: tau 0.05's lift at bag 2-7 is -0.28
  against +0.36 for fitted, yet its decisions are much better. The soft
  lifts get the sign right (overlap 0.26 vs 0.08, value 0.38 vs 0.17).
- **Per-position prediction is weak for every tile metric** (Spearman with
  regret gain 0.10-0.18; soft overlap a little ahead of lift). Use regret
  itself to compare settings, soft overlap as a cheap side score.

### The April spike, replicated and dissected (Oct 18, 2026)

`april-tau05-replica` on deb192: c54ca819 rebuilt, the April config verbatim,
unpaired. At 2,755 games (Oct 18 22:04): **49.62% ± 1.86**, flat across
quarters (49.5, 50.4, 49.8, 48.8). Same code, config, data files (md5 against
the old disk, now on deb128) and CPU generation; inferred racks per turn
match April (median 218 vs 215, same by bag).

The old disk also held a June rerun on master `13b729b` (pre-imputation):
`tau0.05-v-sim-redux`, 6,063 games, **52.00% ± 1.25**. A crash ended it; the
hand-fixed games file matches the turn log (6,050 exact, 13 differ by the
end-of-game rack penalty, the 16 missing games were mid-game).

Fingerprints (points/turn, exchanges, game length, tiles and leave per move)
are identical across all runs. The two 52% runs look like luck of two
different kinds:

| run | games | win % | spread | bingo diff/game | win %, equal bingos |
|---|---|---|---|---|---|
| tau 0.04 | 8,970 | 50.23 | +0.44 ± 2.10 | +0.016 | 50.61 |
| tau 0.05 April | 9,725 | 52.00 | +4.21 ± 2.03 | +0.069 | 51.22 |
| tau 0.06 | 7,293 | 50.31 | +0.42 ± 2.31 | +0.026 | 50.56 |
| tau 0.05 June | 6,063 | 52.00 | +1.61 ± 2.56 | +0.027 | 50.78 |
| oracle (April) | 7,698 | 53.34 | +6.98 ± 2.30 | +0.050 | 54.32 |

April's edge came with bingo luck beyond the oracle's; June's 52% has the
spread of a 50.3% run. The oracle is real on every measure. Unpaired autoplay
caps a turn at 30 s, inference included, so the infer bot's sim gets ~9 s
(a spike of cut-off sims at 8.5-9.5 s) -- in April as now, so not the
difference. Paired mode has no midgame deadline. Scripts (scratchpad, not
committed): `fingerprint.py`, `gamestats.py`, `infcount.py`.

**Replica at 10,000 games (Oct 20): 50.36% ± 0.98.** The two old runs pooled
(52.00) minus the replica = 1.64 ± 1.25 (p ~0.01); all three pooled 51.4 ± 0.6.

Last-ditch setup check, nothing found:
- June redux binary and replica binary: both go1.26.1. Data dirs md5-identical.
  config.yaml differs only in ttable-mem-fraction (no endgame in NO_EG bots).
- Old bash history: tau 0.03, 0.05 and 0.1 ran on one binary built at the pull
  just before 0.03 (Apr 16 00:57), i.e. c54ca819 plus a test-only commit, the
  replica's code. 0.07/0.04/0.06 ran on a later build (after a master merge).
- Throughput ~203 games/h in every April run, 208 redux, 205 replica.
- April is exactly 5,057/9,725 = 52.0000%, redux 52.004%; no shared games.

Inside the games:
- Tile-draw luck (what each side drew, valued by regression, composition only):
  April +0.30, redux -0.05, replica -0.01. Adjusted: 51.70 / 52.06 / 50.38.
- Per-phase play profile (points, tiles, exchanges, bingos, racks inferred):
  the same in every run.
- Where the spread came from: April +3.1/game at bag>40, redux +1.5 at bag 0,
  replica nothing; the oracle gains in every phase (+2.3 / +1.3 / 0 / +2.7).

The only untested item: the June redux binary (deb128 `~/macondo/bin/shell`)
still exists, the one surviving binary from a 52% run.

### Oracle ceiling at 2 and 5 plies (launched Oct 18, 2026 22:57)

On deb128 (EPYC 7702P, 128 threads), the current engine (deb192's cap05
build): `SIMMING_BOT_NO_EG` told the opponent's true leave vs the same bot,
NO_EG both sides as in the inference baselines, 7,500 pairs each (±0.5),
seed 20261020, 2-ply then 5-ply. Smoke (64 pairs): 2-ply 53.1% ± 6.9 at
~545 games/h, 5-ply 53.1% ± 6.5 at ~150 games/h (both batch-tail bound).
ETA: 2-ply ~1 day, 5-ply ~3-4 days after.

**2-ply result (Oct 19 14:25, 927 min): 53.51% ± 0.60** paired over 7,500
pairs, spread +8.33 ± 1.06; quarters 54.3 / 53.9 / 52.0 / 52.9 at 5,130
pairs, so no drift. Against the 2-ply inference baseline (50.55% ± 0.62)
inference has about 3.0 points of headroom and captures ~0.5 of it, about
15%, which matches decidelab's "reads close ~15% of the gap". 5-ply started
14:25 (124 pairs at 15:44: 56.5% ± 4.9).

## Running log

What is on deb192 and when it changed. Results go in the sections above.

- Oct 16 22:12: `infer-v-sim-2ply` match launched (192 threads).
- Oct 17 01:13: `queue1.sh` (decidelab oracle runs) launched, 48 threads.
- Oct 17 02:54: `dl-2ply-oracle` done (above). `dl-5ply-oracle` briefly
  stopped over a suspected shared seed (false alarm), relaunched 02:58 as
  `queue2.sh`.
- Oct 17 03:05: `queue3.sh` queued behind it: `dl-2ply-late-variants`.
- Oct 17 05:00: `dl-5ply-oracle` done (above); `dl-2ply-late-variants`
  started (~7 h: four inferring arms).
- Oct 17 05:15: `queue4.sh` queued behind it: `dl-2ply-infobits` (3,000
  positions from a fresh snapshot, `decidelab-v3`).
- Oct 17 08:31: `dl-2ply-late-variants` done (above); `dl-2ply-infobits`
  started.
- Oct 17 08:40: `queue5.sh` queued behind it: `dl-2ply-late-tau`.
- Oct 17 10:39: `dl-2ply-infobits` done (gate: no); `dl-2ply-late-tau` started.
- Oct 17 13:44: `dl-2ply-late-tau` done (tau 0.05 best).
- Oct 17 13:50: `infer-v-sim-2ply` stopped at 4,863 pairs: **50.55% ± 0.62**
  (spread +1.46 ± 1.02). Old binary kept as `bin/shell-1169b8c1`.
- Oct 17 13:57: `repro-check`: new binary (f4e4300b), no cap, first 192
  pairs of the baseline's seed. All 192 pairs identical to the baseline (384 of 384
  scores), so the new binary plays as the old one.
- Oct 17 14:31: `infer-v-sim-2ply-cap05` launched: the baseline's config and
  seed plus `inferenceTauCap 0.05`, 192 threads. Compared pair by pair with
  `infer-v-sim-2ply` once it passes 4,863 pairs (~14 h).
- Oct 17 20:36: cap05 interim, 2,246 pairs: 50.66% ± 0.93; minus baseline
  -0.23 ± 0.48 (paired).
- Oct 17 21:08: `queue6.sh` armed: stops cap05 at 4,863 pairs, then
  `dl-knobs-late` and `dl-knobs-early` (`bin/decidelab-v4`, b3d8de19,
  180 at once).
- Oct 18 01:00: cap05 interim, 3,950 pairs: minus baseline -0.10 ± 0.37.
- Oct 18 01:23: `queue7.sh` armed: after the sweep, the April tau 0.05
  replication (user's call): `c54ca819` in `~/macondo-april`, April config
  verbatim, unpaired, alone on the machine, up to 40,000 games (~38,000 for
  ± 0.5, ~8 days at April's 207 games/h). First look at ~10,000 games:
  below 51.0% means 52 is outside the interval, stop. 2-game smoke passed
  (tau 0.05 in use, inferences fed to the sim).
- Oct 18 03:21: queue6 stopped cap05 at 4,868 pairs: -0.07 ± 0.33 vs
  baseline (paired).
- Oct 18 06:08 / 08:27: `dl-knobs-late` / `dl-knobs-early` done.
- Oct 18 08:27: `april-tau05-replica` started (~200 games/h, as in April).
- Oct 18 22:04: replica 49.62% ± 1.86 at 2,755 games; June redux found (52.00%); fingerprints say both 52% runs were luck.
- Oct 18 22:57: `oracle-2ply` then `oracle-5ply` launched on deb128 (7,500 pairs each).
- Oct 19 14:25 (deb128): `oracle-2ply` done, 53.51% ± 0.60 (above); `oracle-5ply` started.
- Oct 19 15:44 (deb192): replica 6,374 games, 50.55% ± 1.23; running on to 10,000.
- Oct 20 12:14 (deb192): replica stopped, FINAL 10,572 games = **50.45% ± 0.95**. `june-redux-rerun` started: the June redux binary itself (13b729b, md5 e11a50ca), redux config, old-disk data, in `~/macondo-june`.
- Oct 21 15:49 (deb192): June rerun stopped, FINAL 5,756 games = **49.90% ± 1.29**. The machine hard-crashed as the load fell from 190 to 0 (journal stops with no error); back after a power cycle. Rule: never let deb192 go idle all at once.
- Oct 21 16:04 (deb192): `infer-v-sim-2ply-10k` started: 2-ply inference baseline, 10,000 pairs, oracle seed and oracle build (md5 410b0a42), budget 200, defaults. ETA ~26 h, ±0.43.

## Idea queue

1. **Ceiling by phase (decidelab).** Add an oracle chooser (true leave) to
   decidelab. Where perfect reads gain little, stop inferring; where they gain
   much more than our reads, read quality is the limit there.
2. **Spend the leaf budget where it pays.** 1,000-2,000 leaves at bag ≤ 20,
   fewer or none early, where inference gains about nothing.
3. **A stronger sim on strong reads.** Sim noise flips 14.8% of moves; more
   iterations or plies when the read is confident.
4. **Exchange-read turns.** Check in decidelab that the leave-value exchange
   model gains on them.
5. **More candidate plays.** Both bots already sim 100 candidates, each with
   at least 500 rollouts. First log the static rank of the plays inference
   picks; if they sit near 100, try 200.
6. **tau 0.05 replication**: queued (queue7). If it holds at ~52%, port its
   pieces one at a time into the current code, each a paired match.

7. **Leave-quality rule** (Oct 21): read the opponent's leave
   *quality* from the tiles played and the score of their last move (2 tiles
   for 10 points: they kept 5 good tiles), and give the sim opponent strong
   racks accordingly, sampling several racks within one sim. We do this only
   for exchanges (`exchange_value.go`) and one-tile plays (`SetValueTerm`).
   Test: fit beta(tiles, score) by max likelihood on true leaves in the turn
   logs, compare regret in decidelab, then a 2-ply paired match (baseline
   50.55, ceiling 53.51). It costs almost nothing per turn, so the sim keeps
   its full budget.
### Oct 22 01:00: oracle 5-ply final, leave-quality reads (dl-lq)

- **oracle-5ply FINAL** (deb128, 7,500 pairs): **53.73% ± 0.61**, same as
  2-ply (53.51). Depth does not raise the perfect-read ceiling.
- **dl-lq** (1,500 positions from the cap05 2-ply log, agree 16): mean
  decision regret per read (lower is better; truth = 0):

  | read | all | 1 tile (39) | 2 tiles (221) | 3 (411) | 4+ (783) |
  |---|---|---|---|---|---|
  | prior | 0.898 | 3.93 | 1.82 | 0.98 | 0.43 |
  | infer (ours) | 0.793 | 3.48 | 1.57 | 0.87 | 0.39 |
  | lq x1 / x2 / x4 | 0.861 / 0.826 / 0.833 | 3.99 / 3.33 / 2.72 | 1.66 / 1.54 / 1.56 | 0.92 / 0.90 / 0.97 | worse |
  | lqx x1 / x2 / x4 | 0.774 / 0.785 / 0.764 | 3.24 / 3.42 / 2.79 | 1.47 / 1.49 / 1.34 | 0.87 / 0.87 / 0.89 | = infer |

  Nothing is significant overall (lqx4 − infer −0.029 ± 0.066). On 1-2 tile
  plays lqx4 − infer = −0.30 ± 0.32 per turn (n=260): suggestive, matching the
  bits result. The pure rule loses on 3+ tiles, as expected.
- Follow-up **dl-lq12** running on deb128 since 01:02: 4,000 positions with the
  opponent playing 1-2 tiles only (new `-maxplayed` flag), scales 1,2,4,8
  (~3 h). Decides whether a hybrid (lq/lqx on 1-2 tiles, ours else) is worth a
  match.
- deb192 baseline infer-v-sim-2ply-10k: 3,373 pairs, 50.12% ± 0.72.
- **Leave-quality bot option** (c0311252, env `MACONDO_LQ_TABLE/SCALE/MAXPLAYED`,
  off by default): reweights the complete posterior the sim samples from
  after placements of ≤ 2 tiles. Built as **f4e4300b + this patch only**
  (`bin/shell-lq` on deb128), so the only difference from the baseline binary
  (410b0a42 = f4e4300b) is the rule. Smoke-tested on deb128 (applies 4·beta,
  e.g. 0.097 for 2 tiles/38 pts, 0.206 for 2 tiles/24 pts).
- **lq4-2ply** armed on deb128 (queue4.sh, after dl-lq12, ~04:15): 5,000 pairs,
  scale 4, same seed (20261020) and setup as the deb192 baseline, so game
  seeds pair with its first 5,000 pairs. ~25 h. If dl-lq12 favours another
  scale, swap before it starts.
- deb192: queue9 armed, filler `infer-v-sim-2ply-ext` (same baseline, seed
  20261022) runs after the 10k baseline so the machine never idles.

### Oct 22 09:00: dl-lq12 result, lq4-2ply early

**dl-lq12** (4,000 positions, opponent placed 1-2 tiles): mean decision regret,
and the paired difference from our read (negative = better):

| read | all (4,000) | 1 tile (708) | 2 tiles (3,292) |
|---|---|---|---|
| prior | 1.952 (+0.275 ± 0.057) | 3.749 | 1.566 |
| infer (ours) | 1.677 | 3.060 | 1.379 |
| lq x2 / x4 / x8 | -0.03 / -0.08 / -0.05 (n.s.) | -0.38 / -0.57 / -0.55 | +0.05 / +0.02 / +0.06 |
| **lqx x1** | -0.144 ± 0.045 | -0.51 ± 0.19 | -0.064 ± 0.034 |
| **lqx x2** | **-0.192 ± 0.059** | -0.75 ± 0.26 | **-0.072 ± 0.043** |
| **lqx x4** | **-0.180 ± 0.070** | **-0.80 ± 0.31** | -0.048 ± 0.052 |
| lqx x8 | -0.101 ± 0.083 | -0.66 ± 0.35 | +0.02 |

- **The combined read (ours x the rule) is significantly better on 1-2 tile
  plays**: it adds about 70% to what our read gains over the prior there
  (0.275). Most of it is on 1-tile plays (regret 3.06 -> 2.27). The pure rule
  alone does not beat ours. x2 and x4 are level; lq4-2ply (x4) keeps running.
- **lq4-2ply early** (1,027 pairs, same seeds as the baseline): 49.05% vs the
  baseline's 50.15% on the same pairs, difference -1.10 ± 1.14 (n.s.). Too
  early; ~25 h total.
- Baseline infer-v-sim-2ply-10k: 6,516 pairs, 50.71% ± 0.52.

### Oct 22 20:00: baseline final, lq4 at 3,449 pairs

- **infer-v-sim-2ply-10k FINAL** (deb192, 10,000 pairs, build f4e4300b,
  engine defaults, budget 200): **50.66% ± 0.43**. The 2-ply inference
  baseline; oracle ceiling 53.51 (2-ply) / 53.73 (5-ply).
- Filler infer-v-sim-2ply-ext (seed 20261022) started 17:58 back-to-back.
- Rule-off check: `shell-lq` with the rule off reproduced all 32 baseline games
  on the first 16 seeds exactly, so the build is clean. With the rule on, 24 of
  those 32 games are unchanged.
- **lq4-2ply** at 3,449 pairs: 49.38%; baseline on the same seeds 50.14;
  **difference -0.76 ± 0.64**. The read gain in dl-lq12 does not show up as
  wins; if anything the rule hurts.

### Oct 23 01:35: two leave-quality follow-ups queued

lq4 at 4,616 pairs: -0.73 ± 0.54 vs the baseline on the same seeds (626 pairs
changed). Two single-cause follow-ups, both seed 20261020 (paired with the
10k baseline), `shell-lq` eba0a241 (f4e4300b + c0311252 patch):

- **lq4max1-2ply** (deb192, started 01:33, ~13 h): x4 on 1-tile plays only.
  dl-lq12's gain was on 1-tile plays (-0.80), while 2-tile plays (gain -0.05)
  are 83% of the rule's firings: tests whether the 2-tile reweighting costs.
  The filler infer-v-sim-2ply-ext was stopped at 2,784 pairs (50.57% ± 0.81)
  after a 2-minute overlap.
- **lq1-2ply** (deb128, queue5 after lq4, ~25 h): x1 on 1-2 tile plays; tests
  whether x4 is too sharp in games.

### Oct 23 10:20: lq4 final, lq4max1 and lq1 midway

Differences from the 10k baseline on the same seeds:

| run | pairs | win % | baseline, same seeds | difference | pairs changed |
|---|---|---|---|---|---|
| **lq4-2ply FINAL** (x4, 1-2 tiles) | 5,000 | 50.09 | 50.76 | **-0.66 ± 0.53** | 691 |
| lq4max1-2ply (x4, 1 tile) | 3,378 | 49.81 | 50.15 | -0.34 ± 0.31 | 117 |
| lq1-2ply (x1, 1-2 tiles) | 1,571 | 50.13 | 50.48 | -0.35 ± 0.59 | 92 |

- Every version is below the baseline, including the 1-tile-only one, where
  dl-lq12 found the biggest read gain (-0.80 regret). The agreement-regret
  metric does not predict wins here. Same pattern as fixed tau 0.05 (dl
  +0.22/turn, cap05 match null).
- deb192: queue11 filler infer-v-sim-2ply-ext2 (seed 20261023) armed after
  lq4max1 (~15:00).

### 2-ply inference base rate (summary, Oct 23)

Inference (engine defaults at f4e4300b, budget 200) vs plain sim, both 2-ply,
paired, on deb192:

| run | seed | pairs | win % |
|---|---|---|---|
| infer-v-sim-2ply-10k | 20261020 | 10,000 | 50.66 ± 0.43 |
| infer-v-sim-2ply-ext (filler, stopped) | 20261022 | 2,840 | 50.53 ± 0.81 |
| **pooled** | | **12,840** | **50.63 ± 0.38** |

The earlier `infer-v-sim-2ply` (Oct 17, 4,863 pairs, 50.55 ± 0.62) was an
older build and is not pooled. Matches on seed 20261020 are compared with the
10k run on the same seeds.

### Oct 23 12:00: divergence judge (divjudge) on lq4 vs the baseline

`divjudge` (5114b7d5): of the 10,000 seatings, 6,841 are identical in both
runs and 3,159 diverge, every one first after an opponent 1-2 tile play (719
one-tile, 2,440 two-tile), always at the inferring bot's move. Each first
divergence was judged from the same position, 5,000 iterations at 5 plies (all
unseen tiles at the end), once with the opponent's true leave known and once
without. Differences are lq4's move minus the baseline's:

| group | n | judge, leave known | judge, leave unknown | score | tiles | static equity |
|---|---|---|---|---|---|---|
| **all** | 3,159 | **-0.47 ± 0.18** | -1.42 ± 0.10 | -4.0 | -0.49 | -3.4 |
| bag 61-100 | 976 | -0.62 ± 0.16 | -1.22 | -4.4 | -0.47 | -3.7 |
| bag 41-60 | 606 | -0.66 ± 0.25 | -1.25 | -5.0 | -0.62 | -3.5 |
| bag 21-40 | 637 | -0.43 ± 0.27 | -1.22 | -2.5 | -0.49 | -2.7 |
| bag 8-20 | 500 | +0.26 ± 0.46 | -1.35 | -3.9 | -0.41 | -3.1 |
| bag 1-7 | 440 | -0.74 ± 1.03 | -2.47 | -3.9 | -0.49 | -3.8 |

(win % points per divergence)

- **The rule makes the bot play smaller, more defensive moves**: 4 points and
  half a tile less per changed move, 3.4 points of static equity given up.
- **Even a judge that knows the opponent's real leave says those moves are
  worse** (-0.47 per divergence), most clearly early in the game. Only bag
  8-20 is level.
- The judge agrees with the games: where it prefers lq4's move (701), lq4's
  game result is +3.4 ± 3.9; where it prefers the baseline's (1,007), -4.8 ±
  3.4. So the judge is a fair referee, and dl-lq12's agreement regret was not.
- Reading: x4 piles the read onto the very strongest leaves. The opponent did
  keep good tiles, but the sim then sees bingo threats everywhere and gives up
  score to block. The agreement metric (best move for each sampled leave, one
  at a time) does not see that cost, because the match sim averages over the
  read in one search.

## Oct 23 22:30: lq4max1 final, lq1 midway, deb192 crash

- **lq4max1** (×4, 1-tile plays only), 4,999 of 5,000 pairs: 50.47% ± 0.61
  vs plain sim; vs the baseline on the same pairs **−0.28 ± 0.26** (180 pairs
  changed). deb192 crashed while the last game was finishing, so one pair is missing.
- **lq1** (×1, 1–2 tiles, deb128), 3,731 of 5,000 pairs: 50.07% ± 0.71;
  vs the baseline **−0.19 ± 0.40** (236 changed).
- So far no version of the rule beats the baseline; the gentler it is, the
  smaller the loss. This matches the divergence judge (the rule over-blocks).
- **deb192 crash, ~14:27 Oct 23**: at the end of lq4max1. Over the last
  20 minutes, log output fell from ~3,000 to ~40 lines per minute as the last
  games finished. The machine died before the queued rule-only match
  (lqpure2) started. The previous crash (Oct 21, 15:48) came under steady
  full load, with temperatures normal in the owner's power log (not running
  since). Mitigation: queue13 starts the next run once fewer than 400 games
  remain, so load never tapers. lqpure2 restarted 22:23 Oct 23, then
  fillers ext2–ext4.

## Oct 23 23:00: screening judge (fast variant tests without matches)

`cmd/screendecide` rebuilds positions from the baseline's per-turn log, with
each game's own seed, and asks a bot what it would play. Seeded sims make
this exact: the baseline config reproduces the logged move (23 of 23 and
6 of 6 in smoke tests). So a variant (set by environment) differs only where
it really changes its mind, with no sim-noise flips. `divjudge -mode screen`
then judges each changed move against the control's, with a 5-ply,
5,000-iteration sim that knows the opponent's true leave (and one that
doesn't). Cost: about an hour of deb192 per variant, against a day for a
5,000-pair match, which mostly measures the unchanged games.

New rule settings: `MACONDO_LQ_MINPLAYED`, `MACONDO_LQ_MAXBAG`,
`MACONDO_LQ_PURE_SIGNED`. The fitted β is about 0 after 5-tile plays and
negative (−0.02 to −0.04) after 6-tile plays: the one kept tile tends to be
worse than random.

**Round 1 (deb192, from 22:42 Oct 23; lqpure2 soaks idle cores at nice 19):**
- inference bot after 1–2 tile plays (3,000 positions): rule ×4 (calibration;
  the match lost 0.66), ×1, ×0.5, ×0.25; late-bag variants come from the
  same data by filtering on the bag;
- inference bot after 6-tile plays: rule ×1, ×2 (negative β, so weak leaves
  are favoured);
- plain bot after 1–6 tile plays (4,000): rule-only ×2 (the lqpure2 config),
  ×1, and signed ×1 up to 6 tiles, each against the plain bot as control.

Early calibration check: ×4 changes 25% of decisions after 1–2 tile plays.

**deb128, after lq1 (~04:00 Oct 24):** divjudge on lq1; then the headroom
map. The oracle and plain bots re-decide 4,000 logged inference positions,
judged against the inference moves and against each other. This shows where
the ~3 points between inference (50.6) and the oracle (53.7) are won.

## Oct 24 21:00: screening round 1, headroom map, lq1 and lqpure2 final

Endgame positions (empty bag) are excluded throughout. Verdict = judge win %
of the variant's move minus the control's, per changed move. Judge: 5-ply,
5,000 iterations, opponent's true leave known; "unknown" uses random racks.

**Matches (final):**
- lq1 (×1, 1–2 tiles) vs the baseline: **−0.04 ± 0.35** (5,000 pairs,
  327 changed); 50.71% ± 0.62 vs plain.
- lqpure2 (rule-only plain bot, ×2, ≤4 tiles) vs plain: **49.98% ± 0.42**
  (10,000 pairs). Inference gets 50.66 on the same seeds; the rule alone does
  nothing.
- divjudge lq1: 1,514 divergences, **+0.39 ± 0.27** (bag 8–20 +1.11 ± 0.65,
  bag 21+ +0.18 ± 0.17).

**Round 1, inference bot after 1–2 tile plays (2,497 midgame positions):**

| rule | changed | known | unknown |
|---|---|---|---|
| ×4 | 675 | −0.05 ± 0.42 | −1.54 ± 0.23 |
| ×1 | 282 | **+1.01 ± 0.59** | −0.58 ± 0.27 |
| ×0.5 | 174 | +0.90 ± 0.75 | −0.56 ± 0.30 |
| ×0.25 | 110 | +0.55 ± 0.81 | −0.34 ± 0.31 |

By bag, ×4: about −0.65 per change at bag 21+, +1.92 ± 1.30 at bag 8–20.
×1: +0.1 to +0.3 at bag 21+, +3.11 ± 1.96 at 8–20, +1.69 ± 2.33 at 1–7.
This agrees with divjudge, whose first divergences come mostly early.
The rule hurts early and helps late.

The match can't see this: ×1 changes about one game in seven, so +0.4 per
divergence is about +0.06 per game.

After 6-tile plays (signed β): 22 and 32 changes out of 2,540, −0.80 and
−0.68 (dropped). Rule-only, plain bot vs plain control: ×2 +0.23 ± 0.63,
×1 +0.58 ± 0.79, signed ≤6 +0.39 ± 0.67 (not significant).

**Headroom map (deb128, 3,799 midgame positions of the inference bot):**
- The oracle changes 794 moves (21%), **+1.58 ± 0.26** per change, +0.33 per
  position.
- Plain vs inference: 257 changes, plain worse by 0.62 ± 0.40, i.e. inference
  gains +0.042 per position. That is ~11% of the oracle's +0.38 per position
  over plain.
- **Where the oracle's gain is**, by bag: 1–7: +1.31 per position (43% of the
  gap); 8–20: +0.32 (15%); 21+: ~+0.19 (42%). By the opponent's last play:
  1 tile +1.42 per position, 2 tiles +0.84, 3 tiles +0.44 (together 74%);
  4–6 tiles 22%; bingo 0; exchange/pass 2%.

**Round 2 (late game, started 20:50 Oct 24):** deb192: the rule at bag ≤20
after 1–4 tile plays, ×1/×2/×4 up to 2 tiles and ×1/×2 up to 4 tiles, on fresh
positions (×1 up to 2 tiles replicates round 1). deb128: tau 0.03 and 0.1
(default 0.05), and inference budget 1,000 and 2,000 (default 200), at bag
≤20 after any play.

## Screening round 2: late game (bag ≤ 20)

Fresh positions; judge as in round 1 (win % per changed move, true leave known).

| variant | positions | changed | known | unknown |
|---|---|---|---|---|
| rule ×1, ≤2 tiles (replication) | 2,476 | 91 | +1.11 ± 1.72 | −0.60 ± 0.65 |
| rule ×2, ≤2 tiles | 2,476 | 137 | +0.04 ± 1.46 | −1.39 ± 0.61 |
| rule ×4, ≤2 tiles | 2,476 | 211 | −0.02 ± 0.97 | −1.34 ± 0.51 |
| rule ×1, ≤4 tiles | 2,476 | 174 | +1.03 ± 1.03 | −0.32 ± 0.41 |
| rule ×2, ≤4 tiles | 2,476 | 284 | +0.21 ± 0.89 | −0.99 ± 0.38 |
| tau 0.03 (default 0.05) | 2,392 | 252 | +0.51 ± 1.13 | −2.04 ± 0.56 |
| tau 0.1 | 2,392 | 181 | +1.23 ± 1.02 | −1.52 ± 0.50 |
| budget 1,000 (default 200) | 2,392 | 28 | +0.66 ± 1.33 | |
| budget 2,000 | 2,392 | 29 | +1.06 ± 1.44 | |

- Late, ×1 beats ×2 and ×4 again.
- A bigger budget barely changes a move late (about 1%).
- Late verdicts are noisy: the spread of per-change differences is 7–9 win
  points, against ~0.7 of judge noise. So more positions help, not more judge
  iterations.
- Both tau directions came out positive. That raises a question: does any
  perturbation of the 2-ply bot look good to the 5-ply judge? Round 3 adds a
  placebo (`screendecide -salt`: a salted seed changes moves only through sim
  and inference noise) and reruns tau 0.1, rule ×1 ≤4 tiles, and both
  combined, on 12,000 shared late positions.

## Screening round 3: placebo and the late candidates

12,000 sampled late positions (bag ≤ 20, any opponent play); 2,336 had an
empty bag and were dropped, leaving 9,664 shared positions. Early placebo:
3,000 round 1 positions (bag > 20, opponent played 1–2 tiles). Win % with the
true leave known; "per position" counts unchanged positions as 0.

| variant | changed | per change | per position | unknown leave |
|---|---|---|---|---|
| placebo (salted seed), late | 324 | −0.11 ± 0.40 | −0.004 ± 0.013 | −0.01 ± 0.22 |
| placebo, early | 302 | +0.05 ± 0.33 | +0.005 ± 0.033 | −0.15 ± 0.14 |
| tau 0.1 | 786 | +1.16 ± 0.49 | +0.094 ± 0.040 | −1.54 ± 0.28 |
| rule ×1, ≤4 tiles | 470 | +1.01 ± 0.56 | +0.049 ± 0.027 | −0.75 ± 0.26 |
| tau 0.1 + rule ×1 ≤4 | 951 | +1.05 ± 0.44 | +0.103 ± 0.044 | −1.64 ± 0.25 |

By bag (per position):

| variant | bag 1–7 | bag 8–20 |
|---|---|---|
| placebo | +0.003 ± 0.017 | −0.010 ± 0.021 |
| tau 0.1 | +0.155 ± 0.077 | +0.037 ± 0.028 |
| rule ×1, ≤4 tiles | +0.057 ± 0.046 | +0.042 ± 0.031 |
| both | +0.144 ± 0.081 | +0.064 ± 0.036 |

- The placebo is flat. Noise-only move changes judge at zero, so the judge has
  no built-in preference for departures from the baseline's move, and the
  positive verdicts are real.
- Late, tau 0.1 is the strongest single change, mostly at bag 1–7, where the
  headroom map put 43% of the oracle gap. Per position it recovers about a
  tenth of that bucket's gap (+0.155 of +1.31).
- Both effects are real, but they don't add up: the combination is about tau
  0.1 alone at bag 1–7, with the rule adding a little at bag 8–20.
- The unknown-leave judge disagrees in sign, as expected: these moves use the
  read of the opponent's leave, which a judge without it can't credit.
- Next: find the late tau optimum (0.15, 0.2), and test tau 0.1 early against
  a second early placebo.
