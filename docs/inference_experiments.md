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
6. **tau 0.05 replication** (above), when out of ideas.
