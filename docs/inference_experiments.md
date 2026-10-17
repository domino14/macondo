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
