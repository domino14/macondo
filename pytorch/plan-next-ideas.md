# Plan: next ideas for the value net (10/9/26)

Baseline throughout: streamopen, 57.65% ± 0.18 vs HastyBot (100k pairs).
Every idea is judged the same way: 100k pairs vs HastyBot, and head to head
vs the current best (`-tritonmodel1/2`) when it looks like a gain.

The GPU does one thing at a time (training, matches and self-play
generation all need it; see GPU memory headroom). CPU-only work (Hasty game
generation, sim labelling) can run beside anything.

## Batch 1: sim distillation, top-k loss  (when deb192 finishes, ~10/13)

Batch 2 runs first, in the GPU time before the labels are done.

Tries 1 and 2 (listwise loss over all 50 candidates) lowered the ranking
loss without raising top-pick agreement. With all 1.32M labels: a loss on
the top of the list only (cross-entropy over the sim's top 3-5, or pairwise
best-vs-each weighted by gap / SE). Details in plan-sim-distill.md.
Cost: ~1 h to build groups, ~5 h fine-tune, ~3.5 h match.

## Batch 2: more spatial heads  (each ~1 day training + match)

Why the four Scribblez heads worked: one game result is one bit per
position; 900 per-square bits per position tell the trunk where the action
is. More heads help only if they carry new information (the spatial share
is fixed, so extra heads dilute the others). The targets come from the
game logs we already have (the streamed producer replays full games), so
no new games are needed: producer change, training run, match.

In order, one at a time:

1. Ownership: for every square empty now, who covers it by the end of the
   game (mover / opponent / nobody), 3-way per square. KataGo's ownership
   head is the analogue. Looks past the next move to the rest of the game.
   Built 10/9: producer `-ownership` (two planes, self_own / opp_own, after
   the four), trainer `MACONDO_OWNERSHIP=1` (checkpoint hparams record
   n_spatial, so export needs no setting), run-stream `PRODUCER_EXTRA`.
   Queued (`queue-own.sh`, TAG own): streamopen's recipe from scratch plus
   the two heads, after the simft2 match; ~22 h training, result ~10/11.
   Each spatial head gets its own share (0.1) of the primary's pull, so the
   two new heads add auxiliary pull rather than diluting the old four.
2. Big-score threats: squares covered by a move scoring 30+ or a bingo, per
   player, over the next move each.
3. Longer horizon: squares covered within the next 2-4 moves.

Skip deterministic targets (hooks, legal placements): computable from the
board, little to learn.

## Batch 3: net-vs-net games (AlphaZero-style)  (~2-3 weeks)

The net learns a position's value under the policy that played the games:
now HastyBot vs HastyBot (first ~2 plies softmax T=3, then greedy), while
the net drives a stronger bot. Games played by the net itself should give
values closer to good play. BestBot games cost ~45 CPU-min each (6,000 a
day on deb192), too slow; the net costs GPU time instead.

1. Generator: autoplay FAST_ML_BOT vs FAST_ML_BOT (best net so far,
   probably after batch 2), the same randomized opening plies as the open
   sets, turn logs in the producer's format. Measure throughput first;
   batching many games' evaluations together is the lever (engine 13k
   pos/s, ~2,500 evaluations per game: ~300-400k games/day if batched).
2. Generate ~3M games (~10 days of GPU). Meanwhile, on the CPU, 3M fresh
   Hasty games as the control.
3. Fine-tune the same starting net on each set, same steps and settings,
   mixing in old data so self-play does not narrow the net. Match both.
4. If net-vs-net wins: iterate (new net plays the next generation).

## Not repeating

- Opening scheme (open vs nwl23): no difference (57.65 vs 57.55).
- Top 100 candidates instead of 50: no gain.
- Energy (Ising) in candidate choice or sim leaves: null.
