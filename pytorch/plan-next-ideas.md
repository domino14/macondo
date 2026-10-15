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
Built 10/14: trainer `--rank-loss topk|pair` (`--rank-k`), groups
~/data/simdistill/groups-top16.bin (1.26M, top 16 each). Ready to run after
the GPU swap: `queue-simft3.sh` (pair) then `queue-simft4.sh` (topk, k 4),
each 10k steps from streamopen + 100k pairs vs HastyBot.
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
   Result 10/11: 57.65% ± 0.18, identical to streamopen. No gain; items 2
   and 3 move behind batch 3 (same kind of signal).
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
   Done 10/11: no new code needed. autoplay FAST_ML_BOT vs FAST_ML_BOT
   with the open sets' opening flags; the opening sampler serves both seats
   whatever the bot, FAST_ML_BOT honours the quick endgame, and the turn
   log has the same columns as open.txt (the producer reads it as is).
   Smoke test: 2,000 games in 222 s (12 threads, 3070 Ti), ~32k games an
   hour, ~780k a day: 3M games ~4 days, not 10. `pytorch/gen-nvn.sh`
   generates 12 chunks of 250k (~8 h each) into ~/data/nvn/nvn-NNN.txt.gz,
   skips finished chunks, stops between chunks on ~/data/nvn/STOP (for the
   GPU swap). Started 10/11 15:01 with streamopen.
2. Generate ~3M games (~4 days of GPU on the 3070 Ti). Meanwhile, on the CPU, 3M fresh
   Hasty games as the control.
3. Fine-tune the same starting net on each set, same steps and settings,
   mixing in old data so self-play does not narrow the net. Match both.
4. If net-vs-net wins: iterate (new net plays the next generation).

## Small things to test after the GPU swap

Swap done 10/15 00:45 (5070 Ti, PCIe x16): all 20 TensorRT engines rebuilt
(18 s each; old ones in ~/data/trt-rebuild/old-3070ti), streamopen v1 now
exports 4 outputs (value/spread identical), container recreated. Net-vs-net
autoplay 68k games/h at 12 threads (old card 32.7k); 24 threads no faster.
Serving benchmark 10/15 (bench-serving.sh, 3,000 games each, games/h):
baseline 68.4k (12 thr) / 67.1k (24); 2 instances 87.8k / 87.8k (+28%);
dynamic batching (200 us queue, batch <= 512) 55.4k / 49.3k / 47.6k (12/24/48
thr, slower: the queue delay adds latency to a latency-bound loop); both
58.7k / 73.0k (24/48). Two instances are now the default (copy_model.sh,
streamopen's config): same engine, identical outputs, more requests in flight.

- fp32 value tail: the TensorRT engines are fp16, and value = P(win) -
  P(loss) with both near 0.5 is computed in fp16, so differences under
  ~0.001 are rounded away (an opening's 8F/8G/8H JEU: 0.0176 each on the
  GPU, 0.0211/0.0196/0.0186 in fp32). Keep the wdl softmax and subtraction
  in fp32, rebuild, match against the fp16 engine. Pilot (558 positions):
  fp16 vs fp32 pick the same move 530 times, sim agreement 68.8 vs 67.9.

## Not repeating

- Opening scheme (open vs nwl23): no difference (57.65 vs 57.55).
- Top 100 candidates instead of 50: no gain.
- Energy (Ising) in candidate choice or sim leaves: null.
