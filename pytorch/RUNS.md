# Run ledger

One line per model or experiment in the value-net program, newest last.
The narrative and the numbers' provenance are in `experiments.md`; this is
the map. Every match is 100,000 game pairs vs HastyBot, NWL23 unless noted,
FastMlBot ranking HastyBot's top 50 plays by the net; "± " is the paired
standard error. Spread is FastMlBot's average final spread per game.

## Data sets (turn logs in `~/data`, gzipped)

| name | games | how generated | used by |
|---|---|---|---|
| file 5 (`autoplay-softmax-v-hasty-5`) | ~? | June 2025 temperature bot vs HastyBot, NWL18 | heads2 (table labels) |
| fresh | 27M | temperature bot vs HastyBot, NWL18, greedy endgames (deleted 9/27) | fresh |
| nwl23 a+b+c | 54M | temperature bot (top 50, softmax T=1 while bag>60) vs HastyBot, NWL23, quick 2-ply endgames | nwl23, nwl23s, nwl23st, stream |
| open | 27M | HastyBot vs HastyBot, first K~Exp(2) plies softmax(T=3) over top 50, then greedy; quick endgames | open, streamopen |
| open2 | 27M | same as open, generated 9/28 | streamopen |

Labels in every run since `result`: the mover's true game result (WDL) and
spread to the end, one position per game drawn uniformly from turns 1..30
(from the last sampled ply for openings games), empty-bag draws skipped.
The producer replays half of all games transposed.

## Models

| tag | date | data | recipe change under test | steps | paired win rate | spread | notes |
|---|---|---|---|---|---|---|---|
| tf | 9/19 | file 5, table labels | transformer instead of CNN | 79.5k | 52.33 ± 0.22 | | = CNN plateau (52.5) |
| 1head | 9/20 | file 5 | value head only, 25k steps | 25k | 50.98 ± 0.18 | -9.3 | short-run control |
| heads | 9/20 | file 5 | + spread/wdl/opp heads, nominal weights | 25k | 51.76 ± 0.18 | -5.8 | heads help at equal steps |
| heads2 | 9/21 | file 5 | gradient-balanced aux heads (share 0.15) | 79.5k | 53.19 ± 0.18 (NWL18), 53.31 ± 0.18 (NWL23) | -2.7 | best before spatial heads |
| gen1 | 9/22 | file 5 | rollout labels (16x2-ply, net at leaf) | | 52.03 ± 0.18 | -4.6 | uniform opp racks: optimistic |
| result | 9/22 | 5.4M pos | true result, one pos/game | | 50.98 ± 0.18 | -10.7 | too little data |
| fresh | 9/24 | fresh 27M (20M pos) | true result, WDL primary | 49k | 53.10 ± 0.18 | -2.6 | |
| nwl23 | 9/26 | nwl23 54M (36.4M pos) | + quick endgames, NWL23, more games | 53k | 53.13 ± 0.18 | -1.4 | more games: no gain |
| **nwl23s** | 9/27 | nwl23 (36.4M pos) | **+ four per-square placement heads** (share 0.1) | 53k | **57.17 ± 0.18** (replication 57.13 ± 0.40) | +1.2 | archive `~/data/results/nwl23s-spatial-heads` |
| nwl23s + decided rule | 9/28 | same model | bot ranks decided games by expected final spread | | 57.00 ± 0.18 | +8.5 | rule is win-neutral, +7.3 spread; on for every match since |
| nwl23st | 9/28 | nwl23 (36.4M pos) | + per-batch transpose 0.5 | 53k | 57.30 ± 0.18 | +9.1 | n.s. vs decided-rule nwl23s; dropped |
| open | 9/28 | open 27M (17.9M pos) | sampled-openings games (clean labels), cache | 25k | 56.07 ± 0.18 | +7.2 | half the positions and steps of nwl23s; not a verdict on the scheme |
| **stream** | 9/29-30 | nwl23 54M, streamed | 6 passes, fresh turn per game per pass, no cache | 101k | **57.55 ± 0.18** | +9.3 | best so far; +0.55 ± 0.25 over nwl23s + decided rule (twice the steps, fresh positions) |
| streamopen | 9/30-10/1 | open+open2 54M, streamed | same as stream on openings games | 101k | (training since 9/30 07:15, batch 256x8 --compile) | | isolates the opening scheme; compare with stream 57.55 |

## Bot changes that affect matches

| date | change | effect |
|---|---|---|
| 9/24 | quick 2-ply endgame search in autoplay bots (margin 60) | game generation only |
| 9/28 | decided-game rule (`ai/bot/mlrank.go`): when the best value is beyond ±0.97, near-ties ranked by spread after the play + predicted spread change; Triton client requests the spread head | win rate unchanged, +7.3 spread/game; every match from nwl23st on |

## Tools

- `cmd/mlproducer`: turn log -> training frames (`-labeler result -per-game -endgame-plies 2`; `-picks K`; `-holdout-mod M -split train|val`).
- `pytorch/training.py`: trainer (cache mode and stream mode); `run-spatial.sh`, `run-openings.sh`, `run-stream.sh` drivers; `train-fresh.sh` + `watch-tf-heads.sh` deploy and match. `run-stream.sh` probes the fastest of batch 128x16 / 256x8 / 256x8 `--compile` (same effective batch 2048) before training; `probe-<tag>-*.log` keep the timings.
- `cmd/mlreads` + `pytorch/reads_gallery.py`: where the net departs from equity, adjudicated by the 5-ply sim; gallery https://claude.ai/artifact/1dy3cwMCdmcMwdYU2N5mSD.
- `pytorch/pairs-stats.py`: paired match statistics; `autoanalyze` in the shell prints the same.
