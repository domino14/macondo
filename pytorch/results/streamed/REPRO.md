# stream / streamopen: streamed training, and the openings scheme at full scale

Results (2026-09-30 and 2026-10-01), 100,000 game pairs vs HastyBot, NWL23,
FastMlBot ranking HastyBot's top 50 plays, decided-game rule on, no solver:

| model | games | paired win rate | swept / lost | spread |
|---|---|---|---|---|
| `stream` | 54M temperature games (nwl23 a+b+c) | 57.55% ± 0.18 | 24.8% / 9.9% | +9.3 |
| `streamopen` | 54M sampled-openings games (open + open2) | 57.65% ± 0.18 | 25.0% / 9.8% | +9.5 |

Baselines under the same bot: `nwl23s` (cached, 53k steps) 57.00% ± 0.18,
`nwl23st` 57.30% ± 0.18. So streaming six fresh-turn passes (twice the
steps) is worth about +0.4 to +0.5, and the two ways of generating games
are indistinguishable (+0.10, standard error 0.13). The "±" in the table
is the 95% interval half-width (1.96 standard errors).

The network, targets and loss are those of `nwl23s`
(`../nwl23s-spatial-heads/REPRO.md`): transformer d=192 x 8 layers, WDL
primary, scalar aux heads at gradient share 0.15, four per-square placement
heads at share 0.1, transpose augmentation off.

## What changed: no frame cache

`pytorch/run-stream.sh`. The producer rescans the gzipped turn logs once
per pass and draws a fresh turn for every game each time (uniform over
turns 1..30, or from the last sampled opening ply; empty-bag draws
skipped), piping frames through a FIFO into one trainer process:

    for pass in 1..6:
      zcat <logs> | bin/mlproducer -labeler result -per-game -endgame-plies 2 \
                      -holdout-mod 20 -split train
    | python training.py --arch transformer --primary wdl --w-wdl 1 --w-value 0 \
        --aux-share 0.15 --spatial-share 0.1 --transpose-prob 0 \
        --batch-size B --accum A --epochs 1 --val-cache val-<tag>.bin --val-size 150000 \
        --shuffle-buffer 1024 --total-steps 101000 --snapshot-every 10000

Games whose ID hashes to 0 mod 20 are held out entirely (`-split val`,
scanned once into `val-<tag>.bin`), so validation never sees a training
game at another turn. AdamW 3e-4, 2,000 warmup steps, cosine to zero at
101,000 steps, effective batch 2,048, bf16.

| | stream | streamopen |
|---|---|---|
| started | 2026-09-28 23:44 | 2026-09-30 07:15 |
| batch x accum | 128 x 16 | 256 x 8, `--compile` (picked by the speed probe) |
| rows per pass | ~34.6M | 34.0M |
| steps taken | 101,000 | ~99,600 (the six passes ran out first; LR was ~0) |
| speed, wall time | 2,054 pos/s, 28.0 h | 2,567 pos/s, 22.1 h |
| best held-out WDL loss | 0.50033 (step 98,500) | 0.48736 (step 97,500) |

The two validation losses are on different game distributions and are not
comparable; the matches are.

## Code

Branch `claude/transformer-valuenet` (draft PR #535).

- `9bfd80f3` autoplay sampled openings (`-openingplies -openingtemp
  -openingtopn -openinguniform`, `openingplies` log column).
- `f093d47d`, `5cbd103b` streamed training (producer split flags, FIFO
  feeder, header-row handling). `stream` ran this code.
- `d1784d92` decided-game rule in the bot (every match here).
- `f9a02cf4` trainer `--compile` and the batch/compile probe. `streamopen`
  ran this code.

## Data

- `stream`: the 54M games of `../nwl23s-spatial-heads/DATA.md`.
- `streamopen`: `~/data/open.txt.gz` (2026-09-27) and `~/data/open2.txt.gz`
  (2026-09-29), 27M games each, md5 in `DATA-MD5SUMS`:

      bin/shell autoplay -botcode1 HASTY_BOT -botcode2 HASTY_BOT -lexicon NWL23 \
        -quickendgame 2 -quickendgamemargin 60 -quickendgamecap 250 \
        -openingplies 2 -openingtemp 3 -openingtopn 50 -openinguniform 0 \
        -numgames 27000000 -block true -experimentid open2

  Both sides are HastyBot; the first K ~ round(Exp(mean 2)) plies are
  sampled by softmax(equity / 3) over the top 50, then best play to the
  end. A training position is only drawn from ply K on. Unseeded, so the
  games are reproducible only from the kept logs.

## Matches

    MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
    MACONDO_TRITON_MODEL_NAME=macondo-nn-tf-<tag> MACONDO_TRITON_MODEL_VERSION=1 \
    bin/shell autoplay -botcode1 FAST_ML_BOT -botcode2 HASTY_BOT -lexicon NWL23 \
      -numgames 100000 -gamepairs true -threads 12 -block true \
      -experimentid tf-<tag>-v-hasty-pairs

Master seeds (replay with `-seed`): stream 3415107630100574503, streamopen
1159194858927648171. Statistics: `pytorch/pairs-stats.py`.

## Archive

`~/data/results/streamed-stream-streamopen/` on the training machine:
checkpoints and ONNX (`model/`, md5 in `MD5SUMS`), loss CSVs, trainer and
probe logs (`logs/`), per-game and per-turn match logs (`match/`).

## Reproducing

1. Check out `f9a02cf4` or later; `make`; `go build -o bin/mlproducer ./cmd/mlproducer`.
2. `cd pytorch && ./run-stream.sh` for `stream` (set `BATCH=128 ACCUM=16`
   for the exact microbatch), or
   `TAG=streamopen LOGS="open.txt.gz open2.txt.gz" VAL_LOG=open2.txt.gz ./run-stream.sh`.
   The script scans the held-out games, trains, exports, checks parity and
   plays the match.
