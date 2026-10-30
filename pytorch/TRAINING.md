# Training a value net: the shipped recipe, for any lexicon

How `macondo-tf` and `macondo-tf-selfplay` were made, step by step, so they
can be rebuilt or trained for another lexicon (CSW24, ...). The commands
assume this repository at `~/code/macondo`, logs in `~/data`, and the local
Triton container from `triton-container.sh` (gRPC on `localhost:8101`).
Times are for one desktop (i5-12600K, RTX 5070 Ti); an RTX 3070 Ti trains
about half as fast.

## What matters in the recipe

From the experiments (`RUNS.md`; 100k game pairs vs HastyBot each, 95%
interval about ±0.18):

| Change | Effect |
|---|---|
| Transformer instead of the CNN | none by itself (52.33% vs 52.5%) |
| Auxiliary heads with gradient-balanced weights | +0.86 |
| True game result as the label, one position per game, WDL cross-entropy | same strength as the old labels, much cleaner; needs ~20M+ positions (5.4M: 50.98%) |
| **Four per-square spatial heads** (training only) | **+4.0** (53.13% → 57.17%), the big one |
| **Streamed training**: 6 passes, a fresh random turn per game each pass, 101k steps | **+0.55** over the cached run (57.00% → 57.55%, 4 SE) |
| Sampled openings instead of temperature-bot games | +0.10 ± 0.13, i.e. none measurable |

About the streamed-training gain: two things changed together. The cached
run trained on one position per game (36.4M), three times over, for 53k
steps; the streamed run drew a new turn from every game on each of 6
passes (about 218M mostly distinct positions from the same 54M games) for
101k steps. The run that would separate "more distinct positions" from
"twice the steps" (101k steps on the old cache) was never done. Doubling
the *games* alone earlier gained nothing (27M → 54M: 53.10% → 53.13%).

The decided-game ranking rule in the bot (`ai/bot/mlrank.go`) changes the
win rate by nothing and the average spread by +7.3 per game; every match
since 9/28 has it on. Sampled openings are kept because no label then
depends on a random move. The ideas that did not help, and how to get their code back, are in
`SHELVED.md`.

## Before you start

- **Lexicon.** Any lexicon with the English letter distribution: the
  features have one plane per letter A-Z (`game/mlhelper.go` refuses other
  distributions). The lexicon needs its word graph and leave values in
  `data/lexica/gaddag/` (`<LEX>.kwg`, `<LEX>.klv2`), which NWL23 and
  CSW24 have. Another alphabet means new feature planes and a new model
  shape.
- **Machine.** Linux, an NVIDIA GPU with 8 GB or more, Docker with the
  NVIDIA Container Toolkit, Go, and the Python venv from `README.md`.
- **Disk.** About 15 GB per 27M-game gzipped turn log (54M games: ~31 GB),
  plus a few GB of checkpoints and match logs.
- **Build.** `make` (in the repo root) and
  `go build -o bin/mlproducer ./cmd/mlproducer`.
- **GPU rule.** Training never shares the GPU with a match or served
  models: `run-stream.sh` waits for ML matches to end and unloads Triton
  before it trains.

Set the lexicon once for the commands below:

    export LEXICON=CSW24 L=csw24     # L: a lower-case tag for file names

## 1. Generate 54M games (~10 h, CPU only)

HastyBot vs HastyBot. Each game's first K plies, K ~ round(Exp(2)), are
sampled from a softmax over static equity (temperature 3 points, top 50
plays); after that both play their best move. Close endgames are played
with a quick 2-ply search. Two batches of 27M (the shipped model used
`open` and `open2`):

    cd ~/code/macondo/pytorch
    TAG=${L}open  GAMES=27000000 THREADS=16 ./gen-openings.sh
    TAG=${L}open2 GAMES=27000000 THREADS=16 ./gen-openings.sh

Output: `~/data/${L}open.txt.gz`, `~/data/${L}open2.txt.gz` (the per-turn
logs the producer reads) and `games-*.txt` (one line per game). About
120 games/s per thread: `open` took 5.2 h on 12 threads (~1,450 games/s),
so ~5 h per batch. The logs carry an
`openingplies` column; the producer only draws training positions after
the last sampled ply.

## 2. Train (~13 h on a 5070 Ti)

    TAG=${L}stream \
    LOGS="$HOME/data/${L}open.txt.gz $HOME/data/${L}open2.txt.gz" \
    VAL_LOG=$HOME/data/${L}open2.txt.gz \
      ./run-stream.sh

What it does (the defaults are the shipped recipe):

1. **Validation set.** Games whose ID hashes to 0 mod 20 (5%) are held out
   of training entirely; up to 150,000 positions from them are scanned
   once into `val-${L}stream.bin`.
2. **Speed probe.** Picks the fastest of batch 128×16, 256×8 and 256×8 with
   `torch.compile` (always an effective batch of 2,048). Set
   `BATCH=128 ACCUM=16` to skip it.
3. **Training.** `PASSES=6` scans of the logs piped through
   `bin/mlproducer -lexicon $LEXICON -labeler result -per-game`
   into `training.py`, which trains:
   - the transformer (8 layers, width 192, 6 attention heads, 3.7M
     parameters; the shipped `macondo-tf` and `macondo-tf-csw` are 12
     layers, width 256, 8 heads, 9.6M parameters, trained on 81M games with
     `TRAIN_EXTRA="--d-model 256 --layers 12 --heads 8" BATCH=128 ACCUM=16`);
   - WDL cross-entropy as the main target (the exported `value` is
     P(win) − P(loss));
   - auxiliary heads (spread, opponent bingo, opponent score) balanced to
     15% of the main head's gradient (`--aux-share 0.15`);
   - the four spatial heads at 10% (`--spatial-share 0.1`);
   - AdamW 3e-4, 2,000 warmup steps, cosine to zero, bf16.
4. **Schedule length.** `STEPS` comes from `ROWS_PER_PASS × PASSES / 2048`.
   The default `ROWS_PER_PASS` (34.58M) is right for 54M games (a pass
   yields about 0.64 positions per game: the 95% not held out, minus games
   whose drawn turn has an empty bag). For a different number of games,
   set `ROWS_PER_PASS` to about 0.64 × games; the first pass's count is
   the `vectorsEmitted` line at the end of `producer-${L}stream.log`.
5. **Deploy and match.** The best-validation checkpoint
   (`best-tf-${L}stream.pt`) is exported to ONNX and TensorRT
   (`copy_model.sh`, with a parity check) as Triton model
   `macondo-nn-tf-${L}stream`, then plays 100,000 game pairs against
   HastyBot in `$LEXICON` (`watch-tf-heads.sh`, about 3.5 h). Statistics:
   `python pairs-stats.py ../games-tf-${L}stream-v-hasty-pairs.txt`.

For NWL23 this gave 57.65% ± 0.18. HastyBot is relatively stronger in CSW,
so expect a smaller margin there.

## 3. Optional: the self-play fine-tune (~1.5 days)

`macondo-tf-selfplay` is the step-2 model fine-tuned on its own games
(details and results: `results/selfplay/REPRO.md`):

    MODEL=macondo-nn-tf-${L}stream TAG=${L}nvn ./gen-nvn.sh   # 12 chunks of 250k games, ~1 day
    R=$(zcat ~/data/nvn/${L}nvn-*.txt.gz | grep -vc '^playerID,')
    python3 mix-logs.py ~/data/mix-${L}nvn.txt.gz "$HOME/data/nvn/${L}nvn-*.txt.gz" ~/data/${L}open2.txt.gz:$R
    TAG=${L}nvnft LOGS=~/data/mix-${L}nvn.txt.gz VAL=val-${L}stream.bin PASSES=7 ROWS_PER_PASS=4000000 \
    STEPS=11000 BATCH=128 ACCUM=16 GPU_MEM=5120 LR=1e-4 WARMUP=500 DEPLOY_FINAL=1 \
    TRAIN_EXTRA="--init-ckpt best-tf-${L}stream.pt" \
      ./run-stream.sh

Judge it head to head against its parent, not only against HastyBot: for
NWL23 it lost 0.5 points against HastyBot but beat its parent 50.37% ±
0.16. A head-to-head match, 100k pairs:

    M1=macondo-nn-tf-${L}nvnft M2=macondo-nn-tf-${L}stream ./match-ml-v-ml.sh

## 4. Ship it

Install the checkpoint under its permanent name and commit the ONNX file
and Triton config (engines are built per machine, never committed):

    ./copy_model.sh macondo-tf-${L} best-tf-${L}stream.pt
    git add -f ../data/strategy/default/models/macondo-tf-${L}/config.pbtxt \
               ../data/strategy/default/models/macondo-tf-${L}/1/model.onnx

Add it to the Models table in the top-level `README.md`, and archive the
checkpoint, loss CSV and match logs outside the repo (as in
`results/streamed/REPRO.md`).

**Choosing the model at play time.** Macondo does not pick a model by
lexicon: `triton-model-name` (or `MACONDO_TRITON_MODEL_NAME`) is global and
defaults to the NWL23 model, which would happily, and wrongly, score CSW
positions. Set it for a CSW session, or per player in autoplay with
`-tritonmodel1/-tritonmodel2`. A lexicon-to-model default belongs in the
config once a second lexicon ships.
