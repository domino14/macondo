# Plan: distill the 5-ply sim into the value net

Status: started 2026-10-05. Owner of each step in brackets.

## Idea

The net (streamopen) plays at 57.65% vs HastyBot, about the strength of a
2-ply sim; a fixed 5-ply sim plays at ~62.7%. The net's only job at play
time is to rank the top 50 plays of one position, but it learns that
ranking indirectly, from one win/loss per position. Give it the 5-ply sim's
verdict on all 50 candidates of a position and train it to order them the
same way, on top of what it already learns.

Design decisions (see the discussion of 10/5/26 in experiments.md):

- Positions come from games we already have (open + open2, 54M games, the
  data streamopen was trained on), one per game, not from new BestBot games.
  The labels are what matter, and each position keeps its true result.
- The labeller is a plain sim: fixed 5 plies, top 50 by static equity,
  Stop99, no inference (the net cannot see an inferred rack), no endgame.
  End of line scored with the win-percentage table, as BestBot does.
- The true game result stays the main training signal; the sim labels add
  a ranking loss. Fine-tune streamopen; do not train from scratch.

## Steps

### 1. Labelling tool  [Claude]

`cmd/simlabel`, two modes:

- `select`: read turn logs, pick one position per game (turns from the last
  sampled opening ply to 30, at least 9 unseen tiles), write
  `positions-*.jsonl.gz`: a key (log, game, half, turn), the CGP of the
  position with the mover's rack, the played move and the true result.
  Games whose ID hashes to 0 mod 20 go to a held-out file (as streaming).
- `sim`: read positions, sim each (one thread per position, many in
  parallel), write `labels-*.jsonl.gz`: the key and, per candidate, the move,
  sim win probability and its standard error, equity and its standard
  error, iterations, and whether the sim pruned it. Resumable: a restart
  skips keys already written. Shardable: `-shard i -shards n`.

Positions are self-contained (CGP), so the labelling machine needs no game
logs, only the repo and two small lexicon files.

### 2. Local pilot  [Claude]

First numbers (10/5, 24 positions, machine also running the 6-ply match):
~52 s wall and ~88 CPU-seconds per position (less on an idle machine),
~3,700 iterations; the stopping rule prunes ~90% of candidates after
128-256 iterations, so only ~4-5 per position are simmed to the end. In
decided games every candidate's win probability is ~1 and only the sim
equity separates them: the ranking target must fall back on equity there
(the same problem as the value head's saturation, see mlrank.go).

- Select 2,000 held-out positions; sim them here (~1 CPU-minute each: ~3 h
  on 12 threads; runs beside the sim benchmark).
- Measure: CPU per position, labels per day per core.
- Measure the room for improvement before any training: how often the
  net's top pick equals the sim's, and the rank correlation over the 50.
- Decide 5-ply vs 3-ply for the big run (3-ply is ~2.7x cheaper and plays
  ~2.6 points weaker than 5-ply).

### 3. The big run  [you, on the borrowed machine; Claude prepares]

What the machine needs: Linux or macOS, Go 1.26+, git, ~10 GB free disk,
many cores (no GPU). Nothing else; the positions file and two lexicon files
are copied over.

```
# once
git clone https://github.com/domino14/macondo && cd macondo
git checkout claude/transformer-valuenet
go build -o bin/simlabel ./cmd/simlabel
mkdir -p data/lexica/gaddag
# from the home machine:
#   scp data/lexica/gaddag/NWL23.kwg data/lexica/gaddag/NWL23.klv2 big:macondo/data/lexica/gaddag/
#   scp positions-train.jsonl.gz big:macondo/
# run (nohup so it survives logout; restartable)
nohup bin/simlabel sim -in positions-train.jsonl.gz -out labels.jsonl.gz -threads $(nproc) > simlabel.log 2>&1 &
tail -f simlabel.log        # progress: positions done, rate, ETA
# when done, copy labels.jsonl.gz back
```

Size: set by the pilot's rate. At ~1 CPU-minute per position a 128-core
machine labels ~180,000 positions a day; target 1M (about 6 days) or
whatever the machine allows. Several machines: give each a shard
(`-shard i -shards n`).

### 4. Training data  [Claude]

Converter (in `cmd/mlproducer`): replay each labelled game locally from the
logs to the position, build the net's input vector for every labelled
candidate (the same code the bot uses), and stream groups of up to 50
candidate frames with their sim labels into the trainer.

### 5. Fine-tune  [Claude]

From best-tf-streamopen.pt, learning rate ~1e-4 to 0 over 10-20k steps;
half of each batch ordinary streamed positions with all current losses
(true result, spread, auxiliary and placement heads), half sim groups with
a listwise ranking loss: cross-entropy between softmax(net value / t) and
softmax(sim win / t') over the group's candidates, each candidate weighted
by its iterations. Trainer changes: group batches, ranking loss, mixing.

### 6. Evaluate  [Claude]

- Held-out sim positions: top-1 agreement and rank correlation, before and
  after.
- 100k pairs vs HastyBot (baseline 57.65%) and 100k pairs head to head vs
  streamopen (`-tritonmodel1/2`).

## Risks

- Sim win% is noisy for candidates the sim prunes early; the iteration
  weights handle it, and win% is recorded with its standard error.
- The table's end-of-line estimate is miscalibrated in the tails; a ranking
  loss within a position mostly cancels it.
- Volume: one labelled position is closer to one rich example than to 50
  independent ones; this can refine streamopen, not replace its 54M games.
