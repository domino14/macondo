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

Pilot result (558 held-out positions, 531 contested): the net's pick is
the sim's in 70.8% of positions (static equity: 64.2%) and gives up 0.46
points of sim win% per move (static equity: 1.11). 54.5 CPU-seconds per
position beside the 6-ply match.

### 3. The big run  [you, on the borrowed machine; Claude prepares]

What the machine needs: Linux (or macOS), git, ~5 GB free disk, as many
cores as possible, no GPU. ~260 MB of memory for 6 simulations at once, so
memory is not a concern. Before cloning, the branch must be pushed
(`git push origin claude/transformer-valuenet` on the home machine).

From the home machine, copy over three files (~110 MB in all):

```
scp ~/data/simdistill/positions-train.jsonl.gz BIG:
scp data/lexica/gaddag/NWL23.kwg data/lexica/gaddag/NWL23.klv2 BIG:
```

On the big machine:

```
# Go 1.26 without root, if it is not installed
mkdir -p ~/sdk && curl -L https://go.dev/dl/go1.26.1.linux-amd64.tar.gz | tar -C ~/sdk -xz
export PATH=~/sdk/go/bin:$PATH

git clone https://github.com/domino14/macondo && cd macondo
git checkout claude/transformer-valuenet
go build -o bin/simlabel ./cmd/simlabel
mkdir -p data/lexica/gaddag && mv ~/NWL23.kwg ~/NWL23.klv2 data/lexica/gaddag/
export MACONDO_DATA_PATH=$PWD/data

# a 2-minute smoke test: label 4 positions
bin/simlabel sim -in ~/positions-train.jsonl.gz -out smoke.jsonl -threads 4 -limit 4

# the run: detached, restartable (rerun the same line after a crash or
# reboot; finished positions are skipped)
nohup bin/simlabel sim -in ~/positions-train.jsonl.gz -out labels.jsonl -threads $(nproc) > simlabel.log 2>&1 &
tail -f simlabel.log     # every 30 s: positions done, rate per hour, ETA
```

Several machines: the same command with `-shard i -shards n` on each
(i = 0..n-1); each writes its own labels file.

When done (or whenever you want a partial copy): `gzip -k labels.jsonl` and
copy `labels.jsonl.gz` home. A fresh clone was tested this way with only the
git-tracked data and the two lexicon files (10/5).

Size: the positions file holds 1,320,078 positions (4% of the 54M openings
games). At ~25-90 CPU-seconds each (25 on an idle core; ~90 here beside
the 6-ply match) a 128-core machine does ~5,000-18,000 an hour: all of it
in 3-10 days. Stop whenever; any number of labels is usable.

Running since 2026-10-06 00:03 on deb192 (dual EPYC 7K62, 96 cores / 192
threads, 503 GB): `~/macondo/simdistill/` holds the positions, `labels.jsonl`,
`simlabel.log`, `run.sh` (supervisor: restarts the labeller if it dies; after
a reboot run `setsid -f ~/macondo/simdistill/run.sh > /dev/null 2>&1 < /dev/null`)
and `status.sh` (progress). Go 1.26.1 in ~/sdk/go; NWL23.kwg was downloaded by
the shell (different node order from ours, the same 212,868 words; klv2
identical). 192 threads: ~67 s per position per thread, ~9,400-10,300
positions an hour, ~5.5 days for all 1.32M.

### 4. Training data  [Claude]  (built 10/5)

`simlabel frames -turns <log> -tag <tag> -positions ... -labels ... -out groups.bin`
replays each labelled game from the logs to the position, builds the net's
input for every candidate (the bot's own code) and writes one record per
position: the sim's win, equity, standard error and iterations per
candidate plus the packed rows. Decided positions (sim win span < 0.5 pt)
are skipped. ~135 KB per position; 1M positions ~135 GB, so for the full
run the trainer reads it memory-mapped (or it is built in parts).
`simlabel check` reports the served net's agreement with the sim.

### 5. Fine-tune  [Claude]  (trainer support built 10/5)

`training.py --init-ckpt best-tf-streamopen.pt --sim-groups groups.bin
--sim-val-groups groups-val.bin --groups-per-micro 2 --rank-weight 1
--rank-tau 2` plus the usual streamed data. Smoke-tested on the pilot
groups (20 steps): held-out agreement and sim win given up are printed at
every validation and written to `<csv>.rank.csv`.

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
