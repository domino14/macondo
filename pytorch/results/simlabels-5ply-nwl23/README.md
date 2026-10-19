# 5-ply sim labels, NWL23 (October 2026)

Archive: `~/data/results/simlabels-5ply-nwl23/` on the home machine (hard
links to `~/data/simdistill/`); a second copy of the labels stays on deb192
in `~/macondo/simdistill/`. Made for the sim distillation
(shelved, see `pytorch/SHELVED.md`), kept for other uses.

## What it is

1,320,078 mid-game positions, each with the 5-ply sim's verdict on its 50
best plays by static equity: 66M simulated candidates. Labelling took
~124 h on deb192 (2x EPYC 7K62, 192 threads), ~82 CPU-seconds a position.

**Positions** (`positions-open.jsonl`, `positions-open2.jsonl`; the same
set gzipped as `positions-train.jsonl.gz`): one per game from the open and
open2 HastyBot-vs-HastyBot sets (54M games, sampled openings), a turn from
the last sampled opening ply to turn 30 with at least 9 unseen tiles.
Fields: `key` (tag:gameID:turn), `game`, `turn`, `cgp` (the position with
the mover's rack; opponent's rack hidden), `played` (HastyBot's move),
`result` (mover's true result, 1/0/-1), `spread` (now), `final` (final
spread), `unseen`. Games whose ID hashes to 0 mod 20 were held out:
`positions-val.jsonl.gz`, 68,932 positions, not labelled except the
558-position pilot (`pilot-positions-val.jsonl`, `pilot-labels-val.jsonl`).

**Labels** (`labels-full.jsonl.gz`, 3.7 GB, 12.9 GB unzipped; 1,320,080
lines, 2 duplicates from restarts): `key`, `plies` (5), `iterations`
(total, ~4,000 a position), `seconds`, and `cands`, best sim win first,
each with `move`, `score`, `static_eq`, `win` (sim win probability for the
mover) and `win_se`, `eq` (sim equity: spread change plus leftover) and
`eq_se`, `iters`, `pruned`.

How the sim ran: `cmd/simlabel sim`; fixed 5 plies, top 50 by static
equity, Stop99 stopping rule (about 92% of candidates are pruned after
128-256 iterations, so only the top few are simmed to the end: trust
`win` in proportion to `iters`), opponent racks drawn uniformly from the
unseen tiles (no inference), no endgame solver, end of line scored with
the win-percentage table (`data/strategy/winpct.csv` refit, "base").

## Uses so far

- simft, simft2: listwise ranking fine-tunes of streamopen on the first
  485k positions (57.48 and 57.18 vs streamopen 57.65: no gain).
- 10/13: top-16 candidate groups (`simlabel frames -top 16`) for a top-k
  loss.

## Ideas

Rare-and-expensive data: a 5-ply verdict on 50 plays per position. Also
useful for: checking a bot's move choice against the sim
(`simlabel check`), fitting or checking static evaluators (leaves, the
win% table), a policy head target, and as a benchmark of how often a new
net picks the sim's best play.
