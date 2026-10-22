# A Spanish value net (FILE2017)

Goal: `macondo-tf-spanish`, the transformer value net trained on FILE2017
HastyBot self-play, picked by Macondo by itself for any game with the
Spanish distribution (`game.ResolveMLModel`).

## What already works (Oct 22, claude/network-improvements)

- Features follow the alphabet (`game/mllayout.go`, `pytorch/features.py`).
  Spanish has 28 tiles besides the blank (A-Z less K and W, plus CH, LL, Ñ,
  RR as tiles of their own), so 3*28+7 = 91 planes and 2*28+20 = 76 scalars.
  The blank is counted as in English: its own plane for blanked tiles, the
  first slot of the rack and unseen scalars, the first power tile.
- Vowels come from the distribution: A E I O U.
- Power tiles: the blank, then the five highest-scoring tiles. Z is worth
  10, then five tiles tie at 8 (J, LL, Ñ, RR, X) and the tie goes by
  alphabet order, so X is left out. Its count is still in the unseen
  scalars, so the net can learn it; not worth a special case.
- `run-stream.sh` sets `MACONDO_LETTERS=28` for `FILE*`/`FISE*`.
- Tests: `TestSpanishMLLayout`, `TestSpanishMLVectors` (a FILE2017 game:
  float and byte paths agree, the digraph tiles land in their planes).
- Smoke run: 200 HastyBot games (digraphs logged as `[CH]`, `[LL]`,
  `[RR]`; leaves from `data/lexica/gaddag/FILE2017.klv2` in use) -> producer
  (21,456-float frames, as expected) -> 20 CPU steps -> ONNX -> FastMlBot
  played 6 games with `macondo-tf-spanish`. A throwaway net, not kept.

## Steps

1. **Generation speed.** Time 10k FILE2017 HastyBot games with the
   production flags (`gen` settings: quick endgame 2 plies margin 60,
   sampled openings). Spanish games are longer than English ones (more
   tiles: 100 + 2 blanks, and digraphs), so the English rate will not
   carry over. Generation is CPU-only and can run beside training.
2. **Leaves check.** HastyBot's games are only as good as its leaves.
   Look at `FILE2017.klv2` (where it came from, a few sanity values: ?,
   S-equivalents, Q without U, CH/LL/RR/Ñ leaves). If they look
   untrained, generate leaves first (MAGPIE autoplay leave generation);
   otherwise go ahead.
3. **Data.** 27M games, one batch, as the first English stream runs used.
   Doubling to 54M/81M only if the 27M net is still improving at the end
   of training (it did for English: 81M beat 27M h2h).
4. **Training.** From scratch, `run-stream.sh` with `LEXICON=FILE2017`,
   the current English recipe (transformer 8x192, WDL primary, aux and
   spatial heads). No warm start from the English net: its input layer
   is shaped for 85 planes, and a transplant would be a second change in
   the same run. That could be a later experiment (map the shared
   letters' planes over) if Spanish data turns out to be the bottleneck.
5. **Match.** FastMlBot (the new net) vs HastyBot, 100k pairs, FILE2017,
   win % with a paired 95% interval. English nets got 57-58% here.
6. **Ship.** `copy_model.sh` as `macondo-tf-spanish` (config.pbtxt sized
   from features.py with `MACONDO_LETTERS=28`), commit `model.onnx`,
   README model table, `build-engines.sh`.

## Scheduling

The GPU is busy with csw24huge (results about Oct 24). If csw24huge wins,
the NWL23 108M run is next in line; Spanish training queues behind
whichever of those the user picks. Generation (steps 1 and 3) is CPU-only
and can start sooner, but not while the CPU is generating other batches.

## Open questions

- Lexicon: FILE2017 is the one on disk (and what Woogles plays). FISE2
  would use the same distribution and the same net.
- MAGPIE would generate faster, but the network branch is master-based;
  generate on master, so the producer rows need no parity check.
