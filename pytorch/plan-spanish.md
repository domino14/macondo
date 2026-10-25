# A Spanish value net (FILE2017)

Goal: `macondo-tf-spanish`, the transformer value net trained on FILE2017
HastyBot self-play, picked by Macondo by itself for any game with the
Spanish distribution (`game.ResolveMLModel`).

## What the code already handles

- Features follow the alphabet (`game/mllayout.go`, `pytorch/features.py`).
  Spanish has 28 tiles besides the blank (A-Z less K and W, plus the
  digraphs CH, LL, RR and the letter Ñ as tiles of their own), so
  3*28+7 = 91 planes and 2*28+20 = 76 scalars, a 21,456-float frame.
  The blank is counted as in English: its own plane for blanked tiles, the
  first slot of the rack and unseen scalars, the first power tile.
- Vowels come from the distribution: A E I O U.
- Power tiles: the blank, then the five highest-scoring tiles. Z is worth
  10, then five tiles tie at 8 (J, LL, Ñ, RR, X) and the tie goes by
  alphabet order, so X is left out. Its count is still in the unseen
  scalars, so the net can learn it; not worth a special case.
- `run-stream.sh` sets `MACONDO_LETTERS=28` for `FILE*`/`FISE*`.
- Tests: `TestSpanishMLLayout`, `TestSpanishMLVectors` (a FILE2017 game:
  float and byte paths agree, the multi-letter tiles land in their planes).
- Leaves: HastyBot uses `data/lexica/gaddag/FILE2017.klv2`, which looks
  trained (Q -22.2, QU -10.6, Z +7.2, [CH] -6.4, ?S +26.0, AEINRS +32.3).

## Steps

1. **Data.** 27M FILE2017 HastyBot games with the production generation
   settings (`gen-openings.sh`: sampled openings, quick 2-ply endgames,
   margin 60). Generate on master-based code, so the producer's frames
   need no parity check against another movegen. The producer keeps about
   0.6 training rows per game.
2. **Training.** From scratch, `run-stream.sh` with `LEXICON=FILE2017`
   and the current English recipe (transformer 8x192, WDL primary, aux and
   spatial heads), 6 passes. No warm start from an English net: its input
   layer is shaped for 85 planes, and a transplant would be a second
   change in the same run. That could be a later experiment (map the
   shared letters' planes over) if Spanish data turns out to be the
   bottleneck.
3. **Match.** FastMlBot (the new net) vs HastyBot, 100k pairs, FILE2017,
   win % with a paired 95% interval. English nets get 57-58% here.
4. **More data, if it pays.** In NWL23 and CSW24, 81M games beat 54M head
   to head (about 50.6%) and 108M (CSW24) added nothing. If the 27M
   Spanish net is still improving at the end of training, generate more
   and compare head to head.
5. **Ship.** `copy_model.sh` as `macondo-tf-spanish` (config.pbtxt sized
   from features.py with `MACONDO_LETTERS=28`), commit `model.onnx`,
   README model table, `build-engines.sh`.

## Open questions

- Lexicon: FILE2017 is the one on disk (and what Woogles plays). FISE2
  would use the same distribution and the same net.
