# Plan: features borrowed from Scribblez (eigen7, state at 2026-07-31)

Source: holistic read of Scribblez at commit 52458bcf. Its spatial input is
plane-for-plane ours (85 planes: 26 letters, blank, 26+26 cross-checks, 4
premiums, 2 placement planes). Its labels are now what ours are (WLD 3-logit
on the true result, no endgame positions). The differences that could matter
are in augmentation, auxiliary supervision, data reuse, and how self-play
games are diversified. This plan takes the four that are cheap relative to
their expected value, in the order they can be tested.

Baseline to beat: heads2 under NWL23, 53.31% ± 0.18 (100k pairs vs HastyBot).

## 1. Diagonal-transpose augmentation (trainer only, test on the existing cache)

**Status 2026-09-26: already in place at game level.** The producer replays
half of all games transposed (`shouldTranspose` in
`cmd/mlproducer/game_assembler.go`, by game-ID hash), so every run so far
was augmented this way. A per-position flip in the trainer would only show
each position in both orientations across epochs; not worth a dedicated
run. Keep it as a free add-on inside step 2 (it must transpose the new
spatial targets too).

What: with probability 0.5 per row, transpose every 15x15 plane across the
main diagonal and swap the horizontal cross-check planes (27..52) with the
vertical ones (53..78). Scalars are untouched (nothing in the 72 scalars is
directional). The main diagonal is the only symmetry that preserves word
reading order, so it is the only one available.

Why: the premium layout and the game are exactly symmetric under it, so it
doubles the effective data for free, and it stops the transformer's 225
learned position embeddings from encoding "rows are different from columns".
The original CNN trained this way; the transformer never has.

Where: `unpack_batch` in `pytorch/training.py`, on the GPU, after the bits
are unpacked: build a flip mask, `board.transpose(2, 3)` with a fixed channel
permutation, `torch.where`. No producer or cache change. Also apply at
export-parity time only in the identity orientation (inference never flips).

Test: retrain the heads2 configuration on the nwl23 cache with the flag on,
same steps and schedule, 100k-pair match. Expected: small positive or
neutral. Cost: one training run.

Trainer flag: `--transpose-aug` (default on once it is shown not to hurt).

## 2. Dense spatial auxiliary heads (producer + cache + model)

What: four 15x15 per-square binary targets, one bit per square each, taken
from the game log at the sampled position:

- `opp_next_placement`: squares the opponent's next move covers.
- `self_next_placement`: squares the mover's own next move covers.
- `opp_win_placement`: the first, zeroed unless the opponent went on to win.
- `self_win_placement`: the second, zeroed unless the mover went on to win.

A pass or exchange gives an all-zero plane. Loss: per-square BCE, weight
shared through `--aux-share` like the other auxiliary heads (start at 0.5 of
the total auxiliary share for the two marginals and 0.5 for the two
conjunctions, then let the gradient balancer set the actual weights).

Why: today the trunk learns from one number per position. These heads give
it 900 supervised bits per position from data we already have, and they ask
for exactly the board knowledge a value net needs but can only reach
indirectly through the outcome: where the hot spots are, which lanes the
opponent will use, which squares are worth blocking, where the mover's own
follow-up lands. KataGo's ownership head is the precedent; it was the single
largest data-efficiency gain in that paper. Scribblez trains them at weight
0.5 each and reads them for its sim-evidence loop, but their value to us is
as regularizers of the square tokens.

Where:

- Producer (`cmd/mlproducer/game_assembler.go`): the window already holds
  `plies[next]` (opponent's reply) and the ply after it (mover's next). Emit
  the four planes as packed bits appended to the frame. Per-game mode has
  the full history, so the two next moves are always available except at
  game end (all zeros, like Scribblez).
- Cache format (`pack_rows` / `unpack_batch`): +900 bits = 113 bytes per row
  (2,699 -> 2,812 B, +4%). Unpack with the same shift trick as the planes.
  Bump the cache version so old caches are refused.
- Transformer (`pytorch/transformer_model.py`): the 225 square tokens exist
  already. Add `nn.Linear(d, 4)` on the final-normed square tokens; output
  `(B, 4, 15, 15)` logits. CNN: a 1x1 conv on the last feature map.
- `compute_loss`: BCE-with-logits per plane, mean over squares; the
  transpose in step 1 must transpose these targets too.
- Export: leave them out of the ONNX (`EXPORTED_HEADS` unchanged), so Triton
  and the bot see nothing new. Scribblez never reads them for move selection
  either; its agent ranks on P(win)+0.5P(draw) only.
- Head form: one `Linear(d, 4)` shared across the 225 square tokens (his 1x1
  conv is the same thing). If per-cell BCE proves inert (98% of squares are
  empty; he saw magnitude errors and moved to a softmax over 2,927 footprint
  classes in Aug 2026), the footprint formulation is the fallback.

Test: needs a producer re-run over the nwl23 logs (about 10 h at the current
scan rate), then one training run and a match. Do it together with step 3
so the re-run is paid once.

## Parked for now (user decision 2026-09-26: only steps 1 and 2)

### 3. Several distinct positions per game (producer flag, disk)

What: `-per-game -picks K` in the producer: draw K distinct turns per game
from the eligible range instead of one. Train one epoch over the K-fold
cache instead of 3 epochs over a 1-fold cache.

Why: Scribblez's finding is that a game reused about 4 times at 4 different
turns trains fine, and about 40 passes memorized outcomes (train accuracy
kept rising while held-out quality and play strength fell). Our 3-epoch runs
re-see the same row three times; three different positions from the same
game carry the same outcome bit but different boards, so they are the safer
way to spend the same GPU time and get three times the coverage of each
game's states.

Cost: the cache grows K-fold. At K=3 the 20M-row nwl23 cache becomes 60M
rows, about 170 GB at the new row size; K=2 is about 113 GB. Check free
NVMe first (about 171 GB at last look). Producer time is nearly unchanged;
replay dominates and a second pick only adds a vector build.

Test: heads2 configuration, K=3, one epoch, transpose on, spatial heads on.
Compare to the step-1 run to isolate steps 2+3 from step 1.

### 4. Random openings in generation (autoplay flag, new batch)

What: `-random-opening-mean M` in autoplay. Per game, K ~ round(Exp(M))
opening plies are drawn uniformly from all legal placements plus all legal
exchanges (Scribblez uses M=2; about 22% of games get no random ply). The
game record carries K, and the producer's eligible range starts at the ply
after the last random one, so no label is contaminated by a random move.

Why: HastyBot self-play visits a narrow set of states. Random openings reach
odd boards and odd leaves with clean outcome labels, because everything
after the last random ply is agent play. Our current diversifier is softmax
sampling (T=1 over the top 50 by equity while the bag is over 60), which
puts noise into the labels of every position before the last sampled move
and only reaches positions near the equity policy. Random openings are the
cleaner instrument; the two can also be combined.

Where: `automatic/` (game loop: before turn K, pick uniformly from
`GenerateMoves(all)` plus exchanges), the turn log (record K per game), the
producer's `wanted` predicate (`pick` drawn from K+1..30). The greedy bots
should then play the rest with the existing quick 2-ply endgame.

Test: needs a fresh batch. Start with a matched-size A/B at 20M games each:
(a) current temperature sampling, (b) random openings with greedy bots,
same producer settings (K=3 picks, spatial targets). Train both, match
both. Cost: about 13 h of generation per arm at 435 games/s plus producer
and training. This is the next generation batch, so it is the moment to
decide on the parked Ising board-shape explorer too (see memory note).

## Not taken, and why

- Global-pooling residual blocks: the transformer's attention already has
  global reach at every layer.
- The contingent-draw lexical map: Scribblez ships it off by default; its
  reported result did not use it.
- Unseen-pool thermometer and unclipped score scalar: representational
  choices with no evidence either way; ours already carry the same
  information.
- Score-diff Gaussian head: Scribblez cut its weight to 0.0002, effectively
  off; our balanced spread head already sits at a small share.
- Endgame solver at match time: it would raise FastMlBot's headline number
  (Scribblez's solver alone is worth about 2 points against greedy
  endgames) but says nothing about the net. Keep matches net-only so runs
  stay comparable with the 53.31% baseline; decide separately whether the
  production bot should get the quick endgame.

## Order and cost

| Step | Code | Data | Test cost |
|---|---|---|---|
| 1 transpose | trainer only | none | 1 train + 1 match |
| 2 spatial heads | producer, cache, model, loss | producer re-run (~10 h) | shared with 3 |
| 3 K picks/game | producer flag | cache x K on NVMe | 1 train + 1 match |
| 4 random openings | autoplay, turn log, producer | new 2 x 20M-game batch | 2 train + 2 match |

Steps 1 and 2+3 can be built while the current nwl23 top-up batch and match
finish; step 4 waits for the generation slot.
