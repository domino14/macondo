# Plan: bootstrapped labels + auxiliary heads for the value net

Written 2026-09-19, after the transformer matched the CNN at the board
(52.33% ± 0.22 vs HastyBot over 100k pairs; CNN 52.5%). Two very different
architectures converging on the same number means the ceiling is in the
labels, not the network. This plan attacks the labels. It folds in a
collaborator's three points: averaged K-ply bootstrapped targets, a
score-differential head plus a win/draw/loss head (KataGo §4.1), and
per-game position sampling (KataGo §3.1).

## 0. What the pipeline does today

- `cmd/mlproducer` replays autoplay logs (`~/data/autoplay-softmax-v-hasty-*.txt`,
  12 columns per turn). For every ply it builds the feature vector for the
  position *after* the mover's play, from the mover's side, opponent's rack
  thrown back in the bag (`updateBoardAndExtractFeatures`).
- Label (`makeTrainingVector`): walk `NPlies = 5` real plies forward, take
  the mover's spread and the bag count there, look up `winpct.csv`
  (spread x bag -> win%), rescale to [-1, 1]. If the game ended inside the
  window, use the real result. One continuation, one set of draws, half of
  the replies played by the softmax bot.
- Four targets are emitted: `[bogowin, total_points/1600, opp_bingoed_next,
  opp_next_score/300]`. Only the first is trained on. The other three heads
  exist in `training.py` but are commented out (June 2025, "trying more
  heads", no result recorded in experiments.md).
- Frames go through a 20k-vector shuffle buffer (100k was better but OOMed
  with the 96-ch CNN).
- `training.py` trains CNN or transformer with smooth-L1 on the single
  value head; checkpoints on val loss. `export.py` exports one output.

## 1. Why 5 plies is no longer needed

The 5-ply walk was doing two jobs at once:

1. **Getting past the draw noise.** Spread one ply after our move is mostly
   "what did the opponent draw". Walking further lets that average out a
   little, at the cost of adding more draws.
2. **Getting to a point where the table is trustworthy.** `winpct.csv`
   sees only spread and bag count. It knows nothing about the board, the
   leave, or the unseen pool. At ply 0 it is a terrible evaluator, at ply 5
   the spread has absorbed some of that information.

Averaging N sampled continuations does job 1 directly, and a net at the leaf
does job 2 directly, because the net already sees the board, the leave and
the unseen tiles. That is the same reason BestBot needs 5 plies: its leaf
evaluator is spread + leave value, which is weak, so it has to look far. A
strong leaf evaluator lets you look shallow. TD-Gammon, AlphaZero and KataGo
all use K of 1 to a few plies for exactly this reason.

**Choice of K.** Positions are stored from the mover's side, right after
the play, before the draw. With K=1 the leaf is the opponent's post-move
position and the label is `-V(leaf)`. With K=2 the leaf is back on our side
and the label is `+V(leaf)`. Both are valid. Start with **K=2, N=16** and
make both flags, then run K=1 as the first ablation:

- K=2 is the exact shape of the "2-ply sim with the net at the leaf" that
  we want for BestBot anyway, so one rollout routine serves both.
- K=2 puts one of our own real replies in the label, which reduces how
  much of the label is the net's own opinion (less self-reinforcing bias
  per generation).
- K=1 is half the movegen cost and worth measuring; the collaborator
  expects it to win once the leaf evaluator is good.

Rollout plies are played by the static best play (`GenBestStaticTurn`,
i.e. HastyBot), not the softmax bot. That decouples position diversity
(from the softmax games) from label quality (from best play).

**Cost.** Per label: N x (draw racks + K movegens ~0.25 ms each + 1 vector
build ~0.1 ms) on CPU, plus N net evaluations on the GPU. The GPU is the
binding constraint: the transformer through TensorRT fp16 does roughly
10k positions/s.

| N  | K | CPU per label | labels/s (16 cores) | labels/s (GPU cap) | 10M labels |
|----|---|---------------|---------------------|--------------------|------------|
| 16 | 2 | ~10 ms        | ~1600               | ~600               | ~5 h       |
| 32 | 2 | ~20 ms        | ~800                | ~300               | ~9 h       |
| 16 | 1 | ~6 ms         | ~2700               | ~600               | ~5 h       |

So 10M clean labels per generation is an overnight job. Today's run used
163M noisy labels; 10M averaged ones is very likely worth more. Using the
CNN as the leaf evaluator would triple GPU throughput if needed.

## 2. Heads

Keep the value head as the thing the bot ranks on. Add:

| head       | target                                         | loss        | activation | weight |
|------------|------------------------------------------------|-------------|------------|--------|
| value      | bogowin (gen 0) / mean rollout value (gen 1+)  | smooth-L1   | tanh       | 1.0    |
| spread     | spread delta over the horizon, tanh(x/130)     | smooth-L1   | tanh       | 0.5    |
| wdl        | final game result for the mover (3 classes)    | cross-ent   | softmax    | 0.25   |
| opp_bingo  | opponent bingos next turn (already emitted)    | BCE         | logit      | 0.1    |
| opp_score  | opponent's next score /300 (already emitted)   | smooth-L1   | linear     | 0.1    |

Drop `total_game_points`: it is a function of the inputs (both scores are
implied by spread + turn history), so it is not a useful auxiliary target.

Why this should work when the June 2025 attempt did not (no result was
recorded, so this is diagnosis from the diff, not a post-mortem):

- The old weights (points 0.5, bingo 0.3, opp_score 0.3) put more than half
  the gradient on auxiliaries with different loss scales, so the value head
  was being crowded. Weights above are small and each loss is on a [-1,1]
  or probability scale.
- Checkpoint on **value val loss only**, never on total loss, so runs stay
  comparable with 0.0913 / 0.0917 and a noisy auxiliary can't pick the
  checkpoint.
- Log every head's val loss in its own CSV column so a bad head is visible
  immediately.
- Export only `value` and `spread` to ONNX; Go keeps reading `value` by
  name, so the bot and Triton config don't change unless we want the
  spread output (the simmer can use it for its equity stat).

The spread and wdl heads are the KataGo §4.1 argument: score is a dense,
low-noise signal that regularizes the trunk; win/draw/loss is what we
actually care about. With bootstrapped labels the wdl head is also the
only target anchored in real outcomes rather than the net's own opinion,
which is the guard against generation-over-generation drift.

## 3. Position sampling (the QANAT point)

The worry is about the **label**, not the input. Yes, the QANAT board is
in the training data either way. The problem arises only when every
position from a game carries the *same* label (the game result): then the
net sees 24 examples of "QANAT on the board -> win" and can learn that
spurious association. Our current 5-ply label is different for each of
those 24 positions, and adjacent positions share only 4 of 5 future plies,
so the association isn't there; what remains is mild label correlation
between neighbours, which costs sample efficiency but not bias. The shuffle
buffer exists for that.

Two things change that:

- The **wdl head is a game-outcome label**, so it *does* have the QANAT
  problem. Sampling a fraction of positions per game is the mitigation,
  and it's cheap because we have far more positions than we can label.
- The rollout labeler is expensive per position, so sampling 25% is also
  the cost control (KataGo's "fast mode" for the unsampled 75% is, for
  us, simply not labeling them; the game was already played).

So: `-sample 0.25`, a per-position coin flip. Do not enable it in Phase 1
so the heads change is measured on its own (one variable at a time; the
tau-schedule lesson).

## 4. Phases

### Phase 0: bookkeeping (now)

- Commit the transformer work (`transformer_model.py`, training.py CLI,
  export/TensorRT changes) and log the 52.33% ± 0.22 pairs result in
  `experiments.md`.

### Phase 1: heads on existing data (~2 days incl. a training run)

Go (`cmd/mlproducer`):
- Buffer a whole game before emitting (needed for the final result). Cost:
  ~25 vectors x 77 KB per live game per worker; fine.
- Emit 5 targets: `[bogowin, spread_delta_norm, wdl(-1/0/1), opp_bingo,
  opp_score]`. `spread_delta` is the commented-out `futureSpread - nowSpread`
  passed through `NormalizeSpreadForML`.
- Fix the `unseenVector` divide-by-zero when the bag is empty.

Python:
- `N_TARGETS = 5`; head modules on both archs (transformer: from the
  `ln_f(CLS)` -> shared 128 hidden -> one linear per head).
- `compute_loss` with per-head weights as CLI flags (`--w-spread 0.5`
  etc.), per-head CSV columns, checkpoint on `value_loss`.
- `export.py`: outputs `value`, `spread`; keep `copy_model.sh` and
  `config.pbtxt` in step.

Run: transformer, same file, same hyperparameters as the 0.0913 run, so
the value val loss is directly comparable. Then the 100k-pair HastyBot
match. Expected: a small gain (tenths of a point); the real payoff is the
spread head and the regularization for Phase 2.

### Phase 2: rollout labeler (~1 week)

New `cmd/mlproducer -labeler rollout -K 2 -N 16 -sample 0.25`:
- Per sampled position (game replayed as today, racks thrown in, bag =
  all unseen): N times, draw our replenishment and the opponent's rack
  uniformly from the bag, play K plies of static best play, build the
  leaf vector from the mover-at-the-leaf's side, and batch all leaves to
  Triton. Label = mean of leaf values (negated when K is odd), with real
  results when the game ends inside the rollout.
- Spread target = mean over rollouts of (K-ply spread delta +
  leaf spread-head output), i.e. the spread is bootstrapped the same way.
- wdl from the real game, sampled positions only.
- Leaf evaluator: the Phase 1 transformer (`macondo-nn-tf`), via the
  existing Triton client, batched N x positions per request (respect the
  128-batch engine profile).
- Reuse `aiturnplayer.GenBestStaticTurn` and the movegen per worker;
  don't drag in the full `Simmer` (its stats and logging are dead weight
  here).

Train the transformer from scratch on the gen-1 labels (10M positions,
several epochs are fine now that the labels are clean; keep a held-out
slice of games). Match vs HastyBot and, with game pairs, vs the Phase 1
net directly.

### Phase 3: iterate

Swap the new net in as the leaf evaluator, relabel, retrain. Two or three
generations. Stop when the pairs match against the previous generation is
flat. Keep a fixed gold set (e.g. 20k positions labeled with N=256) so
val losses are comparable across generations; the streaming val set
changes every time the labeler changes.

### Phase 4: what the rollout code unlocks

- The 2-ply ML-leaf simmer for BestBot (same routine, candidates instead
  of the logged play, ranking by mean leaf value).
- A policy head (score every legal move from one encoder pass) once we
  have per-move targets from sims.

## 5. Open questions to settle by experiment, not argument

- K=1 vs K=2 (cost vs bias).
- N=16 vs N=32 (does the extra averaging show up at the board?).
- Rollout player: static best vs the net choosing (Option A, ~10x slower).
- Whether to keep the bogowin table as a fallback leaf when the bag is
  small (the net was never trained on bag=0 positions since the endgame
  uses HastyBot).
