#!/usr/bin/env bash
# The collaborator's recipe, on the existing logs: value target = the
# mover's real game result, one position per game (a turn drawn from
# 1..30), no rollouts; a sampled position whose bag was already empty is
# labeled by a 2-ply quick endgame search instead of the logged (greedy)
# outcome. Same trainer, five epochs; the wdl head is off since it would be
# the value target again. Queued behind generation 1.
#
# ~6.8M games in the file, ~80% emit a position -> ~5.4M labels;
# 5.3M / 2048 ~ 2,600 steps per epoch x 5, so the schedule stops at 12,500.
# The wait pattern must match generation 1's processes but not this script,
# which is itself a run-gen1.sh.
cd "$(dirname "$0")"
TAG=result \
WAIT_FOR="labels-out gen1-labels|ckpt best-tf-gen1|watch-tf-heads.sh|bin/shell autoplay" \
POSITIONS=999999999 \
EPOCHS=5 \
STEPS=12500 \
PRODUCER_ARGS="-labeler result -per-game -endgame-plies 2" \
TRAIN_ARGS="--w-wdl 0" \
VAL_MAX=0.5 \
exec ./run-gen1.sh
