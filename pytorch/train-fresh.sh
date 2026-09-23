#!/usr/bin/env bash
# Train from an existing frame cache the collaborator's way: WDL head as
# the primary objective (3-logit cross-entropy), value head off, the
# exported 'value' = P(win) - P(loss). Waits for a running cache build to
# finish, gzips the turn log in the background, trains five epochs, then
# deploys and runs the paired match.
#
#   TAG (default fresh), EPOCHS (5), VAL_SIZE (150000), WAIT_FOR (the
#   cache-only writer), TRAIN_ARGS (extra trainer flags)
set -o pipefail
cd "$(dirname "$0")"
source ./venv/bin/activate
log() { echo "$(date '+%F %T') $*"; }

TAG=${TAG:-fresh}
EPOCHS=${EPOCHS:-5}
VAL_SIZE=${VAL_SIZE:-150000}
WAIT_FOR=${WAIT_FOR:-"[t]raining.py --cache-only --cache $TAG-frames.bin"}
TRAIN_ARGS=${TRAIN_ARGS:-"--primary wdl --w-wdl 1 --w-value 0 --aux-share 0.15"}
PRIMARY=${PRIMARY:-wdl}
VAL_MAX=${VAL_MAX:-0.6}

while pgrep -f "$WAIT_FOR" >/dev/null; do sleep 60; done
[ -f "$HOME/data/$TAG.txt" ] && ( cd "$HOME/data" && nohup gzip "$TAG.txt" > /dev/null 2>&1 & )

ROWS=$(python -c "import os; print(os.path.getsize('$TAG-frames.bin') // 2699)")
STEPS=$(python -c "print(max(100, ($ROWS - $VAL_SIZE) * $EPOCHS // 2048 // 1000 * 1000))")
log "training $TAG: $ROWS rows, $EPOCHS epochs, $STEPS steps, $TRAIN_ARGS"
rm -f best-tf-$TAG*.pt
python training.py --arch transformer --ckpt "best-tf-$TAG.pt" --csv "loss_tf_$TAG.csv" \
  $TRAIN_ARGS --epochs "$EPOCHS" --from-cache "$TAG-frames.bin" --val-size "$VAL_SIZE" \
  --total-steps "$STEPS" --snapshot-every 10000 2>&1 | tee "train-tf-$TAG.log" > /dev/null
log "training exit=${PIPESTATUS[0]}"

MODEL=macondo-nn-tf-$TAG CKPT=best-tf-$TAG.pt CSV=loss_tf_$TAG.csv TRAIN_LOG=train-tf-$TAG.log \
  EXP=tf-$TAG-v-hasty-pairs PRIMARY=$PRIMARY VAL_MAX=$VAL_MAX ./watch-tf-heads.sh
