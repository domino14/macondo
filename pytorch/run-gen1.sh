#!/usr/bin/env bash
# Generation 1 of the bootstrapped-label plan, end to end and unattended:
#
#   1. wait for the tf-heads2 match to release the GPU,
#   2. label 25% of the first 40M positions of the autoplay log with 16
#      two-ply rollouts each, the tf-heads2 net at the leaf (~10M labels),
#      streaming the frames into the trainer for epoch 1 and caching them,
#   3. train four more epochs from the cache (schedule matched to 5 epochs),
#   4. deploy as macondo-nn-tf-gen1 and run the 100k-pair match vs HastyBot.
#
# Logs: producer-gen1.log, train-tf-gen1.log, watch-tf-gen1.log; labels in
# gen1-labels.csv (gameID,turn,value,spread); frames in gen1-frames.bin.
# Overridable: TAG (names every output), POSITIONS, EPOCHS, STEPS, WAIT=0
# (don't wait for the match), WATCH=0 (don't deploy/match afterwards).
set -o pipefail
cd "$(dirname "$0")"
source venv/bin/activate

log() { echo "$(date '+%F %T') $*"; }

TAG=${TAG:-gen1}
POSITIONS=${POSITIONS:-40000000}   # lines of the log; 25% sampled -> ~10M labels
EPOCHS=${EPOCHS:-5}
# ~10M labels / 2048 per step ~ 4,880 steps per epoch; stop a little short
# so the cosine reaches zero before the data does.
STEPS=${STEPS:-24000}
VAL_SIZE=${VAL_SIZE:-150000}

if [ "${WAIT:-1}" = 1 ]; then
    while pgrep -f "bin/shell autoplay.*tf-heads2-v-hasty-pairs" >/dev/null; do
        sleep 60
    done
    python pairs-stats.py ../games-tf-heads2-v-hasty-pairs.txt
fi
log "starting $TAG: labeling + training ($POSITIONS positions, $EPOCHS epochs, $STEPS steps)"

export MACONDO_TRITON_URL=localhost:8101
export MACONDO_TRITON_MODEL_NAME=macondo-nn-tf-heads2
export MACONDO_TRITON_MODEL_VERSION=1

rm -f best-tf-$TAG*.pt $TAG-frames.bin
head -n $POSITIONS ~/data/autoplay-softmax-v-hasty-5.txt | \
  ( ../bin/mlproducer -labeler rollout -plies 2 -rollouts 16 -sample 0.25 \
      -labels-out $TAG-labels.csv 2> producer-$TAG.log ; echo "producer exit=$?" >&2 ) | \
  ( pv -br ; echo "pv exit=$?" >&2 ) | \
  ( python training.py --arch transformer --ckpt best-tf-$TAG.pt --csv loss_tf_$TAG.csv \
      --aux-share 0.15 --epochs $EPOCHS --cache $TAG-frames.bin --val-size $VAL_SIZE \
      --total-steps $STEPS --snapshot-every 5000 ; echo "training exit=$?" >&2 )
log "pipeline exit=$?"
tail -1 producer-$TAG.log

if [ "${WATCH:-1}" = 1 ]; then
    MODEL=macondo-nn-tf-$TAG CKPT=best-tf-$TAG.pt CSV=loss_tf_$TAG.csv \
      TRAIN_LOG=train-tf-$TAG.log EXP=tf-$TAG-v-hasty-pairs ./watch-tf-heads.sh
fi
