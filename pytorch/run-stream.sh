#!/usr/bin/env bash
# Streamed training: no frame cache. The producer rescans the gzipped turn
# logs PASSES times, drawing a fresh turn per game each pass, and pipes the
# frames straight into one trainer process; validation comes from the
# producer's held-out split (games whose ID hashes to 0 mod HOLDOUT_MOD),
# scanned once from VAL_LOG into a small cache. Then deploy and match.
#
#   TAG (stream), LOGS (nwl23{a,b,c}.txt.gz), VAL_LOG (nwl23c.txt.gz),
#   PASSES (6), HOLDOUT_MOD (20 = 5% of games held out), ROWS_PER_PASS
#   (36400365 x 0.95 by default; sets the schedule length), SPATIAL_SHARE
#   (0.1), TRANSPOSE (0), BATCH (128), ACCUM (16), VAL_SIZE (150000),
#   SHUFFLE (1024 frames per loader worker), DEVICE (auto|cuda|cpu)
set -o pipefail
cd "$(dirname "$0")"
source venv/bin/activate
log() { echo "$(date '+%F %T') $*"; }

TAG=${TAG:-stream}
LOGS=${LOGS:-"$HOME/data/nwl23a.txt.gz $HOME/data/nwl23b.txt.gz $HOME/data/nwl23c.txt.gz"}
VAL_LOG=${VAL_LOG:-$HOME/data/nwl23c.txt.gz}
PASSES=${PASSES:-6}
HOLDOUT_MOD=${HOLDOUT_MOD:-20}
ROWS_PER_PASS=${ROWS_PER_PASS:-34580000}
SPATIAL_SHARE=${SPATIAL_SHARE:-0.1}
TRANSPOSE=${TRANSPOSE:-0}
BATCH=${BATCH:-128}
ACCUM=${ACCUM:-16}
VAL_SIZE=${VAL_SIZE:-150000}
SHUFFLE=${SHUFFLE:-1024}
DEVICE=${DEVICE:-auto}
PRODUCER="../bin/mlproducer -labeler result -per-game -endgame-plies 2 -holdout-mod $HOLDOUT_MOD"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Training never shares the GPU with a match or with served models.
# (SMOKE=1: a CPU smoke test; skips the GPU guard and the deploy.)
SMOKE=${SMOKE:-0}
if [ "$SMOKE" != 1 ]; then
    while pgrep -f "[b]in/shell autoplay" >/dev/null; do sleep 60; done
    ./triton-models.sh unload-all
fi

VAL=val-$TAG.bin
if [ ! -s "$VAL" ]; then
    log "scanning the held-out games of $VAL_LOG into $VAL"
    zcat "$VAL_LOG" | $PRODUCER -split val 2> "producer-$TAG-val.log" | \
      python training.py --cache-only --cache "$VAL" 2> "cache-$TAG-val.log"
    log "  $(tail -1 "cache-$TAG-val.log")"
fi

STEPS=$(python -c "print(max(100, $ROWS_PER_PASS * $PASSES // ($BATCH * $ACCUM) // 1000 * 1000))")
log "training $TAG: $PASSES passes over $LOGS, ~$ROWS_PER_PASS rows/pass, $STEPS steps"
rm -f best-tf-$TAG*.pt "producer-$TAG.log" "producer-$TAG-passes.log"

# The trainer reads a FIFO; the passes are written into it by a feeder
# whose every producer must succeed. A producer that dies mid-frame leaves
# the stream out of step, so on any failure the feeder kills the trainer
# rather than let the next pass append to a broken stream.
FIFO=$(mktemp -u "stream-$TAG.XXXX.fifo"); mkfifo "$FIFO"
python training.py --arch transformer --ckpt "best-tf-$TAG.pt" --csv "loss_tf_$TAG.csv" \
  --primary wdl --w-wdl 1 --w-value 0 --aux-share 0.15 --spatial-share "$SPATIAL_SHARE" \
  --transpose-prob "$TRANSPOSE" --batch-size "$BATCH" --accum "$ACCUM" \
  --epochs 1 --val-cache "$VAL" --val-size "$VAL_SIZE" --shuffle-buffer "$SHUFFLE" \
  --total-steps "$STEPS" --snapshot-every 10000 --device "$DEVICE" < "$FIFO" 2>&1 | tee "train-tf-$TAG.log" > /dev/null &
TRAIN_PIPE=$!
(
  set -o pipefail
  exec > "$FIFO"
  for pass in $(seq 1 "$PASSES"); do
    if ! ( ( for f in $LOGS; do zcat "$f" || exit 1; done ) | $PRODUCER -split train 2>> "producer-$TAG.log" ); then
      echo "$(date '+%F %T') pass $pass FAILED (producer or zcat); killing the trainer" >> "producer-$TAG-passes.log"
      pkill -f "[t]raining.py --arch transformer --ckpt best-tf-$TAG.pt"
      exit 1
    fi
    echo "$(date '+%F %T') pass $pass scanned" >> "producer-$TAG-passes.log"
  done
) &
FEEDER=$!
wait $TRAIN_PIPE; TRAIN_RC=$?
if [ "$TRAIN_RC" = 0 ]; then
    # The trainer stops at --total-steps, normally a little before the last
    # pass ends: the feeder's producer then hits a closed pipe, which is
    # not a failure. Stop it and ignore its exit.
    pkill -P "$FEEDER" 2>/dev/null; kill "$FEEDER" 2>/dev/null
    pkill -f "[m]lproducer -labeler result -per-game -endgame-plies 2 -holdout-mod $HOLDOUT_MOD -split train" 2>/dev/null
    wait $FEEDER 2>/dev/null
    FEED_RC=0
else
    wait $FEEDER; FEED_RC=$?
fi
rm -f "$FIFO"
log "training exit=$TRAIN_RC (feeder exit=$FEED_RC)"
if [ "$TRAIN_RC" != 0 ]; then
    log "stream run failed; not deploying"
    exit 1
fi

[ "$SMOKE" = 1 ] && { log "smoke run done; not deploying"; exit 0; }
MODEL=macondo-nn-tf-$TAG CKPT=best-tf-$TAG.pt CSV=loss_tf_$TAG.csv TRAIN_LOG=train-tf-$TAG.log \
  EXP=tf-$TAG-v-hasty-pairs PRIMARY=wdl VAL_MAX=0.6 ./watch-tf-heads.sh
