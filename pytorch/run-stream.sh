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
#   (0.1), TRANSPOSE (0), BATCH x ACCUM (probed: fastest of 128x16, 256x8,
#   256x8 --compile; set BATCH/ACCUM/COMPILE/GPU_MEM to skip the probe),
#   VAL_SIZE (150000), SHUFFLE (1024 frames per loader worker), DEVICE
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
BATCH_SET=${BATCH:-}${ACCUM:-}
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
    while pgrep -f "[b]in/shell autoplay.*FAST_ML_BOT" >/dev/null; do sleep 60; done
    ./triton-models.sh unload-all
fi

VAL=${VAL:-val-$TAG.bin}
if [ ! -s "$VAL" ]; then
    log "scanning the held-out games of $VAL_LOG into $VAL"
    zcat "$VAL_LOG" | $PRODUCER -split val 2> "producer-$TAG-val.log" | \
      python training.py --cache-only --cache "$VAL" 2> "cache-$TAG-val.log"
    log "  $(tail -1 "cache-$TAG-val.log")"
fi

# Speed probe: unless BATCH/ACCUM/COMPILE are given, time a few dozen steps of
# each candidate on the validation cache and take the fastest that runs.
# The effective batch is BATCH x ACCUM = 2048 in every candidate, so the
# choice changes speed and GPU memory only, not the recipe.
COMPILE=${COMPILE:-}
GPU_MEM=${GPU_MEM:-}
if [ -z "$PROBE_DONE" ] && [ "$SMOKE" != 1 ] && [ -z "${BATCH_SET}${COMPILE}${GPU_MEM}" ]; then
    best_rate=0; best=""
    for cand in "128 16 0 4096" "256 8 0 5120" "256 8 1 5120"; do
        set -- $cand; b=$1; a=$2; c=$3; m=$4
        flag=""; [ "$c" = 1 ] && flag="--compile"
        log "probe: batch $b x $a${flag:+ $flag} (cap $m MiB)"
        rm -f probe-$TAG.pt probe-$TAG.csv
        # 120 steps, reported at 60 and 120: the steady rate is the second
        # interval's, which excludes start-up and torch.compile time.
        if python training.py --arch transformer --from-cache "$VAL" --val-size 2048 --val-every 60 \
             --total-steps 120 --batch-size $b --accum $a $flag --gpu-mem-mib $m \
             --primary wdl --w-wdl 1 --w-value 0 --aux-share 0.15 --spatial-share "$SPATIAL_SHARE" \
             --ckpt probe-$TAG.pt --csv probe-$TAG.csv > "probe-$TAG-$b-$a-$c.log" 2>&1; then
            rate=$(python - "probe-$TAG-$b-$a-$c.log" <<'PY'
import re, sys
r = [int(x.replace(",", "")) for x in re.findall(r"([0-9,]+) pos/s", open(sys.argv[1]).read())]
if len(r) >= 2 and r[0] > 0 and r[1] > 0:
    n = 60 * 2048
    print(int(n / (2 * n / r[1] - n / r[0])))
PY
)
            peak=$(grep -oE "peak GPU memory: [0-9.]+ GiB" "probe-$TAG-$b-$a-$c.log" | tail -1)
            log "  ${rate:-?} pos/s, $peak"
            if [ -n "$rate" ] && [ "$rate" -gt "$best_rate" ]; then best_rate=$rate; best="$b $a $c $m"; fi
        else
            log "  failed (exit $?): $(grep -iE "error|abort" "probe-$TAG-$b-$a-$c.log" | tail -1 | cut -c1-120)"
        fi
        rm -f probe-$TAG.pt probe-$TAG.csv
    done
    if [ -n "$best" ]; then
        set -- $best; BATCH=$1; ACCUM=$2; [ "$3" = 1 ] && COMPILE=--compile; GPU_MEM=$4
        log "probe picked batch $BATCH x $ACCUM${COMPILE:+ $COMPILE} at $best_rate pos/s"
    else
        log "every probe failed; keeping batch $BATCH x $ACCUM"
    fi
fi
GPU_MEM=${GPU_MEM:-4096}

STEPS=${STEPS:-$(python -c "print(max(100, $ROWS_PER_PASS * $PASSES // ($BATCH * $ACCUM) // 1000 * 1000))")}
# Fine-tuning (e.g. pytorch/plan-sim-distill.md): LR/WARMUP override the
# trainer's defaults, TRAIN_EXTRA adds trainer flags (--init-ckpt,
# --sim-groups, ...), and DEPLOY_FINAL=1 deploys the last step's weights
# instead of the best-validation checkpoint (a fine-tune's val WDL loss need
# not improve while its ranking does).
LR_ARGS=""
[ -n "${LR:-}" ] && LR_ARGS="$LR_ARGS --lr $LR"
[ -n "${WARMUP:-}" ] && LR_ARGS="$LR_ARGS --warmup $WARMUP"
SNAP_EVERY=10000
[ "${DEPLOY_FINAL:-0}" = 1 ] && SNAP_EVERY=$STEPS
log "training $TAG: $PASSES passes over $LOGS, ~$ROWS_PER_PASS rows/pass, $STEPS steps, batch $BATCH x $ACCUM${COMPILE:+ $COMPILE}"
rm -f best-tf-$TAG*.pt "producer-$TAG.log" "producer-$TAG-passes.log"

# The trainer reads a FIFO; the passes are written into it by a feeder
# whose every producer must succeed. A producer that dies mid-frame leaves
# the stream out of step, so on any failure the feeder kills the trainer
# rather than let the next pass append to a broken stream.
FIFO=$(mktemp -u "stream-$TAG.XXXX.fifo"); mkfifo "$FIFO"
python training.py --arch transformer --ckpt "best-tf-$TAG.pt" --csv "loss_tf_$TAG.csv" \
  --primary wdl --w-wdl 1 --w-value 0 --aux-share 0.15 --spatial-share "$SPATIAL_SHARE" \
  --transpose-prob "$TRANSPOSE" --batch-size "$BATCH" --accum "$ACCUM" $COMPILE --gpu-mem-mib "$GPU_MEM" \
  --epochs 1 --val-cache "$VAL" --val-size "$VAL_SIZE" --shuffle-buffer "$SHUFFLE" \
  --total-steps "$STEPS" --snapshot-every "$SNAP_EVERY" --device "$DEVICE" $LR_ARGS ${TRAIN_EXTRA:-} < "$FIFO" 2>&1 | tee "train-tf-$TAG.log" > /dev/null &
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
DEPLOY_CKPT=best-tf-$TAG.pt
if [ "${DEPLOY_FINAL:-0}" = 1 ]; then
    DEPLOY_CKPT=best-tf-$TAG-step$STEPS.pt
    [ -f "$DEPLOY_CKPT" ] || { log "no final snapshot $DEPLOY_CKPT; not deploying"; exit 1; }
    log "deploying the final weights ($DEPLOY_CKPT)"
fi
MODEL=macondo-nn-tf-$TAG CKPT=$DEPLOY_CKPT CSV=loss_tf_$TAG.csv TRAIN_LOG=train-tf-$TAG.log \
  EXP=tf-$TAG-v-hasty-pairs PRIMARY=wdl VAL_MAX=0.6 ./watch-tf-heads.sh
