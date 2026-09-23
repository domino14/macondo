#!/usr/bin/env bash
# Wait for the phase-1 training run to finish, then deploy the checkpoint
# under a new Triton model name and start a paired match vs HastyBot.
# Skips the match if the run died early or the value loss looks broken.
#
# Log: pytorch/watch-tf-heads.log. Match: repo root, experiment
# tf-heads-v-hasty-pairs (games-tf-heads-v-hasty-pairs.txt).
set -u
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
source venv/bin/activate
MODEL=${MODEL:-macondo-nn-tf-heads}
CKPT=${CKPT:-best-tf-heads.pt}
CSV=${CSV:-loss_tf_heads.csv}
TRAIN_LOG=${TRAIN_LOG:-train-tf-heads.log}
EXP=${EXP:-tf-heads-v-hasty-pairs}

log() { echo "$(date '+%F %T') $*"; }

log "waiting for training to finish"
while pgrep -f "python training.py --arch transformer --ckpt $CKPT" >/dev/null; do
    sleep 60
done
log "training process gone"

# --- sanity checks ---------------------------------------------------------
if ! grep -q "Total training time" "$TRAIN_LOG"; then
    log "NOT RUNNING MATCH: training did not finish (crashed?)"
    grep -v Warning "$TRAIN_LOG" | tail -5
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    log "NOT RUNNING MATCH: $CKPT missing"
    exit 1
fi
PRIMARY=${PRIMARY:-value}
read -r STEPS BEST < <(python - "$CSV" "val_$PRIMARY" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(rows[-1]["step"], min(float(r[sys.argv[2]]) for r in rows))
PY
)
log "training done: $STEPS steps, best val_$PRIMARY $BEST"
# The single-head run was at 0.0937 by step 2000 and 0.0913 at the end, so
# anything above 0.095 means something is wrong with a bogowin-labeled run.
# Other targets sit on other scales: +-1 results are ~0.28, smoothed rollout
# labels ~0.005. VAL_MAX overrides the ceiling.
VAL_MAX=${VAL_MAX:-0.095}
if python -c "import sys; sys.exit(0 if float('$BEST') <= float('$VAL_MAX') else 1)"; then
    :
else
    log "NOT RUNNING MATCH: best val_$PRIMARY $BEST > $VAL_MAX, learning looks broken"
    exit 1
fi

# --- deploy ----------------------------------------------------------------
log "exporting $CKPT as $MODEL"
if ! ./copy_model.sh "$MODEL" "$CKPT"; then
    log "NOT RUNNING MATCH: copy_model.sh failed"
    exit 1
fi
VER_DIR=$(ls -d "$REPO/data/strategy/default/models/$MODEL"/[0-9]* | sort -V | tail -1)
VER=$(basename "$VER_DIR")
log "installed version $VER at $VER_DIR"

log "parity check torch vs onnx vs engine"
if ! python export-tester.py --ckpt "${CKPT%.pt}-bak.pt" --onnx "$VER_DIR/model.onnx" \
        --engine "$VER_DIR/model.plan" --frames calibrate.bin 2>&1 | grep -v -i warn | tail -4; then
    log "NOT RUNNING MATCH: parity check failed"
    exit 1
fi

log "loading $MODEL into Triton (explicit model control)"
curl -s -X POST "localhost:8100/v2/repository/models/$MODEL/load" -o /dev/null -w "load http %{http_code}\n"
for i in $(seq 1 30); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8100/v2/models/$MODEL/versions/$VER/ready)" = 200 ]; then
        break
    fi
    sleep 5
done
if [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8100/v2/models/$MODEL/versions/$VER/ready)" != 200 ]; then
    log "NOT RUNNING MATCH: Triton did not report $MODEL v$VER ready"
    docker logs macondo-triton 2>&1 | tail -20
    exit 1
fi
log "$MODEL v$VER ready"

# --- match -----------------------------------------------------------------
cd "$REPO"
log "starting paired match: FAST_ML_BOT($MODEL v$VER) vs HASTY_BOT, 100k pairs, 12 threads"
setsid nohup env MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
    MACONDO_TRITON_MODEL_NAME=$MODEL MACONDO_TRITON_MODEL_VERSION=$VER \
    ./bin/shell autoplay -botcode1 FAST_ML_BOT -botcode2 HASTY_BOT -numgames 100000 \
    -gamepairs true -threads 12 -block true -experimentid $EXP \
    > $EXP.log 2>&1 < /dev/null &
sleep 90
if pgrep -f "bin/shell autoplay.*experimentid $EXP" >/dev/null; then
    log "match running: $(( $(wc -l < games-$EXP.txt) - 1 )) games after 90 s; log $EXP.log"
else
    log "match process died; tail of $EXP.log:"
    tail -20 $EXP.log
    exit 1
fi
