#!/usr/bin/env bash
# Validation match for the decided-game ranking rule (ai/bot/mlrank.go): the
# same nwl23s model, the rebuilt shell, 100k pairs vs HastyBot, so the
# result is directly comparable with 57.17% +/- 0.18. Waits for the nwl23st
# training and its match to finish so it never shares the GPU with a
# trainer; the openings queue then waits for this match.
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
log() { echo "$(date '+%F %T') $*"; }
wait_gone() { while pgrep -f "$1" >/dev/null; do sleep 120; done; }
log "waiting for the nwl23st training"
wait_gone "[t]rain-fresh.sh"
sleep 240
log "waiting for the nwl23st match"
wait_gone "[b]in/shell autoplay"
MODEL=macondo-nn-tf-nwl23s; VER=1; EXP=tf-nwl23s-decided-v-hasty-pairs
./triton-models.sh load $MODEL $VER || { log "could not load $MODEL"; exit 1; }
cd "$REPO"
log "starting paired match: FAST_ML_BOT($MODEL v$VER, decided-game rule) vs HASTY_BOT, NWL23, 100k pairs"
env MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
    MACONDO_TRITON_MODEL_NAME=$MODEL MACONDO_TRITON_MODEL_VERSION=$VER \
    ./bin/shell autoplay -botcode1 FAST_ML_BOT -botcode2 HASTY_BOT -lexicon NWL23 -numgames 100000 \
    -gamepairs true -threads 12 -block true -experimentid $EXP > $EXP.log 2>&1 < /dev/null
log "match done: $(python3 pytorch/pairs-stats.py games-$EXP.txt | tail -1)"
cd pytorch && ./triton-models.sh unload $MODEL
