#!/usr/bin/env bash
# Does the net gain from a wider candidate list? FastMlBot normally ranks
# HastyBot's top 50 plays; this match has it rank the top 100
# (MACONDO_ML_TOPN=100), otherwise identical to every other match: 100k
# pairs vs HastyBot, NWL23, 12 threads, net only. It runs after the
# streamopen training and its match, on whichever of `stream` and
# `streamopen` scored higher at 50 candidates, so the baseline is that
# model's own match.
#   TOPN (100), MODEL (unset = pick the better of the two)
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
TOPN=${TOPN:-100}
log() { echo "$(date '+%F %T') $*"; }
wait_gone() { while pgrep -f "$1" >/dev/null; do sleep 120; done; }
rate() { python3 pairs-stats.py "$REPO/games-tf-$1-v-hasty-pairs.txt" 2>/dev/null | grep -oE "win rate [0-9.]+" | grep -oE "[0-9.]+$"; }
pairs() { python3 pairs-stats.py "$REPO/games-tf-$1-v-hasty-pairs.txt" 2>/dev/null | grep -oE "pairs=[0-9]+" | cut -d= -f2; }

log "waiting for the streamopen training"
wait_gone "[r]un-stream.sh"
sleep 240
log "waiting for the streamopen match"
wait_gone "[b]in/shell autoplay.*FAST_ML_BOT"
sleep 60

if [ -z "$MODEL" ]; then
    TAG=stream
    so=$(rate streamopen); st=$(rate stream); n=$(pairs streamopen)
    log "at 50 candidates: stream ${st:-?}%, streamopen ${so:-?}% (${n:-0} pairs)"
    if [ -n "$so" ] && [ "${n:-0}" -ge 100000 ] && python3 -c "import sys; sys.exit(0 if $so > $st else 1)"; then
        TAG=streamopen
    fi
    MODEL=macondo-nn-tf-$TAG
else
    TAG=${MODEL#macondo-nn-tf-}
fi
VER=1; EXP=tf-$TAG-top$TOPN-v-hasty-pairs
./triton-models.sh load $MODEL $VER || { log "could not load $MODEL"; exit 1; }
cd "$REPO"
log "starting paired match: FAST_ML_BOT($MODEL v$VER, top $TOPN candidates) vs HASTY_BOT, NWL23, 100k pairs"
env MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
    MACONDO_TRITON_MODEL_NAME=$MODEL MACONDO_TRITON_MODEL_VERSION=$VER MACONDO_ML_TOPN=$TOPN \
    ./bin/shell autoplay -botcode1 FAST_ML_BOT -botcode2 HASTY_BOT -lexicon NWL23 -numgames 100000 \
    -gamepairs true -threads 12 -block true -experimentid $EXP > $EXP.log 2>&1 < /dev/null
log "match done: $(python3 pytorch/pairs-stats.py games-$EXP.txt | tail -1)"
cd pytorch && ./triton-models.sh unload $MODEL
