#!/usr/bin/env bash
# Head to head: the best FastMlBot (streamopen, top 50, decided-game rule)
# vs the fixed N-ply no-endgame sim bot, in game pairs. Complements the
# two bots' separate results against HastyBot (57.65% and 58.62% for 2-ply).
# Waits for any running sim benchmark match so the CPU is not shared.
#   MODEL (macondo-nn-tf-streamopen), PLIES (2), PAIRS (1250), THREADS (14),
#   OUT (~/data/simbench), WAIT (1)
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
MODEL=${MODEL:-macondo-nn-tf-streamopen}; VER=1
PLIES=${PLIES:-2}; PAIRS=${PAIRS:-1250}; THREADS=${THREADS:-14}
OUT=${OUT:-$HOME/data/simbench}
EXP=${EXP:-ml-v-sim$PLIES-pairs}
log() { echo "$(date '+%F %T') $*"; }
if [ "${WAIT:-1}" = 1 ]; then
    log "waiting for the sim benchmark queue"
    while pgrep -f "[q]ueue-simbench.sh" >/dev/null || pgrep -f "[b]in/shell autoplay.*SIMMING_BOT_NO_EG" >/dev/null; do sleep 120; done
fi
./triton-models.sh load $MODEL $VER || { log "could not load $MODEL"; exit 1; }
cd "$REPO"
log "starting paired match: FAST_ML_BOT($MODEL v$VER) vs fixed $PLIES-ply SIMMING_BOT_NO_EG, NWL23, $PAIRS pairs, $THREADS threads"
nice env MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
    MACONDO_TRITON_MODEL_NAME=$MODEL MACONDO_TRITON_MODEL_VERSION=$VER \
    ./bin/shell autoplay -botcode1 FAST_ML_BOT -botcode2 SIMMING_BOT_NO_EG -lexicon NWL23 \
    -fixedsimplies2 $PLIES -simthreads2 1 -numgames $PAIRS -gamepairs true -threads $THREADS \
    -block true -experimentid $EXP -outputdir "$OUT" > "$OUT/$EXP.log" 2>&1 < /dev/null
log "match done: $(python3 pytorch/pairs-stats.py "$OUT/games-$EXP.txt" | tail -1)"
cd pytorch && ./triton-models.sh unload $MODEL
