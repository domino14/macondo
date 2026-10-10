#!/usr/bin/env bash
# FastMlBot (streamopen) with energy-chosen extra candidates
# (MACONDO_ML_ENERGY_EXTRA=K: the K lowest- and K highest-ΔE plays outside
# the equity top 50 are also sent to the net) vs HastyBot, 100k pairs, NWL23.
# Compare with streamopen at 50 candidates: 57.65 +/- 0.18.
#   K (10), PAIRS (100000), MODEL (macondo-nn-tf-streamopen)
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
K=${K:-10}; PAIRS=${PAIRS:-100000}; MODEL=${MODEL:-macondo-nn-tf-streamopen}
EXP=tf-${MODEL#macondo-nn-tf-}-energy$K-v-hasty-pairs
log() { echo "$(date '+%F %T') $*"; }
./triton-models.sh load $MODEL 1 || { log "could not load $MODEL"; exit 1; }
cd "$REPO"
log "starting paired match: FAST_ML_BOT($MODEL, top 50 + $K low-ΔE + $K high-ΔE) vs HASTY_BOT, NWL23, $PAIRS pairs"
env MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
    MACONDO_TRITON_MODEL_NAME=$MODEL MACONDO_TRITON_MODEL_VERSION=1 MACONDO_ML_ENERGY_EXTRA=$K \
    ./bin/shell autoplay -botcode1 FAST_ML_BOT -botcode2 HASTY_BOT -lexicon NWL23 -numgames $PAIRS \
    -gamepairs true -threads 12 -block true -experimentid $EXP > $EXP.log 2>&1 < /dev/null
log "match done: $(python3 pytorch/pairs-stats.py games-$EXP.txt | tail -1)"
cd pytorch && ./triton-models.sh unload $MODEL
