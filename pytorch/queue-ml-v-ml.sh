#!/usr/bin/env bash
# Two ML bots head to head, each querying its own Triton model
# (-tritonmodel1/2): by default the two nets handed off to Magpie,
# streamopen (57.65% vs HastyBot) as player 1 and nwl23s (57.17%) as
# player 2. 100k game pairs, NWL23, both with top 50 and the decided-game rule.
#   M1, M2, PAIRS (100000), THREADS (12), EXP
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
M1=${M1:-macondo-nn-tf-streamopen}; M2=${M2:-macondo-nn-tf-nwl23s}
PAIRS=${PAIRS:-100000}; THREADS=${THREADS:-12}
EXP=${EXP:-tf-${M1#macondo-nn-tf-}-v-${M2#macondo-nn-tf-}-pairs}
log() { echo "$(date '+%F %T') $*"; }
./triton-models.sh load $M1 1 && ./triton-models.sh load $M2 1 || { log "could not load the models"; exit 1; }
cd "$REPO"
log "starting paired match: FAST_ML_BOT($M1) vs FAST_ML_BOT($M2), NWL23, $PAIRS pairs, $THREADS threads"
env MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
    MACONDO_TRITON_MODEL_NAME=$M1 MACONDO_TRITON_MODEL_VERSION=1 \
    ./bin/shell autoplay -botcode1 FAST_ML_BOT -botcode2 FAST_ML_BOT -tritonmodel1 $M1 -tritonmodel2 $M2 \
    -lexicon NWL23 -numgames $PAIRS -gamepairs true -threads $THREADS -block true \
    -experimentid $EXP > $EXP.log 2>&1 < /dev/null
log "match done: $(python3 pytorch/pairs-stats.py games-$EXP.txt | tail -1)"
cd pytorch && ./triton-models.sh unload $M1; ./triton-models.sh unload $M2
