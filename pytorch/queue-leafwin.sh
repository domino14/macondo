#!/usr/bin/env bash
# Does board energy at the end of simulated lines help the sim? Fixed 2-ply
# SIMMING_BOT_NO_EG scoring line ends with the fitted model plus energy
# (player 1) vs the same model without energy (player 2), head to head in
# game pairs, NWL23. Only the energy terms differ. Pilot: 5% of decisions
# change (10 of 193). Waits for the sim benchmark queue (5-ply, 6-ply).
#   PAIRS (2500), PLIES (2), THREADS (14), A (energy), B (base)
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
PAIRS=${PAIRS:-2500}; PLIES=${PLIES:-2}; THREADS=${THREADS:-14}; A=${A:-energy}; B=${B:-base}
OUT=$HOME/data/simbench/leafwin; mkdir -p "$OUT"
EXP=sim$PLIES-$A-v-$B-pairs
log() { echo "$(date '+%F %T') $*"; }
log "waiting for the sim benchmark queue"
while pgrep -f "[q]ueue-simbench.sh" >/dev/null || pgrep -f "[b]in/shell autoplay.*SIMMING_BOT_NO_EG" >/dev/null; do sleep 120; done
cd "$REPO"
log "starting fixed $PLIES-ply sim ($A) vs fixed $PLIES-ply sim ($B), NWL23, $PAIRS pairs, $THREADS threads"
nice ./bin/shell autoplay -botcode1 SIMMING_BOT_NO_EG -botcode2 SIMMING_BOT_NO_EG -simleafwin1 $A -simleafwin2 $B \
    -fixedsimplies1 $PLIES -fixedsimplies2 $PLIES -simthreads1 1 -simthreads2 1 -lexicon NWL23 \
    -numgames $PAIRS -gamepairs true -threads $THREADS -block true -experimentid $EXP -outputdir "$OUT" \
    > "$OUT/$EXP.log" 2>&1 < /dev/null
log "match done: $(python3 pytorch/pairs-stats.py "$OUT/games-$EXP.txt" | tail -1)"
