#!/usr/bin/env bash
# Where does the ML bot stand against Monte Carlo sims? The no-endgame
# simming bot (SIMMING_BOT_NO_EG: 100 candidates, Stop99, static play once
# the bag is empty) at each ply depth vs HastyBot, in game pairs, so the
# paired win rates sit beside FastMlBot's (57.65% for streamopen).
# Pairs need a single-threaded sim (-simthreads1 1); games run in parallel.
# Note the bot sims `unseen` plies (9..14) once the bag is down to 2..7
# tiles, at every setting; -minsimplies only sets the depth before that.
# CPU only: safe beside a GPU match or training.
#   PLIES ("2 3"), PAIRS (500), OUT (~/data/simbench)
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
PLIES=${PLIES:-"2 3"}
PAIRS=${PAIRS:-500}
OUT=${OUT:-$HOME/data/simbench}
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*"; }
cd "$REPO"
for p in $PLIES; do
    # Leave room for a running GPU match's game threads and the desktop.
    THREADS=14
    pgrep -f "[b]in/shell autoplay.*FAST_ML_BOT" >/dev/null && THREADS=10
    EXP=sim$p-v-hasty-pairs
    log "starting $p-ply SIMMING_BOT_NO_EG vs HASTY_BOT, NWL23, $PAIRS pairs, $THREADS game threads"
    nice ./bin/shell autoplay -botcode1 SIMMING_BOT_NO_EG -botcode2 HASTY_BOT -lexicon NWL23 \
        -minsimplies1 $p -simthreads1 1 -numgames $PAIRS -gamepairs true -threads $THREADS \
        -block true -experimentid $EXP -outputdir "$OUT" > "$OUT/$EXP.log" 2>&1 < /dev/null
    log "$p-ply done: $(python3 pytorch/pairs-stats.py "$OUT/games-$EXP.txt" | tail -1)"
done
