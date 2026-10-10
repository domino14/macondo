#!/usr/bin/env bash
# Where does the ML bot stand against Monte Carlo sims? The no-endgame
# simming bot (SIMMING_BOT_NO_EG: 100 candidates, Stop99, static play once
# the bag is empty) at each ply depth vs HastyBot, in game pairs, so the
# paired win rates sit beside FastMlBot's (57.65% for streamopen).
# Pairs need a single-threaded sim (-simthreads1 1); games run in parallel.
# N-ply means N plies for the whole game (-fixedsimplies1): without it the
# bot sims `unseen` plies (9..14) once the bag is down to 2..7 tiles and
# plays statically with one tile in the bag, whatever -minsimplies says.
# CPU only: safe beside a GPU match or training.
#
#   RUNS  "plies:pairs:tag ..." played in order; runs of one depth pool
#         (pairs-stats.py games-sim2*.txt). Default: 2-ply then 3-ply,
#         1,250 pairs each.
#   OUT   ~/data/simbench
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
RUNS=${RUNS:-"2:1250:sim2 3:1250:sim3"}
OUT=${OUT:-$HOME/data/simbench}
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*"; }
cd "$REPO"
# One sim match at a time: wait for any already running.
while pgrep -f "[b]in/shell autoplay.*SIMMING_BOT_NO_EG" >/dev/null; do sleep 60; done
for run in $RUNS; do
    IFS=: read -r p pairs tag <<< "$run"
    # Leave room for a running GPU match's game threads and the desktop.
    THREADS=14
    pgrep -f "[b]in/shell autoplay.*FAST_ML_BOT" >/dev/null && THREADS=10
    EXP=$tag-v-hasty-pairs
    log "starting fixed $p-ply SIMMING_BOT_NO_EG vs HASTY_BOT, NWL23, $pairs pairs, $THREADS game threads ($EXP)"
    nice ./bin/shell autoplay -botcode1 SIMMING_BOT_NO_EG -botcode2 HASTY_BOT -lexicon NWL23 \
        -fixedsimplies1 $p -simthreads1 1 -numgames $pairs -gamepairs true -threads $THREADS \
        -block true -experimentid $EXP -outputdir "$OUT" > "$OUT/$EXP.log" 2>&1 < /dev/null
    log "$EXP done: $(python3 pytorch/pairs-stats.py "$OUT/games-$EXP.txt" | tail -1)"
    log "$p-ply pooled: $(python3 pytorch/pairs-stats.py "$OUT"/games-sim$p*-v-hasty-pairs.txt | tail -1 | sed 's/^.*pairs.txt: //')"
done
