#!/usr/bin/env bash
# MAGPIE's distilled student (jvc56/MAGPIE@569cfe09 models/student-t6-so-rack:
# our transformer at 6 layers instead of 8, distilled from streamopen;
# imported with import_magpie_student.py, parity 4e-7):
#   1. deploy as macondo-nn-tf-student6 (TensorRT fp16, 2 instances)
#   2. speed: 3,000 net-vs-net games (gen-nvn.sh's flags) each with
#      streamopen and the student, same 12 threads, games/hour
#   3. strength: 100k pairs vs HastyBot, then 100k pairs vs streamopen
#   4. rerun nvnscr vs hastyscr (killed 10/16 20:20 when the hastyscr
#      match's unload watcher unloaded the model under it)
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
log() { echo "$(date '+%F %T') $*"; }
CKPT=$HOME/data/student-t6-so-rack/student-t6.pt
M=macondo-nn-tf-student6
BENCH=$HOME/data/student-t6-so-rack/bench

while pgrep -f "[b]in/shell autoplay.*FAST_ML_BOT" >/dev/null || pgrep -f "[t]raining.py" >/dev/null; do sleep 60; done

if [ ! -d "../data/strategy/default/models/$M/1" ]; then
    log "deploying $M"
    cp "$CKPT" student-t6.pt
    ./copy_model.sh "$M" student-t6.pt || { log "deploy failed"; exit 1; }
    rm -f student-t6.pt student-t6-bak.pt
fi

mkdir -p "$BENCH"
for model in macondo-nn-tf-streamopen "$M"; do
    ./triton-models.sh load "$model" 1 || exit 1
    name=speed-${model#macondo-nn-tf-}
    t0=$(date +%s)
    ( cd "$BENCH" && rm -f "$name.txt" "games-$name.txt" && \
      env MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
        MACONDO_TRITON_MODEL_NAME="$model" MACONDO_TRITON_MODEL_VERSION=1 \
        "$REPO/bin/shell" autoplay -botcode1 FAST_ML_BOT -botcode2 FAST_ML_BOT \
        -lexicon NWL23 -quickendgame 2 -quickendgamemargin 60 -quickendgamecap 250 \
        -openingplies 2 -openingtemp 3 -openingtopn 50 -openinguniform 0 \
        -numgames 3000 -threads 12 -block true -experimentid "$name" > "$name.log" 2>&1 )
    t=$(( $(date +%s) - t0 ))
    log "speed $model: 3000 games in ${t}s = $(( 3000 * 3600 / t )) games/h"
    ./triton-models.sh unload "$model"
done

log "$M vs HastyBot, 100k pairs"
./triton-models.sh load "$M" 1 || exit 1
EXP=tf-student6-v-hasty-pairs
( cd "$REPO" && env MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
    MACONDO_TRITON_MODEL_NAME=$M MACONDO_TRITON_MODEL_VERSION=1 \
    ./bin/shell autoplay -botcode1 FAST_ML_BOT -botcode2 HASTY_BOT -lexicon NWL23 -numgames 100000 \
    -gamepairs true -threads 12 -block true -experimentid $EXP > $EXP.log 2>&1 < /dev/null )
log "match done: $(cd "$REPO" && python3 pytorch/pairs-stats.py games-$EXP.txt | tail -1)"
./triton-models.sh unload "$M"

log "$M vs streamopen"
M1=$M M2=macondo-nn-tf-streamopen ./queue-ml-v-ml.sh
log "nvnscr vs hastyscr (rerun)"
rm -f "$REPO/games-tf-nvnscr-v-hastyscr-pairs.txt" "$REPO/tf-nvnscr-v-hastyscr-pairs.txt"
M1=macondo-nn-tf-nvnscr M2=macondo-nn-tf-hastyscr ./queue-ml-v-ml.sh
log "done"
