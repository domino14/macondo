#!/usr/bin/env bash
# Net-vs-net autoplay throughput by Triton serving setup (5070 Ti, 10/15).
# Variants are copies of streamopen v1: -i2 (2 instances), -db (dynamic
# batching, engine batch up to 512), -db2 (both). The baseline is the
# production config (one instance, each request run on its own).
# Prints games/h per variant and thread count; leaves nothing loaded.
cd "$(dirname "$0")"
B=$HOME/data/trt-rebuild/bench; mkdir -p "$B"
GAMES=${GAMES:-3000}
log() { echo "$(date '+%F %T') $*"; }
run() { # model threads
  ./triton-models.sh load "$1" >/dev/null || { log "$1 failed to load"; return; }
  local s=$(date +%s)
  ( cd "$B" && "$HOME/code/macondo/bin/shell" autoplay -botcode1 FAST_ML_BOT -botcode2 FAST_ML_BOT \
      -tritonmodel1 "$1" -tritonmodel2 "$1" -lexicon NWL23 -quickendgame 2 -quickendgamemargin 60 \
      -quickendgamecap 250 -openingplies 2 -openingtemp 3 -openingtopn 50 -openinguniform 0 \
      -numgames "$GAMES" -threads "$2" -block true -experimentid "b-$1-$2" > "b-$1-$2.log" 2>&1 )
  local e=$(( $(date +%s) - s ))
  log "$(printf '%-32s threads %2d: %5d games/h  (%ds)' "$1" "$2" $(( GAMES * 3600 / e )) "$e")"
  ./triton-models.sh unload "$1" >/dev/null
}
log "serving benchmark, $GAMES games per run"
for t in 12 24; do run macondo-nn-tf-streamopen $t; done
for t in 12 24; do run macondo-nn-tf-streamopen-i2 $t; done
for t in 12 24 48; do run macondo-nn-tf-streamopen-db $t; done
for t in 24 48; do run macondo-nn-tf-streamopen-db2 $t; done
rm -f "$B"/*.txt
log "done"
