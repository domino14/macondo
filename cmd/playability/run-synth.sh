#!/bin/bash
# Estimator B for playability, resumable: finished chunks are skipped on a
# re-run. Waits until gen has written shards 0..POS_SHARDS-1, takes the saved
# turns from exactly those (so a resume sees the same reservoir), then does
# 7-8 letter alphagrams (two strata, K=300) and 9-15 (full word only, K=60).
# Run from the main macondo checkout (it needs ./data).
# Usage: run-synth.sh [workers]
set -euo pipefail
D=${PLAY_DIR:-$HOME/data/playability}
W=${1:-2}
POS_SHARDS=${POS_SHARDS:-60}
B="$D/playability-synth"
last=$(printf "%s/gen/shard-%06d.gob.gz" "$D" $((POS_SHARDS - 1)))
while true; do
  missing=0
  for i in $(seq 0 $((POS_SHARDS - 1))); do
    [ -f "$(printf "%s/gen/shard-%06d.gob.gz" "$D" "$i")" ] || { missing=1; break; }
  done
  [ $missing = 0 ] && break
  sleep 120
done
echo "$(date) positions ready ($last)" >>"$D/synth.log"
nice -n 15 "$B" synth -in "$D/gen" -pos-shards "$POS_SHARDS" -out "$D/synth78" \
  -minlen 7 -maxlen 8 -k 300 -slack 1 -workers "$W" >>"$D/synth.log" 2>&1
nice -n 15 "$B" synth -in "$D/gen" -pos-shards "$POS_SHARDS" -out "$D/synth9p" \
  -minlen 9 -maxlen 15 -k 60 -slack 0 -workers "$W" >>"$D/synth.log" 2>&1
echo "$(date) synth done" >>"$D/synth.log"
