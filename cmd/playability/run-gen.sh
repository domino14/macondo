#!/bin/bash
# Resumable 10M-game HastyBot self-play for playability (estimator A + the
# position reservoir for synth). Re-running skips finished shards; a killed
# run only loses the shards in progress. Run from the main macondo checkout
# (it needs ./data). Usage: run-gen.sh [workers] [shards]
set -euo pipefail
D=${PLAY_DIR:-$HOME/data/playability}
W=${1:-3}
N=${2:-1000}
exec nice -n 10 "$D/playability" gen -out "$D/gen" -shards "$N" \
  -games-per-shard 10000 -workers "$W" -pos-rate 0.002 >>"$D/gen.log" 2>&1
