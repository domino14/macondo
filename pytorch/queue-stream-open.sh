#!/usr/bin/env bash
# The openings scheme at full scale, streamed: generate a second 27M-game
# sampled-openings batch (open2, same parameters as open) now, on the CPU,
# then after the temperature-games streamed run (run-stream.sh, TAG stream)
# and its match are over, stream all 54M openings games for the same six
# passes as TAG stream. The only difference between `stream` and
# `streamopen` is then how the games were generated.
cd "$(dirname "$0")"
log() { echo "$(date '+%F %T') $*"; }
wait_gone() { while pgrep -f "$1" >/dev/null; do sleep 120; done; }
log "generating open2 (27M sampled-openings games, 8 threads)"
TAG=open2 GEN=1 TRAIN=0 THREADS=8 ./run-openings.sh
log "open2 done; waiting for the temperature-games streamed run and its match"
wait_gone "[r]un-stream.sh"
sleep 240
wait_gone "[b]in/shell autoplay.*FAST_ML_BOT"
log "starting the openings streamed run"
TAG=streamopen LOGS="$HOME/data/open.txt.gz $HOME/data/open2.txt.gz" VAL_LOG="$HOME/data/open2.txt.gz" ./run-stream.sh
