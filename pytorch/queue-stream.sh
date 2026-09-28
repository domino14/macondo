#!/usr/bin/env bash
# The first streamed run (run-stream.sh, TAG stream: 6 passes over the 54M
# NWL23 games, spatial heads, transpose off) after the openings run's
# scan, training and match are all done, so nothing shares the GPU.
cd "$(dirname "$0")"
log() { echo "$(date '+%F %T') $*"; }
wait_gone() { while pgrep -f "$1" >/dev/null; do sleep 120; done; }
log "waiting for the openings run"
wait_gone "[r]un-openings.sh"
wait_gone "[t]rain-fresh.sh"
sleep 240
wait_gone "[b]in/shell autoplay"
log "starting the streamed run"
./run-stream.sh
