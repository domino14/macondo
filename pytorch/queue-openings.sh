#!/usr/bin/env bash
# After the openings generation (run-openings.sh TRAIN=0, started alongside
# the nwl23st training) and after nwl23st's training and match are over,
# scan, train and match the openings batch. Never overlaps a GPU job.
cd "$(dirname "$0")"
log() { echo "$(date '+%F %T') $*"; }
wait_gone() { while pgrep -f "$1" >/dev/null; do sleep 120; done; }
log "waiting for the openings generation"
wait_gone "[r]un-openings.sh"
log "waiting for the nwl23st training and match"
wait_gone "[t]rain-fresh.sh"
sleep 180
wait_gone "[b]in/shell autoplay"
log "scanning, training and matching the openings batch"
GEN=0 ./run-openings.sh
