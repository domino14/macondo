#!/usr/bin/env bash
# Queue the sampled-openings batch behind the spatial runs. Generation is CPU
# only, so it starts once the nwl23s match (12 bot threads + Triton) is over
# and runs alongside the nwl23st training; the scan, training and match wait
# for the nwl23st match to end so the GPU is free.
cd "$(dirname "$0")"
log() { echo "$(date '+%F %T') $*"; }
wait_gone() { while pgrep -f "$1" >/dev/null; do sleep 120; done; }

log "waiting for the nwl23s training and match"
wait_gone "[t]rain-fresh.sh"
sleep 180
wait_gone "[a]utoplay.*experimentid tf-nwl23s-v-hasty-pairs"
log "generating the openings batch (alongside nwl23st training)"
TRAIN=0 ./run-openings.sh

log "waiting for the nwl23st training and match"
sleep 600   # let the queued nwl23st driver start its trainer first
wait_gone "[t]rain-fresh.sh"
sleep 180
wait_gone "[a]utoplay.*experimentid tf-nwl23st-v-hasty-pairs"
log "scanning, training and matching the openings batch"
GEN=0 ./run-openings.sh
