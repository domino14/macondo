#!/usr/bin/env bash
# Second run on the spatial cache: the same recipe plus the per-batch
# transpose (0.5). Waits for the first run's training (train-fresh.sh under
# run-spatial.sh, TAG nwl23s) to finish and for its match to end (the match
# shares the GPU with Triton), then trains from the cache the first run
# built. No rescan.
cd "$(dirname "$0")"
log() { echo "$(date '+%F %T') $*"; }
log "waiting for the nwl23s training"
while pgrep -f "[t]rain-fresh.sh" >/dev/null; do sleep 120; done
sleep 180   # train-fresh.sh exits ~90 s after starting its match
log "waiting for the nwl23s match"
while pgrep -f "[a]utoplay.*experimentid tf-nwl23s-v-hasty-pairs" >/dev/null; do sleep 120; done
log "starting nwl23st (transpose 0.5) from nwl23s-frames.bin"
SCAN=0 TAG=nwl23st CACHE=nwl23s-frames.bin TRANSPOSE=0.5 ./run-spatial.sh
