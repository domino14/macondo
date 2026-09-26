#!/usr/bin/env bash
# Second run on the spatial cache: the same recipe plus the per-batch
# transpose (0.5). Waits for the first run-spatial.sh driver to exit and for
# its match to finish (the match shares the GPU with Triton), then trains
# from the cache the first run built. No rescan.
cd "$(dirname "$0")"
log() { echo "$(date '+%F %T') $*"; }
log "waiting for the nwl23s driver"
while pgrep -f "[r]un-spatial.sh" >/dev/null; do sleep 120; done
sleep 180   # the driver exits ~90 s after starting its match
log "waiting for the nwl23s match"
while pgrep -f "[a]utoplay.*experimentid tf-nwl23s-v-hasty-pairs" >/dev/null; do sleep 120; done
log "starting nwl23st (transpose 0.5) from nwl23s-frames.bin"
SCAN=0 TAG=nwl23st CACHE=nwl23s-frames.bin TRANSPOSE=0.5 ./run-spatial.sh
