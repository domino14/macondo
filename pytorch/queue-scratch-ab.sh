#!/usr/bin/env bash
# From-scratch nets on 3M games only (user's idea, 10/16): what does a net
# trained on nothing but its predecessor's self-play look like? With a
# control on the same number of fresh HastyBot games, so the comparison is
# self-play vs Hasty data, not "3M games vs 54M".
#
#   A  nvnscr:   3M net-vs-net games (~/data/nvn-all.txt.gz, the 12 chunks concatenated)
#   B  hastyscr: 3M HastyBot games (~/data/hasty3m.txt.gz)
#
# Each: run-stream.sh from random init, the streamopen recipe (wdl primary,
# heads, spatial share 0.1) but 20k steps of 2048 (~20 passes; a pass gives
# ~2.0M positions, a fresh turn per game each pass). Validation on each
# arm's own held-out games (5% by game hash), best-validation weights
# deployed (with this little data the last step may be overfit). Then
# 100k pairs vs HastyBot. Waits for any FAST_ML_BOT match (run-stream.sh).
cd "$(dirname "$0")"
DATA=$HOME/data
for arm in "nvnscr $DATA/nvn-all.txt.gz" "hastyscr $DATA/hasty3m.txt.gz"; do
    set -- $arm; tag=$1; shift
    echo "$(date '+%F %T') from scratch: $tag on $*"
    TAG=$tag LOGS="$*" VAL_LOG="$*" PASSES=22 ROWS_PER_PASS=2000000 STEPS=${STEPS:-20000} \
    BATCH=128 ACCUM=16 GPU_MEM=5120 \
      ./run-stream.sh >> "queue-$tag.log" 2>&1 || echo "$(date '+%F %T') $tag failed (see queue-$tag.log)"
done
echo "$(date '+%F %T') done (the last match runs on after this)"
