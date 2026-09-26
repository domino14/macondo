#!/usr/bin/env bash
# Spatial-heads run on the existing NWL23 games: rescan the three gzipped
# turn logs (54M games) into a new frame cache carrying the four 15x15
# placement targets, then train with the spatial heads and the per-batch
# transpose augmentation, deploy, and run the paired match. No new games.
#
#   TAG (default nwl23s), LOGS (the turn logs, default nwl23{a,b,c}.txt.gz),
#   EPOCHS (3), SPATIAL_SHARE (0.1), TRANSPOSE (0.5), SCAN (1: rebuild the
#   cache; 0: reuse it), CACHE (default <TAG>-frames.bin)
#
# Outputs: pytorch/<TAG>-frames.bin, producer-<TAG>-<half>.log,
# cache-<TAG>.log, then train-fresh.sh's outputs under TAG.
set -o pipefail
cd "$(dirname "$0")"
source venv/bin/activate
log() { echo "$(date '+%F %T') $*"; }

TAG=${TAG:-nwl23s}
LOGS=${LOGS:-"$HOME/data/nwl23a.txt.gz $HOME/data/nwl23b.txt.gz $HOME/data/nwl23c.txt.gz"}
EPOCHS=${EPOCHS:-3}
SPATIAL_SHARE=${SPATIAL_SHARE:-0.1}
TRANSPOSE=${TRANSPOSE:-0.5}
SCAN=${SCAN:-1}
CACHE=${CACHE:-$TAG-frames.bin}   # SCAN=0 CACHE=... trains a second run off an existing cache

if [ "$SCAN" = 1 ]; then
    rm -f "$CACHE" "cache-$TAG.log"
    for f in $LOGS; do
        half=$(basename "$f" .txt.gz)
        log "scanning $f"
        zcat "$f" | ../bin/mlproducer -labeler result -per-game -endgame-plies 2 2> "producer-$TAG-$half.log" | \
          python training.py --cache-only --cache "$CACHE" 2>> "cache-$TAG.log"
        log "  $(tail -1 "cache-$TAG.log")"
    done
fi
ROWS=$(python -c "import os, training; print(os.path.getsize('$CACHE') // training.CACHE_ROW_BYTES)")
log "cache $CACHE: $ROWS rows"

TAG=$TAG CACHE=$CACHE EPOCHS=$EPOCHS WAIT_FOR="[n]ever-matches-anything" \
  TRAIN_ARGS="--primary wdl --w-wdl 1 --w-value 0 --aux-share 0.15 --spatial-share $SPATIAL_SHARE --transpose-prob $TRANSPOSE" \
  ./train-fresh.sh
