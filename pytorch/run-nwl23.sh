#!/usr/bin/env bash
# The next generation batch: 54M softmax-vs-Hasty games under NWL23, no
# endgame draws, in two halves of 27M so the raw turn log never exceeds
# ~43 GB on disk; both halves scanned into one cache (~40M positions,
# ~108 GB), then three epochs WDL-primary, deploy, paired match.
#
# Waits for the running NWL23 baseline match first. Everything sequential
# in this one script, so no process polling between stages.
cd "$(dirname "$0")"
log() { echo "$(date '+%F %T') $*"; }

while pgrep -f "[b]in/shell autoplay.*tf-heads2-nwl23" >/dev/null; do sleep 60; done
log "baseline match done; freeing old caches"
rm -f result-frames.bin fresh-frames.bin   # both reproducible from ~/data/*.txt.gz

for HALF in a b; do
    log "generating half $HALF"
    TAG=nwl23$HALF GAMES=27000000 LEXICON=NWL23 CACHE=nwl23-frames.bin APPEND=$([ $HALF = a ] && echo 0 || echo 1) \
      TRAIN=0 WATCH=0 WAIT_FOR="[n]ever-matches-anything" ./run-fresh.sh
done
log "both halves cached: $(python3 -c "import os; print(os.path.getsize('nwl23-frames.bin') // 2699)") rows"

TAG=nwl23 CACHE=nwl23-frames.bin EPOCHS=3 WAIT_FOR="[n]ever-matches-anything" ./train-fresh.sh
