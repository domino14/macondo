#!/usr/bin/env bash
# Candidate groups from the full sim labels (1.32M positions, deb192 10/13),
# the sim's top 16 candidates per position (all 50 would be ~175 GB). The
# held-out groups (pilot, all candidates) stay as they are.
cd "$(dirname "$0")"
D=$HOME/data/simdistill
log() { echo "$(date '+%F %T') $*"; }
rm -f $D/groups-train.bin $D/groups-top16.bin
for t in open open2; do
  log "frames $t"
  ../bin/simlabel frames -top 16 -turns $HOME/data/$t.txt.gz -tag $t -positions $D/positions-$t.jsonl \
    -labels $D/labels-full.jsonl.gz -out - >> $D/groups-top16.bin 2>> $D/frames-full.log || { log "frames $t failed"; exit 1; }
  log "  $(tail -1 $D/frames-full.log)"
done
log "groups-top16.bin $(du -h $D/groups-top16.bin | cut -f1)"
