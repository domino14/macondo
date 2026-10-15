#!/usr/bin/env bash
# When the nwl23s match ends: archive its files and final statistics, then
# play a replication match (fresh master seed, bots in the other order,
# 20k pairs) against the same model and archive that too. Runs alongside
# whatever training follows; the GPU has room for Triton next to a
# batch-128 trainer.
cd "$(dirname "$0")/.."
A=$HOME/data/results/nwl23s-spatial-heads
log() { echo "$(date '+%F %T') $*"; }
source pytorch/venv/bin/activate

log "waiting for the nwl23s match"
while pgrep -f "[a]utoplay.*experimentid tf-nwl23s-v-hasty-pairs" >/dev/null; do sleep 60; done
sleep 10
mkdir -p "$A/match"
cp games-tf-nwl23s-v-hasty-pairs.txt tf-nwl23s-v-hasty-pairs.log tf-nwl23s-v-hasty-pairs.config.json "$A/match/"
gzip -f "$A/match/tf-nwl23s-v-hasty-pairs.txt" 2>/dev/null; cp tf-nwl23s-v-hasty-pairs.txt "$A/match/" && gzip -f "$A/match/tf-nwl23s-v-hasty-pairs.txt"
{
  echo "## Main match (final)"
  echo '```'
  python pytorch/pairs-stats.py games-tf-nwl23s-v-hasty-pairs.txt
  echo '```'
} >> "$A/RESULTS.md"
log "main match archived: $(python pytorch/pairs-stats.py games-tf-nwl23s-v-hasty-pairs.txt | tail -1)"

EXP=tf-nwl23s-rep-v-hasty-pairs
log "starting the replication match: HASTY_BOT first, seed 20260927, 20k pairs"
env MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
    MACONDO_TRITON_MODEL_NAME=macondo-nn-tf-nwl23s MACONDO_TRITON_MODEL_VERSION=1 \
    ./bin/shell autoplay -botcode1 HASTY_BOT -botcode2 FAST_ML_BOT -lexicon NWL23 -seed 20260927 \
    -numgames 20000 -gamepairs true -threads 8 -block true -experimentid $EXP > $EXP.log 2>&1 < /dev/null
cp games-$EXP.txt $EXP.log $EXP.config.json "$A/match/"
cp $EXP.txt "$A/match/" && gzip -f "$A/match/$EXP.txt"
{
  echo
  echo "## Replication match (fresh seed 20260927, HastyBot in seat 1, 20k pairs)"
  echo '```'
  python pytorch/pairs-stats.py games-$EXP.txt
  echo '```'
} >> "$A/RESULTS.md"
log "replication archived: $(python pytorch/pairs-stats.py games-$EXP.txt | tail -1)"
