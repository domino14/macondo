#!/usr/bin/env bash
# Re-measure the 53.19% net (macondo-nn-tf-heads2 v1) under NWL23, since
# every earlier match ran under the shell's NWL18 default. Waits for the
# fresh model's match to finish first.
cd "$(dirname "$0")/.."
log() { echo "$(date '+%F %T') $*"; }
while pgrep -f "[b]in/shell autoplay.*tf-fresh-v-hasty-pairs" >/dev/null || pgrep -f "[w]atch-tf-heads.sh" >/dev/null || pgrep -f "[t]rain-fresh.sh" >/dev/null; do sleep 60; done
curl -s -X POST localhost:8100/v2/repository/models/macondo-nn-tf-heads2/load -o /dev/null
EXP=tf-heads2-nwl23-v-hasty-pairs
log "starting baseline re-run: macondo-nn-tf-heads2 v1 vs HASTY_BOT, NWL23, 100k pairs"
setsid nohup env MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
    MACONDO_TRITON_MODEL_NAME=macondo-nn-tf-heads2 MACONDO_TRITON_MODEL_VERSION=1 \
    ./bin/shell autoplay -botcode1 FAST_ML_BOT -botcode2 HASTY_BOT -lexicon NWL23 -numgames 100000 \
    -gamepairs true -threads 12 -block true -experimentid $EXP > $EXP.log 2>&1 < /dev/null &
sleep 90
log "baseline match running: $(( $(wc -l < games-$EXP.txt) - 1 )) games after 90 s"
