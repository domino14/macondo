#!/usr/bin/env bash
# After queue-scratch-ab.sh (pid 616278) and its last match: head-to-head
# yardsticks for the self-play line. nvnft vs streamopen (did self-play
# fine-tuning beat its parent?), then nvnscr vs hastyscr (from scratch).
cd "$(dirname "$0")"
while kill -0 616278 2>/dev/null; do sleep 60; done
while pgrep -f "[b]in/shell autoplay.*FAST_ML_BOT" >/dev/null; do sleep 60; done
echo "$(date '+%F %T') nvnft vs streamopen"
M1=macondo-nn-tf-nvnft M2=macondo-nn-tf-streamopen ./queue-ml-v-ml.sh
echo "$(date '+%F %T') nvnscr vs hastyscr"
M1=macondo-nn-tf-nvnscr M2=macondo-nn-tf-hastyscr ./queue-ml-v-ml.sh
echo "$(date '+%F %T') done"
