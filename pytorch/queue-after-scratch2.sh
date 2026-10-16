#!/usr/bin/env bash
# After queue-scratch-ab.sh (pid 616280) and its last match: nvnscr vs
# hastyscr head to head. (queue-after-scratch.sh waited on the wrong pid,
# the setsid wrapper, and started nvnft vs streamopen at once.)
cd "$(dirname "$0")"
while kill -0 616280 2>/dev/null; do sleep 60; done
while pgrep -f "[b]in/shell autoplay.*FAST_ML_BOT" >/dev/null; do sleep 60; done
echo "$(date '+%F %T') nvnscr vs hastyscr"
M1=macondo-nn-tf-nvnscr M2=macondo-nn-tf-hastyscr ./queue-ml-v-ml.sh
echo "$(date '+%F %T') done"
