#!/usr/bin/env bash
# After gen-nvn (pid 10360) exits: the serving benchmark, then simft3 (pair
# loss) and simft4 (top-4), each fine-tune + 100k-pair match vs HastyBot.
cd "$(dirname "$0")"
while kill -0 10360 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') gen-nvn done; serving benchmark"
./bench-serving.sh >> bench-serving.log 2>&1
echo "$(date '+%F %T') simft3"
./queue-simft3.sh >> queue-simft3.log 2>&1
echo "$(date '+%F %T') simft4"
./queue-simft4.sh >> queue-simft4.log 2>&1
echo "$(date '+%F %T') done"
