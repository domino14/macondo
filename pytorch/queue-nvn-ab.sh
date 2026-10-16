#!/usr/bin/env bash
# Net-vs-net A/B (plan-next-ideas.md, batch 3, steps 2-3): does training on
# the net's own games beat training on the same number of fresh HastyBot
# games? Both arms fine-tune streamopen with identical settings on a 50/50
# mix of new games and old ones (the head of open2, the same rows in both
# arms, so self-play cannot narrow the net):
#
#   A  nvnft:   3M net-vs-net games (~/data/nvn, gen-nvn.sh) + old
#   B  hastyft: 3M fresh HastyBot games (same opening flags) + old
#
# 1. generate the 3M Hasty control games on the CPU (~/data/hasty3m.txt.gz)
# 2. build the two mixed logs (mix-logs.py; old rows = the nvn row count)
# 3. fine-tune A, then B (run-stream.sh: 4 passes of ~5.7M games each, one
#    turn per game per pass, 11k steps of 2048, lr 1e-4); each deploys its
#    final weights and plays 100k pairs vs HastyBot. B waits for A's match.
#
# Re-runnable: finished outputs (the Hasty log, the mixes) are kept.
set -o pipefail
cd "$(dirname "$0")"
log() { echo "$(date '+%F %T') $*"; }
DATA=$HOME/data
INIT=$DATA/results/streamed-stream-streamopen/model/streamopen/best-tf-streamopen.pt
OLD=$DATA/open2.txt.gz

if [ ! -s "$DATA/hasty3m.txt.gz" ]; then
    log "generating 3M HastyBot control games"
    GEN=1 TRAIN=0 TAG=hasty3m GAMES=3000000 ./run-openings.sh || { log "generation failed"; exit 1; }
fi

if [ ! -s "$DATA/mix-nvn.txt.gz" ] || [ ! -s "$DATA/mix-hasty.txt.gz" ]; then
    R=$(zcat "$DATA"/nvn/nvn-*.txt.gz | grep -vc '^playerID,')
    log "net-vs-net games: $R turn rows; mixing each arm with the first $R rows of open2"
    python3 mix-logs.py "$DATA/mix-nvn.txt.gz.tmp" "$DATA/nvn/nvn-*.txt.gz" "$OLD:$R" && \
      mv "$DATA/mix-nvn.txt.gz.tmp" "$DATA/mix-nvn.txt.gz" || { log "mix A failed"; exit 1; }
    python3 mix-logs.py "$DATA/mix-hasty.txt.gz.tmp" "$DATA/hasty3m.txt.gz" "$OLD:$R" && \
      mv "$DATA/mix-hasty.txt.gz.tmp" "$DATA/mix-hasty.txt.gz" || { log "mix B failed"; exit 1; }
    log "mixes: $(du -h "$DATA"/mix-*.txt.gz | tr '\n' ' ')"
fi

for arm in "nvnft mix-nvn" "hastyft mix-hasty"; do
    set -- $arm
    log "fine-tune $1 on $2"
    TAG=$1 LOGS="$DATA/$2.txt.gz" VAL=val-streamopen.bin PASSES=4 ROWS_PER_PASS=5700000 \
    STEPS=${STEPS:-11000} BATCH=128 ACCUM=16 GPU_MEM=5120 LR=1e-4 WARMUP=500 DEPLOY_FINAL=1 \
    TRAIN_EXTRA="--init-ckpt $INIT" \
      ./run-stream.sh >> "queue-$1.log" 2>&1 || log "$1 failed (see queue-$1.log)"
done
log "done (the last match runs on after this)"
