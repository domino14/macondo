#!/usr/bin/env bash
# One-off (not committed): does oversampling the pre-endgame help the net?
# Two fine-tunes of nwl23l12 on the 27M searched-endgame games (nwl23qe),
# identical but for the producer's -pre-bag-max 6 -pre-share 0.3 (~50% of
# positions decided with 1-6 tiles in the bag, vs ~8% normally):
#   nwl23preft (oversampled) -> 100k pairs vs HastyBot (regression check)
#   nwl23ctlft (control)     -> deployed, no Hasty match
# Then preft vs ctlft: pre-endgame pairs on deb192 (the 4,474 close
# positions, net vs net, both searching endgames) and 100k pairs head to head.
set -eo pipefail
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
log() { echo "$(date '+%F %T') $*"; }
export MODELS_ROOT=$HOME/code/macondo/data/strategy/default/models
export QUICK_ENDGAME="-quickendgame 2 -quickendgamemargin 60 -quickendgamecap 250"
export LEXICON=NWL23

ft() {  # ft TAG PRODUCER_EXTRA NO_MATCH
    TAG=$1 PRODUCER_EXTRA="$2" NO_MATCH=$3 \
    LOGS="$HOME/data/nwl23qe.txt.gz" VAL=$REPO/pytorch/val-nwl23qe.bin \
    PASSES=2 ROWS_PER_PASS=23000000 STEPS=11000 \
    LR=1e-4 WARMUP=500 DEPLOY_FINAL=1 \
    TRAIN_EXTRA="--init-ckpt best-tf-nwl23l12.pt --d-model 256 --layers 12 --heads 8" \
    BATCH=128 ACCUM=16 COMPILE=--compile GPU_MEM=8192 \
      ./run-stream.sh
}

ft nwl23preft "-pre-bag-max 6 -pre-share 0.3" 0
while pgrep -f "[a]utoplay.*experimentid tf-nwl23preft-v-hasty-pairs" >/dev/null; do sleep 60; done
log "match tf-nwl23preft-v-hasty-pairs done: $(python3 pairs-stats.py "$REPO/games-tf-nwl23preft-v-hasty-pairs.txt" | tail -1)"

ft nwl23ctlft "" 1
./triton-models.sh load macondo-nn-tf-nwl23preft
log "both nets loaded; pre-endgame pairs on deb192"
ssh deb192 'cd ~/peglab && MACONDO_DATA_PATH=$HOME/peglab/data nohup setsid ./pegpairs -turns match.txt -n 0 -parallel 150 -pegsecs 30 -egsecs 15 -hybrid 0 \
    -model-a macondo-nn-tf-nwl23preft -model macondo-nn-tf-nwl23ctlft -out e4-preft-v-ctlft.jsonl > /dev/null 2> e4-preft-v-ctlft.err < /dev/null &' || log "deb192 launch failed"
M1=macondo-nn-tf-nwl23preft M2=macondo-nn-tf-nwl23ctlft EXP=tf-nwl23preft-v-nwl23ctlft-pairs ./match-ml-v-ml.sh
log "nwl23pre done"
