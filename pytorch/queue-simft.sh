#!/usr/bin/env bash
# Early test of the sim distillation (pytorch/plan-sim-distill.md) with the
# labels deb192 has so far: copy them, build candidate groups, fine-tune
# streamopen with the ranking loss, deploy the final weights as
# macondo-nn-tf-simft and play 100k pairs vs HastyBot (run-stream.sh ->
# watch-tf-heads.sh). Waits for the base-vs-table sim match (CPU) to end.
#   STEPS (10000), GPM groups per micro-batch (2)
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
D=$HOME/data/simdistill
STEPS=${STEPS:-10000}; GPM=${GPM:-2}
log() { echo "$(date '+%F %T') $*"; }
log "waiting for the base-vs-table sim match"
while pgrep -f "[s]imleafwin1 base -simleafwin2 table" >/dev/null; do sleep 120; done

log "copying the labels from deb192"
timeout 7200 ssh -n -o BatchMode=yes deb192 'gzip -1 -c ~/macondo/simdistill/labels.jsonl' > $D/labels-snap.jsonl.gz.part \
  && mv $D/labels-snap.jsonl.gz.part $D/labels-snap.jsonl.gz || { log "copy failed"; exit 1; }
log "  $(zcat $D/labels-snap.jsonl.gz | wc -l) labelled positions"

log "building candidate groups (training: open, open2; held out: the 558-position pilot)"
rm -f $D/groups-train.bin
for t in open open2; do
  ../bin/simlabel frames -turns $HOME/data/$t.txt.gz -tag $t -positions $D/positions-$t.jsonl \
    -labels $D/labels-snap.jsonl.gz -out - >> $D/groups-train.bin 2>> $D/frames.log || { log "frames $t failed"; exit 1; }
done
../bin/simlabel frames -turns $D/open2-head.txt -tag open2 -positions $D/pilot-positions-val.jsonl \
  -labels $D/pilot-labels-val.jsonl -out $D/groups-val.bin 2>> $D/frames.log || { log "frames val failed"; exit 1; }
tail -3 $D/frames.log | sed "s/^/  /"
log "  groups-train.bin $(du -h $D/groups-train.bin | cut -f1)"

log "fine-tuning streamopen: $STEPS steps, $GPM groups per micro-batch, then deploy and 100k-pair match"
INIT=$HOME/data/results/streamed-stream-streamopen/model/streamopen/best-tf-streamopen.pt
TAG=simft LOGS="$HOME/data/open.txt.gz $HOME/data/open2.txt.gz" VAL=val-streamopen.bin PASSES=1 \
STEPS=$STEPS BATCH=128 ACCUM=16 GPU_MEM=5120 LR=1e-4 WARMUP=500 DEPLOY_FINAL=1 \
TRAIN_EXTRA="--init-ckpt $INIT --sim-groups $D/groups-train.bin --sim-val-groups $D/groups-val.bin --groups-per-micro $GPM" \
  ./run-stream.sh
