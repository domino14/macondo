#!/usr/bin/env bash
# Sim-distillation fine-tune, try 2: the same groups as simft (485k, from
# deb192's first 506k labels) with the ranking loss rebalanced. In try 1 the
# ranking loss pulled ~40x harder on the trunk than the WDL loss (so
# gradient clipping mostly scaled the ranking step) and its target (tau 2)
# was so diffuse that 90% of the loss was the target's own entropy. Here:
# --rank-share 1 (ranking pull = WDL pull, re-measured every 500 steps) and
# --rank-tau 1 (a sharper target). Deploys as macondo-nn-tf-simft2 and plays
# 100k pairs vs HastyBot. run-stream.sh waits for the running match.
cd "$(dirname "$0")"
D=$HOME/data/simdistill
INIT=$HOME/data/results/streamed-stream-streamopen/model/streamopen/best-tf-streamopen.pt
TAG=simft2 LOGS="$HOME/data/open.txt.gz $HOME/data/open2.txt.gz" VAL=val-streamopen.bin PASSES=1 \
STEPS=${STEPS:-10000} BATCH=128 ACCUM=16 GPU_MEM=5120 LR=1e-4 WARMUP=500 DEPLOY_FINAL=1 \
TRAIN_EXTRA="--init-ckpt $INIT --sim-groups $D/groups-train.bin --sim-val-groups $D/groups-val.bin --groups-per-micro 2 --rank-share 1.0 --rank-tau 1.0" \
  ./run-stream.sh
