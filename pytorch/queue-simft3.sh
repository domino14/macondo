#!/usr/bin/env bash
# Sim-distillation fine-tune, try 3: streamopen + the full sim labels (1.26M
# groups, the sim's top 16 candidates each, groups-top16.bin) with a loss on
# the top of the list only. Tries 1-2 (listwise over all 50) lowered the
# ranking loss without sharpening the top pick and hurt play (-0.17, -0.47).
# This one: pairwise: the sim's best against each other candidate, weighted by how clearly the sim separated them (full weight from 2 SE).
# Gradient-balanced (--rank-share 1), tau 1. 6 groups of 16 per micro-batch
# (as many rows as 2 groups of 50): 96 groups/step x 10k steps = ~0.75 pass.
# Deploys as macondo-nn-tf-simft3 and plays 100k pairs vs HastyBot.
# Run after the GPU swap, with gen-nvn stopped (run-stream.sh waits for any
# FAST_ML_BOT autoplay).
cd "$(dirname "$0")"
D=$HOME/data/simdistill
INIT=$HOME/data/results/streamed-stream-streamopen/model/streamopen/best-tf-streamopen.pt
TAG=simft3 LOGS="$HOME/data/open.txt.gz $HOME/data/open2.txt.gz" VAL=val-streamopen.bin PASSES=1 \
STEPS=${STEPS:-10000} BATCH=128 ACCUM=16 GPU_MEM=5120 LR=1e-4 WARMUP=500 DEPLOY_FINAL=1 \
TRAIN_EXTRA="--init-ckpt $INIT --sim-groups $D/groups-top16.bin --sim-val-groups $D/groups-val.bin --groups-per-micro 6 --rank-loss pair --rank-share 1.0 --rank-tau 1.0" \
  ./run-stream.sh
