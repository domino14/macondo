#!/usr/bin/env bash
# Ownership heads (pytorch/plan-next-ideas.md, batch 2 item 1): streamopen's
# recipe from scratch (open + open2, 6 passes, probe-picked batch) plus two
# ownership planes: squares empty in the position that the mover / the
# opponent covers before the game ends. Deploys as macondo-nn-tf-own and
# plays 100k pairs vs HastyBot (baseline streamopen 57.65 +/- 0.18).
# Waits for the simft2 queue (training, deploy, match start) to finish;
# run-stream.sh then waits for its match. Kills the run if the training log
# stops changing for 45 minutes (a crashed trainer can hang).
cd "$(dirname "$0")"
log() { echo "$(date '+%F %T') $*"; }
log "waiting for the simft2 queue"
while pgrep -f "[q]ueue-simft2.sh" >/dev/null; do sleep 120; done
log "starting the ownership run"
export MACONDO_OWNERSHIP=1
TAG=own LOGS="$HOME/data/open.txt.gz $HOME/data/open2.txt.gz" VAL_LOG=$HOME/data/open2.txt.gz \
PRODUCER_EXTRA=-ownership ./run-stream.sh &
RS=$!
# Stall watchdog: only while the trainer is running.
while kill -0 $RS 2>/dev/null; do
    sleep 300
    if pgrep -f "[t]raining.py --arch transformer --ckpt best-tf-own.pt" >/dev/null && [ -f train-tf-own.log ] \
       && [ $(( $(date +%s) - $(stat -c %Y train-tf-own.log) )) -gt 2700 ]; then
        log "STALLED: train-tf-own.log unchanged for 45 min; killing the run"
        grep -v Warning train-tf-own.log | tail -5
        pkill -P $RS; kill $RS
        pkill -f "[t]raining.py --arch transformer --ckpt best-tf-own.pt"
        exit 1
    fi
done
wait $RS
log "run-stream.sh exited ($?)"
