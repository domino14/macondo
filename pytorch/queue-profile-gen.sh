#!/usr/bin/env bash
# Profile net-vs-net generation (where does a game's CPU time go?) on an
# idle machine: waits for the playability experiment and queue-student.sh
# (pid 850813) to finish and for the load to settle. Then, each with a CPU
# profile, 6,000 games with gen-nvn.sh's flags:
#   spin   12 thr  streamopen  bin/shell-spin   (HEAD as committed)
#   nospin 12 thr  streamopen  bin/shell-nospin (job feeder blocks instead
#                                                of spinning on `default:`)
#   nospin 16 thr  streamopen
#   nospin 12 thr  student6
#   nospin 16 thr  student6
# Output: ~/data/profile-gen/<name>.{prof,log}, games/h in this log.
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
OUT=$HOME/data/profile-gen
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*"; }

while kill -0 850813 2>/dev/null || pgrep -f "[p]layability" >/dev/null \
      || pgrep -f "[b]in/shell autoplay" >/dev/null || pgrep -f "[t]raining.py" >/dev/null; do sleep 60; done
# then 5 quiet minutes in a row (1-min load under 2)
quiet=0
while [ $quiet -lt 5 ]; do
    sleep 60
    l=$(cut -d' ' -f1 /proc/loadavg)
    if awk "BEGIN{exit !($l < 2)}"; then quiet=$((quiet+1)); else quiet=0; fi
done
log "machine idle (load $(cut -d' ' -f1-3 /proc/loadavg)); profiling"

run() {  # name binary model threads
    local name=$1 bin=$2 model=$3 thr=$4
    ./triton-models.sh load "$model" 1 >/dev/null || return 1
    local t0=$(date +%s.%N)
    ( cd "$OUT" && rm -f "$name.txt" "games-$name.txt" && \
      env MACONDO_CPU_PROFILE="$OUT/$name.prof" MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
        MACONDO_TRITON_MODEL_NAME="$model" MACONDO_TRITON_MODEL_VERSION=1 \
        "$REPO/bin/$bin" autoplay -botcode1 FAST_ML_BOT -botcode2 FAST_ML_BOT \
        -lexicon NWL23 -quickendgame 2 -quickendgamemargin 60 -quickendgamecap 250 \
        -openingplies 2 -openingtemp 3 -openingtopn 50 -openinguniform 0 \
        -numgames 6000 -threads "$thr" -block true -experimentid "$name" > "$name.log" 2>&1 )
    local t=$(python3 -c "import time; print(round(time.time() - $t0, 1))")
    log "$name: 6000 games in ${t}s = $(python3 -c "print(int(6000 * 3600 / $t))") games/h (load $(cut -d' ' -f1 /proc/loadavg))"
    rm -f "$OUT/$name.txt"
}
run spin-12 shell-spin macondo-nn-tf-streamopen 12
run nospin-12 shell-nospin macondo-nn-tf-streamopen 12
run nospin-16 shell-nospin macondo-nn-tf-streamopen 16
run nospin-12-student shell-nospin macondo-nn-tf-student6 12
run nospin-16-student shell-nospin macondo-nn-tf-student6 16
./triton-models.sh unload macondo-nn-tf-streamopen >/dev/null
./triton-models.sh unload macondo-nn-tf-student6 >/dev/null
log "done"
