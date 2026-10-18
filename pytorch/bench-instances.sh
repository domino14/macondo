#!/usr/bin/env bash
# Net-vs-net generation speed vs Triton instance count (gen-nvn.sh flags,
# 3,000 games, 12 threads), with Triton's per-request timings.
#   bench-instances.sh MODEL COUNT...
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
M=$1; shift
CFG=$REPO/data/strategy/default/models/$M/config.pbtxt
OUT=$HOME/data/profile-gen/instances; mkdir -p "$OUT"
cp "$CFG" "$OUT/$M.config.orig"
met() { curl -s localhost:8102/metrics | grep -E "^nv_inference_(request_success|exec_count|queue_duration_us|compute_input_duration_us|compute_infer_duration_us)\{.*model=\"$M\"" | sed 's/{.*}//' | awk '$2>0'; }
for n in "$@"; do
    sed -i "s/count: [0-9]*/count: $n/" "$CFG"
    ./triton-models.sh unload "$M" >/dev/null; sleep 3
    ./triton-models.sh load "$M" 1 >/dev/null || break
    t0=$(date +%s.%N)
    ( cd "$OUT" && rm -f "i$n.txt" "games-i$n.txt" && env MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
      MACONDO_TRITON_MODEL_NAME="$M" MACONDO_TRITON_MODEL_VERSION=1 \
      "$REPO/bin/${BIN:-shell-nospin}" autoplay -botcode1 FAST_ML_BOT -botcode2 FAST_ML_BOT -lexicon NWL23 \
      -quickendgame 2 -quickendgamemargin 60 -quickendgamecap 250 ${NOPRUNE:+-quickendgamenoprune true} -openingplies 2 -openingtemp 3 -openingtopn 50 \
      -openinguniform 0 -numgames ${GAMES:-3000} -threads ${THREADS:-12} -block true -experimentid "i$n" > "i$n.log" 2>&1 )
    t=$(python3 -c "import time; print(time.time() - $t0)")
    met | awk -v n=$n -v t=$t -v games=${GAMES:-3000} '{d[$1]=$2} END{r=d["nv_inference_request_success"]; printf "instances %d: %.0f games/h  %.1f req/exec  queue %.1f ms  input %.2f ms  infer %.2f ms per request\n", n, games*3600/t, r/d["nv_inference_exec_count"], d["nv_inference_queue_duration_us"]/r/1000, d["nv_inference_compute_input_duration_us"]/r/1000, d["nv_inference_compute_infer_duration_us"]/r/1000}'
    rm -f "$OUT/i$n.txt"
done
cp "$OUT/$M.config.orig" "$CFG"
./triton-models.sh unload "$M" >/dev/null
