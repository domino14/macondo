#!/usr/bin/env bash
# Net-vs-net games (plan-next-ideas.md, batch 3): FAST_ML_BOT vs FAST_ML_BOT
# with the open sets' sampled openings (K ~ Exp(2) plies, softmax T=3 over
# the top 50) and quick 2-ply endgames, the same as the HastyBot open sets
# except for the bot. Generated in chunks so it can stop and resume (a GPU
# swap, a training run): each chunk is ~/data/nvn/<TAG>-NNN.txt.gz, a chunk
# whose .gz exists is skipped, `touch ~/data/nvn/STOP` stops after the
# current chunk.
#
#   TAG (nvn), MODEL (macondo-nn-tf-streamopen), VER (1), CHUNKS (12),
#   CHUNK_GAMES (250000), THREADS (12)
#
# Smoke test 10/11 (3070 Ti, 12 threads): 2,000 games in 222 s, ~32k/hour.
set -o pipefail
cd "$(dirname "$0")"
log() { echo "$(date '+%F %T') $*"; }

TAG=${TAG:-nvn}
MODEL=${MODEL:-macondo-tf}
VER=${VER:-1}
CHUNKS=${CHUNKS:-12}
CHUNK_GAMES=${CHUNK_GAMES:-250000}
THREADS=${THREADS:-12}
OUT=$HOME/data/nvn
mkdir -p "$OUT"
rm -f "$OUT/STOP"

if pgrep -f "[t]raining.py" >/dev/null; then
    log "a training run is on the GPU; not starting"
    exit 1
fi
./triton-models.sh load "$MODEL" "$VER" || exit 1

for i in $(seq 1 "$CHUNKS"); do
    name=$(printf '%s-%03d' "$TAG" "$i")
    if [ -f "$OUT/$name.txt.gz" ]; then continue; fi
    if [ -f "$OUT/STOP" ]; then log "STOP file found; stopping before $name"; break; fi
    log "chunk $name: $CHUNK_GAMES games, $MODEL v$VER, $THREADS threads"
    ( cd "$OUT" && rm -f "$name.txt" "games-$name.txt" && \
      env MACONDO_TRITON_USE_TRITON=true MACONDO_TRITON_URL=localhost:8101 \
        MACONDO_TRITON_MODEL_NAME="$MODEL" MACONDO_TRITON_MODEL_VERSION="$VER" \
        "$HOME/code/macondo/bin/shell" autoplay -botcode1 FAST_ML_BOT -botcode2 FAST_ML_BOT \
        -lexicon NWL23 -quickendgame 2 -quickendgamemargin 60 -quickendgamecap 250 -quickendgamenoprune true \
        -openingplies 2 -openingtemp 3 -openingtopn 50 -openinguniform 0 \
        -numgames "$CHUNK_GAMES" -threads "$THREADS" -block true -experimentid "$name" \
        > "$name.autoplay.log" 2>&1 )
    rc=$?
    games=$(( $(wc -l < "$OUT/games-$name.txt") - 1 ))
    if [ $rc -ne 0 ] || [ "$games" -lt "$CHUNK_GAMES" ]; then
        log "chunk $name failed (exit $rc, $games games); tail of its log:"
        tail -5 "$OUT/$name.autoplay.log"
        break
    fi
    gzip -f "$OUT/$name.txt"
    log "chunk $name done: $games games, $(du -h "$OUT/$name.txt.gz" | cut -f1)"
done
./triton-models.sh unload "$MODEL"
log "exiting"
