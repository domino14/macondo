#!/usr/bin/env bash
# Sampled-openings batch: HastyBot vs HastyBot (greedy, quick 2-ply
# endgames) where each game's first K plies are drawn from a softmax over
# static equity, K ~ round(Exp(OPENING_MEAN)). The producer starts a game's
# eligible positions at its last sampled ply, so every outcome label is
# decided by greedy play. Then: scan into a frame cache (spatial targets),
# train with the spatial heads, deploy, paired match. Two phases so the
# generation can overlap someone else's GPU run:
#
#   GEN=1   generate ~/data/<TAG>.txt (default on)
#   TRAIN=1 scan + train + match (default on); set GEN=0 to only do this
#
#   TAG (open), GAMES (27M), THREADS (12), OPENING_MEAN (2), OPENING_TEMP (3),
#   OPENING_TOPN (50), OPENING_UNIFORM (0), TRANSPOSE (0), SPATIAL_SHARE (0.1),
#   EPOCHS (3), BATCH (128), ACCUM (16)
set -o pipefail
cd "$(dirname "$0")"
source venv/bin/activate
log() { echo "$(date '+%F %T') $*"; }

TAG=${TAG:-open}
GAMES=${GAMES:-27000000}
THREADS=${THREADS:-12}
LEXICON=${LEXICON:-NWL23}
OPENING_MEAN=${OPENING_MEAN:-2}
OPENING_TEMP=${OPENING_TEMP:-3}
OPENING_TOPN=${OPENING_TOPN:-50}
OPENING_UNIFORM=${OPENING_UNIFORM:-0}
QUICK_ENDGAME=${QUICK_ENDGAME:-"-quickendgame 2 -quickendgamemargin 60 -quickendgamecap 250"}
GEN=${GEN:-1}
TRAIN=${TRAIN:-1}
TRANSPOSE=${TRANSPOSE:-0}
SPATIAL_SHARE=${SPATIAL_SHARE:-0.1}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-128}
ACCUM=${ACCUM:-16}
DATA=$HOME/data
CACHE=$TAG-frames.bin
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [ "$GEN" = 1 ]; then
    log "generating $GAMES games: openings mean $OPENING_MEAN temp $OPENING_TEMP top $OPENING_TOPN uniform $OPENING_UNIFORM"
    ( cd "$DATA" && rm -f "$TAG.txt" "games-$TAG.txt" && \
      "$HOME/code/macondo/bin/shell" autoplay -botcode1 HASTY_BOT -botcode2 HASTY_BOT \
        -lexicon "$LEXICON" $QUICK_ENDGAME \
        -openingplies "$OPENING_MEAN" -openingtemp "$OPENING_TEMP" -openingtopn "$OPENING_TOPN" -openinguniform "$OPENING_UNIFORM" \
        -numgames "$GAMES" -threads "$THREADS" -block true -experimentid "$TAG" > "$TAG.autoplay.log" 2>&1 )
    log "generated: $(( $(wc -l < "$DATA/games-$TAG.txt") - 1 )) games, turn log $(du -h "$DATA/$TAG.txt" | cut -f1)"
fi
[ "$TRAIN" = 1 ] || exit 0

rm -f "$CACHE" "cache-$TAG.log"
log "scanning $DATA/$TAG.txt"
../bin/mlproducer -labeler result -per-game -endgame-plies 2 < "$DATA/$TAG.txt" 2> "producer-$TAG.log" | \
  python training.py --cache-only --cache "$CACHE" 2> "cache-$TAG.log"
log "cached: $(tail -1 "cache-$TAG.log")"
( cd "$DATA" && nohup gzip "$TAG.txt" > /dev/null 2>&1 & )

TAG=$TAG CACHE=$CACHE EPOCHS=$EPOCHS WAIT_FOR="[n]ever-matches-anything" \
  TRAIN_ARGS="--primary wdl --w-wdl 1 --w-value 0 --aux-share 0.15 --spatial-share $SPATIAL_SHARE --transpose-prob $TRANSPOSE --batch-size $BATCH --accum $ACCUM" \
  ./train-fresh.sh
