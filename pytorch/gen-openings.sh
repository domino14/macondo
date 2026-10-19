#!/usr/bin/env bash
# Sampled-openings games, the training data of the shipped models: HastyBot
# vs HastyBot where each game's first K plies are drawn from a softmax over
# static equity (temperature OPENING_TEMP points, top OPENING_TOPN plays),
# K ~ round(Exp(OPENING_MEAN)), then best play to the end, with quick 2-ply
# endgames. The turn log records K (`openingplies`) and the producer draws
# training positions only after it, so no label depends on a sampled move.
# Writes ~/data/<TAG>.txt.gz (turns) and ~/data/games-<TAG>.txt (games).
# CPU only. See TRAINING.md.
#
#   TAG (open), GAMES (27M), THREADS (12), LEXICON (NWL23), OPENING_MEAN (2),
#   OPENING_TEMP (3), OPENING_TOPN (50), OPENING_UNIFORM (0)
set -o pipefail
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
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
DATA=$HOME/data

log "generating $GAMES $LEXICON games: openings mean $OPENING_MEAN temp $OPENING_TEMP top $OPENING_TOPN uniform $OPENING_UNIFORM"
( cd "$DATA" && rm -f "$TAG.txt" "games-$TAG.txt" && \
  "$REPO/bin/shell" autoplay -botcode1 HASTY_BOT -botcode2 HASTY_BOT \
    -lexicon "$LEXICON" $QUICK_ENDGAME \
    -openingplies "$OPENING_MEAN" -openingtemp "$OPENING_TEMP" -openingtopn "$OPENING_TOPN" -openinguniform "$OPENING_UNIFORM" \
    -numgames "$GAMES" -threads "$THREADS" -block true -experimentid "$TAG" > "$TAG.autoplay.log" 2>&1 ) || {
    log "autoplay failed; see $DATA/$TAG.autoplay.log"; exit 1; }
log "generated: $(( $(wc -l < "$DATA/games-$TAG.txt") - 1 )) games, turn log $(du -h "$DATA/$TAG.txt" | cut -f1)"
log "gzipping the turn log"
( cd "$DATA" && gzip -f "$TAG.txt" ) || exit 1
log "  $(du -h "$DATA/$TAG.txt.gz" | cut -f1)"
