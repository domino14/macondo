#!/usr/bin/env bash
# Fresh games only: generate softmax-vs-Hasty games (greedy endgames), scan
# them into a frame cache with the true-result labeler (one position per
# game, empty-bag positions labeled by the 2-ply quick search), train five
# epochs from the cache, deploy, paired match. Unattended.
#
#   GAMES     games to generate (default 27M -> ~20M positions)
#   TAG       names every output (default fresh)
#   WAIT_FOR  pgrep pattern to wait on before starting (default: the
#             tf-result match)
#
# Outputs: ~/data/<TAG>.txt (turn log, gzipped after the scan),
# pytorch/<TAG>-frames.bin, best-tf-<TAG>.pt, loss_tf_<TAG>.csv,
# train-tf-<TAG>.log, watch-tf-<TAG>.log; match tf-<TAG>-v-hasty-pairs.
set -o pipefail
cd "$(dirname "$0")"
source venv/bin/activate
log() { echo "$(date '+%F %T') $*"; }

TAG=${TAG:-fresh}
GAMES=${GAMES:-27000000}
EPOCHS=${EPOCHS:-5}
WAIT_FOR=${WAIT_FOR:-"bin/shell autoplay.*tf-result-v-hasty-pairs"}
THREADS=${THREADS:-16}
LEXICON=${LEXICON:-NWL23}   # explicit: the shell's default-lexicon is NWL18 on this box
# Close endgames (|spread| <= 60) played with a 2-ply quick search, 250 ms
# per move, so game results are right where greedy play gets them wrong.
QUICK_ENDGAME=${QUICK_ENDGAME:-"-quickendgame 2 -quickendgamemargin 60 -quickendgamecap 250"}
VAL_SIZE=${VAL_SIZE:-150000}
WATCH=${WATCH:-1}
CACHE=${CACHE:-$TAG-frames.bin}   # APPEND=1 adds to an existing cache
APPEND=${APPEND:-0}
TRAIN=${TRAIN:-1}                 # 0: stop after the scan (a driver trains later)
DATA=$HOME/data

while pgrep -f "$WAIT_FOR" >/dev/null; do sleep 60; done
log "starting $TAG: generating $GAMES games"

# 1. Generate. autoplay writes <experimentid>.txt (turns) and
#    games-<experimentid>.txt (summaries) into the working directory.
( cd "$DATA" && rm -f "$TAG.txt" "games-$TAG.txt" && \
  "$HOME/code/macondo/bin/shell" autoplay -botcode1 RANDOM_BOT_WITH_TEMPERATURE -botcode2 HASTY_BOT \
    -lexicon "$LEXICON" $QUICK_ENDGAME -numgames "$GAMES" -threads "$THREADS" -block true -experimentid "$TAG" > "$TAG.autoplay.log" 2>&1 )
log "generated: $(( $(wc -l < "$DATA/games-$TAG.txt") - 1 )) games, turn log $(du -h "$DATA/$TAG.txt" | cut -f1)"

# 2. Scan into the cache: one position per game, true result, quick
#    endgame search where the bag was already empty.
[ "$APPEND" = 1 ] || rm -f "$CACHE"
../bin/mlproducer -labeler result -per-game -endgame-plies 2 < "$DATA/$TAG.txt" 2> "producer-$TAG.log" | \
  python training.py --cache-only --cache "$CACHE" 2> "cache-$TAG.log"
log "cached: $(tail -1 "cache-$TAG.log")"
tail -1 "producer-$TAG.log"
( cd "$DATA" && nohup gzip "$TAG.txt" > /dev/null 2>&1 & )
[ "$TRAIN" = 1 ] || exit 0

# 3. Train from the cache. Steps: five epochs of (rows - 150k val) / 2048,
#    rounded down to the thousand so the cosine reaches zero first.
ROWS=$(python -c "import os; print(os.path.getsize('$CACHE') // 2699)")
STEPS=$(python -c "print(max(100, ($ROWS - $VAL_SIZE) * $EPOCHS // 2048 // 1000 * 1000))")
log "training: $ROWS rows, $EPOCHS epochs, $STEPS steps"
rm -f best-tf-$TAG*.pt
python training.py --arch transformer --ckpt "best-tf-$TAG.pt" --csv "loss_tf_$TAG.csv" \
  --aux-share 0.15 --w-wdl 0 --epochs "$EPOCHS" --from-cache "$CACHE" --val-size "$VAL_SIZE" \
  --total-steps "$STEPS" --snapshot-every 10000 2>&1 | tee "train-tf-$TAG.log" > /dev/null
log "training exit=${PIPESTATUS[0]}"

# 4. Deploy and match.
[ "$WATCH" = 1 ] || exit 0
MODEL=macondo-nn-tf-$TAG CKPT=best-tf-$TAG.pt CSV=loss_tf_$TAG.csv TRAIN_LOG=train-tf-$TAG.log \
  EXP=tf-$TAG-v-hasty-pairs VAL_MAX=0.5 ./watch-tf-heads.sh
