#!/bin/bash
# Lexicon in, playability study list out: ranks.tsv with every alphagram of
# every length ranked 1..N by rack playability (see README.md).
#
#   gen     HastyBot self-play, SHARDS x 10k games (estimator A + positions)
#   synth   conditioned re-deals for 7-8 letter racks (estimator B), started
#           once the first POS_SHARDS gen shards exist, alongside gen
#   score   merge into per-word and per-alphagram tables
#   analyze combine, shrink toward the prior, rank -> $D/final/ranks.tsv
#
# Every step is resumable: finished shards and chunks are skipped, and score
# and analyze are cheap to redo. 9+ letter racks get no synth (it misses the
# hooks that make most of their value); self-play plus the prior cover them.
#
# Run from the main macondo checkout (gen needs ./data).
# Usage: run-pipeline.sh LEXICON [LETTER_DIST]
# Env: PLAY_DIR (default ~/data/playability/LEXICON), SHARDS (1000),
#      GEN_WORKERS (3), SYNTH_WORKERS (2), POS_SHARDS (60)
set -euo pipefail
LEX=${1:?usage: run-pipeline.sh LEXICON [LETTER_DIST]}
LD=${2:-english}
D=${PLAY_DIR:-$HOME/data/playability/$LEX}
SHARDS=${SHARDS:-1000}
GW=${GEN_WORKERS:-3}
SW=${SYNTH_WORKERS:-2}
POS_SHARDS=${POS_SHARDS:-60}
SRC=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$D"
log() { echo "$(date '+%F %T') $*" | tee -a "$D/pipeline.log"; }

# A fresh binary per run directory, unless one is already there (a running
# gen may be using it).
B="$D/playability-bin"
[ -x "$B" ] || (cd "$SRC" && go build -o "$B" ./cmd/playability)
log "lexicon $LEX ($LD), dir $D, $SHARDS shards, binary $B"

have() { ls "$1"/$2 2>/dev/null | wc -l; }

if [ "$(have "$D/gen" 'shard-*.gob.gz')" -lt "$SHARDS" ]; then
  log "gen: $(have "$D/gen" 'shard-*.gob.gz')/$SHARDS shards present"
  nice -n 10 "$B" gen -lexicon "$LEX" -ld "$LD" -out "$D/gen" -shards "$SHARDS" \
    -games-per-shard 10000 -workers "$GW" -pos-rate 0.002 >>"$D/gen.log" 2>&1 &
  GEN=$!
else
  GEN=
fi

# synth needs a fixed reservoir of positions: exactly shards 0..POS_SHARDS-1.
for i in $(seq 0 $((POS_SHARDS - 1))); do
  f=$(printf "%s/gen/shard-%06d.gob.gz" "$D" "$i")
  until [ -f "$f" ]; do
    [ -n "$GEN" ] && ! kill -0 "$GEN" 2>/dev/null && { log "gen exited early"; exit 1; }
    sleep 60
  done
done
log "synth 7-8: start"
nice -n 15 "$B" synth -lexicon "$LEX" -ld "$LD" -in "$D/gen" -pos-shards "$POS_SHARDS" \
  -out "$D/synth78" -minlen 7 -maxlen 8 -k 300 -slack 1 -workers "$SW" >>"$D/synth.log" 2>&1
log "synth 7-8: done"

if [ -n "$GEN" ]; then wait "$GEN"; fi
n=$(have "$D/gen" 'shard-*.gob.gz')
[ "$n" -ge "$SHARDS" ] || { log "gen stopped at $n/$SHARDS shards"; exit 1; }
log "gen: done"

nice -n 10 "$B" score -lexicon "$LEX" -ld "$LD" -gen "$D/gen" -synth "$D/synth78" \
  -out "$D/score" >>"$D/score.log" 2>&1
log "score: done"
python3 "$SRC/cmd/playability/analyze.py" "$D/score" "$D/final" | tee "$D/final.log" | head -8
log "done: $D/final/ranks.tsv"
