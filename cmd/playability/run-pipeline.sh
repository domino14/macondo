#!/bin/bash
# Lexicon in, playability study list out: ranks.tsv with every alphagram of
# every length ranked 1..N by rack playability (see README.md).
#
#   gen     HastyBot self-play, SHARDS x 10k games
#   score   merge into per-word and per-alphagram tables
#   analyze shrink toward the prior, rank -> $D/final/ranks.tsv
#
# gen is resumable (finished shards are skipped); score and analyze are cheap
# to redo. The synthetic estimator (`playability synth`) is not part of the
# pipeline: at 10M games self-play covers 99.9% of 7-8 letter racks, and for
# 9+ synth misses the hooks that make most of their value.
#
# Run from the main macondo checkout (gen needs ./data).
# Usage: run-pipeline.sh LEXICON [LETTER_DIST]
# Env: PLAY_DIR (default ~/data/playability/LEXICON), SHARDS (1000),
#      GEN_WORKERS (3)
set -euo pipefail
LEX=${1:?usage: run-pipeline.sh LEXICON [LETTER_DIST]}
LD=${2:-english}
D=${PLAY_DIR:-$HOME/data/playability/$LEX}
SHARDS=${SHARDS:-1000}
GW=${GEN_WORKERS:-3}
SRC=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$D"
log() { echo "$(date '+%F %T') $*" | tee -a "$D/pipeline.log"; }

# A fresh binary per run directory, unless one is already there (a running
# gen may be using it).
B="$D/playability-bin"
[ -x "$B" ] || (cd "$SRC" && go build -o "$B" ./cmd/playability)
log "lexicon $LEX ($LD), dir $D, $SHARDS shards, binary $B"

have() { ls "$1"/$2 2>/dev/null | wc -l; }

n=$(have "$D/gen" 'shard-*.gob.gz')
if [ "$n" -lt "$SHARDS" ]; then
  log "gen: $n/$SHARDS shards present"
  nice -n 10 "$B" gen -lexicon "$LEX" -ld "$LD" -out "$D/gen" -shards "$SHARDS" \
    -games-per-shard 10000 -workers "$GW" -pos-rate 0 >>"$D/gen.log" 2>&1
  n=$(have "$D/gen" 'shard-*.gob.gz')
  [ "$n" -ge "$SHARDS" ] || { log "gen stopped at $n/$SHARDS shards"; exit 1; }
fi
log "gen: done"

nice -n 10 "$B" score -lexicon "$LEX" -ld "$LD" -gen "$D/gen" -out "$D/score" >>"$D/score.log" 2>&1
log "score: done"
python3 "$SRC/cmd/playability/analyze.py" "$D/score" "$D/final" | tee "$D/final.log" | head -8
log "done: $D/final/ranks.tsv"
