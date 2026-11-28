#!/bin/bash
# Times MAGPIE's sim on a file of positions (one CGP per line; the
# opponent's rack is ignored): the 15 best plays by static equity,
# round-robin, two plies, with Macondo's NWL23 word graph, leaves and word
# map, by wall clock around MAGPIE's own commands (bench/magpie_sim.c: the
# sim time MAGPIE prints leaves part of the work out). The Zig equivalent
# is `zig build bench -Doptimize=ReleaseFast -- <positions> <per play>`
# with BENCH_MODE=sim (and BENCH_THREADS, BENCH_RIT).
#
# MAGPIE's win table is in a format whose data isn't in its repository, so
# a synthetic one is written: the numbers it gives differ, the work per
# rollout doesn't. With MAGPIE_RIT=1 it also builds MAGPIE's rack info
# table from Macondo's leaves and word map (about 20 s and 1.9 GB, in the
# work directory) and times the sims with it as well.
#
# Usage: [MAGPIE_RIT=1] [THREADS="1 4 16"] [PER_PLAY=1000] magpie_sim.sh
#        <positions> [MAGPIE checkout] [Macondo data path]
# MAGPIE must have been built with `make magpie BUILD=no_pgo_release`.
set -euo pipefail
positions=$(realpath "$1")
magpie=$(realpath "${2:-$HOME/code/magpie}")
data=$(realpath "${3:-$(dirname "$0")/../../data}")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/data/lexica" "$work/data/strategy"
for d in layouts leaves letterdistributions; do ln -s "$magpie/data/$d" "$work/data/"; done
ln -s "$data/lexica/gaddag/NWL23.kwg" "$data/lexica/gaddag/NWL23.klv2" "$data/lexica/NWL23.wmp" "$work/data/lexica/"
python3 - "$work/data/strategy/winpct_english.csv" <<'PY'
import math, sys
max_bag, max_spread = 93, 300
def row(sample, bag, on, off, sd):
    h = [round(1000 * math.exp(-((b - max_spread - 1) ** 2) / (2 * sd * sd))) for b in range(2 * max_spread + 3)]
    return f"{sample} {bag} {on} {off} 0 " + " ".join(map(str, h))
lines = [f"winpct 2 7 {max_bag} {max_spread}"]
for s in range(2):
    lines += [row(s, 0, on, off, 15) for on in range(1, 8) for off in range(1, 8)]
    lines += [row(s, bag, 7, 7, 20 + bag) for bag in range(1, max_bag + 1)]
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
PY

here=$(cd "$(dirname "$0")" && pwd)
objs=$(ls -d "$magpie"/obj/no_pgo_release-*-b15-r7 | head -1)
gcc -O3 -flto -march=native -DNDEBUG -DBOARD_DIM=15 -DRACK_SIZE=7 \
  -I"$magpie/src" -o "$work/magpie_sim" "$here/magpie_sim.c" \
  $(find "$objs/src" -name '*.o' | grep -v '/main.o$') -lm -lpthread

cd "$work"
rits=false
if [ "${MAGPIE_RIT:-0}" = 1 ]; then
  echo "convert klvwmp2rit NWL23.NWL23 english NWL23 NWL23 -savesettings false" | "$magpie/bin/magpie" > /dev/null 2>&1
  rits="false true"
fi
for rit in $rits; do
  for t in ${THREADS:-1}; do
    "$work/magpie_sim" "$positions" "$work/data" "$rit" "$t" "${PER_PLAY:-200}"
  done
done
