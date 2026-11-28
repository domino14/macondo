#!/bin/bash
# Times MAGPIE's best-move search on a file of positions (one CGP per line,
# from `go test ./zigcore/crosscheck/ -run TestWritePositions
# -crosscheck.positions <file>`), with Macondo's NWL23 word graph, leaves
# and word map, for comparison with `zig build bench`.
#
# Usage: magpie_bench.sh <positions> [MAGPIE checkout] [Macondo data path]
# MAGPIE must have been built with `make magpie BUILD=no_pgo_release`.
set -euo pipefail
positions=$(realpath "$1")
magpie=$(realpath "${2:-$HOME/code/magpie}")
data=$(realpath "${3:-$(dirname "$0")/../../data}")
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

objs=$(ls -d "$magpie"/obj/no_pgo_release-*-b15-r7 | head -1)
gcc -O3 -flto -march=native -DNDEBUG -DBOARD_DIM=15 -DRACK_SIZE=7 \
  -I"$magpie/src" -o "$work/magpie_bench" "$here/magpie_bench.c" \
  $(find "$objs/src" -name '*.o') -lm -lpthread

# MAGPIE's data layout, with Macondo's lexicon files.
mkdir -p "$work/data/lexica"
for d in "$magpie"/data/*; do
  [ "$(basename "$d")" = lexica ] || ln -s "$d" "$work/data/"
done
ln -s "$data/lexica/gaddag/NWL23.kwg" "$data/lexica/gaddag/NWL23.klv2" "$data/lexica/NWL23.wmp" "$work/data/lexica/"

cd "$magpie"
for wmp in false true; do
  "$work/magpie_bench" "$positions" "$work/data" "$wmp"
done
