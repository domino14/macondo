#!/bin/bash
# Times MAGPIE's sim on a file of positions (one CGP per line; the
# opponent's rack is ignored), single-threaded, the 15 best plays by
# static equity, two plies, 200 rollouts per play, with Macondo's NWL23
# word graph, leaves and word map. The Zig equivalent is `zig build bench
# -Doptimize=ReleaseFast -- <positions> 200` with BENCH_MODE=sim.
#
# MAGPIE's win table is in a format whose data isn't in its repository, so
# a synthetic one is written: the numbers it gives differ, the work per
# rollout doesn't. Prints MAGPIE's own timing of the sims and the wall
# time less that of a run without them.
#
# Usage: magpie_sim.sh <positions> [MAGPIE checkout] [Macondo data path]
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

{
  echo "set -path $work/data -lex NWL23 -threads 1 -numplays 15 -plies 2 -sr rr -threshold none -iterations 3000 -minplayiterations 1 -seed 1 -pfrequency 0"
  while read -r p; do
    echo "cgp ${p% lex*} -lex NWL23"
    echo "gen"
    echo "sim"
  done < "$positions"
} > "$work/sim.txt"
grep -v '^sim$' "$work/sim.txt" > "$work/nosim.txt"

cd "$magpie"
wall() { local s e; s=$(date +%s%N); ./bin/magpie < "$1" > "$2" 2>&1; e=$(date +%s%N); echo $(( (e - s) / 1000 )); }
sim_us=$(wall "$work/sim.txt" "$work/sim.out")
nosim_us=$(wall "$work/nosim.txt" /dev/null)
awk -v wall=$((sim_us - nosim_us)) '/Iters:/{it+=$2} /Time:/{t+=$2}
  END{printf "magpie: %d rollouts, own timing %.2f us/rollout, wall %.2f us/rollout\n", it, t*1e6/it, wall/it}' "$work/sim.out"
