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
# MAGPIE's CLI loads the lexicon named in its settings from ./data when it
# starts, and a later `set -path` does not reload a lexicon of the same
# name, so MAGPIE runs from a work directory whose ./data holds Macondo's
# files, and never saves its settings. With MAGPIE_RIT=1 it also builds
# MAGPIE's rack info table from Macondo's leaves and word map (about 20 s
# and 1.9 GB, in the work directory) and times the sims with it as well.
#
# Usage: [MAGPIE_RIT=1] magpie_sim.sh <positions> [MAGPIE checkout] [Macondo data path]
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

cd "$work"
bin="$magpie/bin/magpie"
rits=false
if [ "${MAGPIE_RIT:-0}" = 1 ]; then
  echo "convert klvwmp2rit NWL23.NWL23 english NWL23 NWL23 -savesettings false" | "$bin" > /dev/null 2>&1
  rits="false true"
fi
wall() { local s e; s=$(date +%s%N); "$bin" < "$1" > "$2" 2>&1; e=$(date +%s%N); echo $(( (e - s) / 1000 )); }
for rit in $rits; do
  {
    echo "set -lex NWL23 -rit $rit -threads 1 -numplays 15 -plies 2 -sr rr -threshold none -iterations 3000 -minplayiterations 1 -seed 1 -pfrequency 0 -savesettings false"
    while read -r p; do
      echo "cgp ${p% lex*}"
      echo "gen"
      echo "sim"
    done < "$positions"
  } > sim.txt
  grep -v '^sim$' sim.txt > nosim.txt
  sim_us=$(wall sim.txt sim.out)
  nosim_us=$(wall nosim.txt /dev/null)
  if [ "$rit" = true ] && grep -q "no rack info table\|playing without the rack info table" sim.out; then
    echo "MAGPIE did not use its rack info table:" >&2
    grep "rack info table" sim.out | sort -u >&2
    exit 1
  fi
  awk -v rit=$rit -v wall=$((sim_us - nosim_us)) '/Iters:/{it+=$2} /Time:/{t+=$2}
    END{printf "MAGPIE (rack info table %s): %d rollouts, own timing %.2f us/rollout, wall %.2f us/rollout\n", rit, it, t*1e6/it, wall/it}' sim.out
done
