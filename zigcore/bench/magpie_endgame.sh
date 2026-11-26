#!/bin/bash
# Solves a file of endgames (one CGP per line, both racks, bag empty, as
# TestWriteEndgames writes them) to a depth in MAGPIE and in the Zig engine,
# one thread each, and compares the value of every position, then the time.
#
# MAGPIE runs its own benchmark (egspeedbench in its test binary) with its
# forced-pass bypass off, so that a pass uses a ply as it does here. The
# engines then search the same game: the leaf playouts pick plays the same
# way (score, plus twice the opponent's rack for going out), and a solve
# that reaches the end of every line must agree. At a depth short of that,
# a few values can differ: the engines break ties between equal moves in
# their own ways, which the playouts can see.
#
# Usage: magpie_endgame.sh <endgames> <plies> [lexicon] [MAGPIE checkout]
# MAGPIE's test binary must be a release build:
#   make magpie_test BUILD=no_pgo_release
set -euo pipefail
positions=$(realpath "$1")
plies=$2
lexicon=${3:-NWL23}
magpie=$(realpath "${4:-$HOME/code/magpie}")
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# MAGPIE reads the options in the dash form.
sed -E 's/ lex ([A-Za-z0-9]+); ld ([a-z_]+);( bdn [A-Za-z]+;)?$/ -lex \1 -ld \2/' "$positions" > "$work/magpie.txt"
n=$(wc -l < "$positions")

(cd "$magpie" && ./bin/magpie_test "egspeedbench:cgp=$work/magpie.txt,lex=$lexicon,plies=$plies,threads=1,max=$n,bypass=0") > "$work/magpie.out"
(cd "$here/.." && zig build -Doptimize=ReleaseFast > /dev/null &&
  BENCH_MODE=endgame BENCH_ROWS=1 ./zig-out/bin/macondo-zig-bench "$positions" "$plies" ../data) > "$work/zig.out"

python3 - "$work/magpie.out" "$work/zig.out" <<'PY'
import sys
magpie, zig = {}, {}
for line in open(sys.argv[1]):
    f = line.split()
    if f and f[0] == 'BENCHROW' and len(f) >= 6:
        magpie[int(f[1])] = (int(f[2]), float(f[4]))
    elif f and f[0] == 'BENCHSUM':
        print('MAGPIE:', ' '.join(f[2:]))
for line in open(sys.argv[2]):
    f = line.split()
    if len(f) == 9 and f[1] == 'value':
        zig[int(f[0])] = (int(f[2]), float(f[8]))
    elif f and f[0] == 'endgame':
        print('Zig:   ', line.strip())
diff = [i for i in sorted(zig) if i in magpie and magpie[i][0] != zig[i][0]]
print(f'{len(zig)} positions, {len(diff)} with different values')
for i in diff:
    print(f'  {i}: MAGPIE {magpie[i][0]}, Zig {zig[i][0]}')
print(f'time: MAGPIE {sum(t for _, t in magpie.values()):.2f} s, Zig {sum(t for _, t in zig.values()):.2f} s')
PY
