#!/usr/bin/env python3
"""Compares the Zig engine's move generation with MAGPIE's.

For every position of a file (one CGP per line, as TestWritePositions or
TestWriteEndgames writes them), both engines generate every play of the
rack on turn: the Zig engine through UCGI's `gen`, MAGPIE through its
command line's `generate` with no cap on the plays kept or shown. The tile
plays of the two (placement, tiles with played-through squares marked, and
score) must be the same; exchanges and passes are left out, as MAGPIE
lists them its own way.

Usage: magpie_movegen.py <positions> <lexicon> [MAGPIE checkout] [engine]
Run from zigcore/ after `zig build -Doptimize=ReleaseFast`; MAGPIE must be
built (`make magpie BUILD=no_pgo_release`) with the lexicon in its data.
"""

import collections
import os
import re
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
positions, lexicon = sys.argv[1], sys.argv[2]
magpie = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser('~/code/magpie'))
engine = os.path.abspath(sys.argv[4] if len(sys.argv) > 4 else os.path.join(here, '..', 'zig-out', 'bin', 'macondo-zig'))
data = os.path.abspath(os.path.join(here, '..', '..', 'data'))

# The board, racks, scores and scoreless turns; the options are the lexicon's.
cgps = [line.split(' lex ')[0].strip() for line in open(positions) if line.strip()]

# A tile play: where it starts (row first: across; column first: down), its
# tiles with a '.' for each square played through, and its score.
place = r'(\d{1,2}[A-Z]|[A-Z]\d{1,2})'

commands = [f'setoption name data-path value {data}']
for cgp in cgps:
    commands += [f'position cgp {cgp} lex {lexicon};', 'gen']
commands.append('quit')
out = subprocess.run([engine], input='\n'.join(commands) + '\n', capture_output=True, text=True, check=True).stdout
zig_move = re.compile('^' + place + r'\.(.+)$')
zig, plays = [], []
for line in out.splitlines():
    f = line.split()
    if f and f[0] == 'move':
        m = zig_move.match(f[1])
        if m:
            plays.append((m.group(1), m.group(2), int(f[3])))
    elif f and f[0] == 'genok':
        zig.append(plays)
        plays = []

commands = [f'set -lex {lexicon} -numplays 1000000 -maxnumdplays 1000000 -s1 score -s2 score']
for cgp in cgps:
    commands += [f'cgp {cgp}', 'generate']
out = subprocess.run(['./bin/magpie'], cwd=magpie, input='\n'.join(commands) + '\n', capture_output=True, text=True, check=True).stdout
# A row of the move table, beside the board: its number, the play (played-
# through tiles in parentheses), the leave if any, the score and equity.
magpie_move = re.compile(r'\s(\d+):\s+' + place + r' (\S+)\s+(?:\S+\s+)?(\d+)\s+-?\d+\.\d+\s*$')
first_row = re.compile(r'\s1:\s+\S')
mag, plays = [], None
for line in out.splitlines():
    m = magpie_move.search(line)
    if m or first_row.search(line):
        if (int(m.group(1)) if m else 1) == 1:
            if plays is not None:
                mag.append(plays)
            plays = []
        if m:
            tiles = re.sub(r'\(([^)]*)\)', lambda t: '.' * len(t.group(1)), m.group(3))
            plays.append((m.group(2), tiles, int(m.group(4))))
if plays is not None:
    mag.append(plays)

if len(zig) != len(cgps) or len(mag) != len(cgps):
    sys.exit(f'{len(cgps)} positions, but the Zig engine answered {len(zig)} and MAGPIE {len(mag)}')
differ = 0
for i, (cgp, z, m) in enumerate(zip(cgps, zig, mag)):
    zc, mc = collections.Counter(z), collections.Counter(m)
    if zc != mc:
        differ += 1
        if differ <= 5:
            print(f'position {i}: {cgp}')
            print('  only the Zig engine:', sorted((zc - mc).elements())[:8])
            print('  only MAGPIE:', sorted((mc - zc).elements())[:8])
print(f'{lexicon}: {len(cgps)} positions, {sum(map(len, zig))} tile plays, {differ} positions differ')
sys.exit(1 if differ else 0)
