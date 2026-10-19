#!/usr/bin/env python3
"""Look up racks by playability rank in a ranks.tsv from analyze.py.

usage: query.py RANKS_TSV LENGTH FROM [TO]
e.g.   query.py final/ranks.tsv 7 201 400
"""
import sys

path, L = sys.argv[1], sys.argv[2]
lo = int(sys.argv[3])
hi = int(sys.argv[4]) if len(sys.argv) > 4 else lo
with open(path) as f:
    print(f.readline(), end="")
    for line in f:
        c = line.split("\t", 3)
        if c[0] == L and lo <= int(c[1]) <= hi:
            print(line, end="")
