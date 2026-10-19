"""Interleave gzipped autoplay turn logs into one gzipped log.

    python mix-logs.py OUT.txt.gz SRC[:MAX_ROWS] [SRC[:MAX_ROWS] ...]

Each SRC is a turn log (.txt.gz) or a space-free glob of several, read in
order; MAX_ROWS caps the turn rows taken from it (the default is all). The
sources are interleaved in blocks of BLOCK rows, round robin, until every
one is spent, so a streamed trainer with a small shuffle buffer sees them
mixed throughout instead of one after the other. The producer assembles
games by gameID, so blocks may split a game; a game cut off by MAX_ROWS
never ends and the producer drops it. One header row is kept.
"""
import glob
import subprocess
import sys

BLOCK = 2000


def rows(spec):
    pattern, _, cap = spec.partition(":")
    cap = int(cap) if cap else None
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no files match {pattern}")
    n = 0
    for f in files:
        p = subprocess.Popen(["zcat", f], stdout=subprocess.PIPE)
        for line in p.stdout:
            if line.startswith(b"playerID,"):
                continue
            if cap is not None and n >= cap:
                break
            n += 1
            yield line
        p.kill()
        p.wait()
        if cap is not None and n >= cap:
            break
    print(f"{spec}: {n} rows from {len(files)} file(s)", file=sys.stderr)


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    header = subprocess.run(f"zcat {sorted(glob.glob(specs[0].partition(':')[0]))[0]} | head -1",
                            shell=True, capture_output=True).stdout
    gz = subprocess.Popen(["gzip", "-1"], stdin=subprocess.PIPE, stdout=open(out, "wb"))
    gz.stdin.write(header)
    live = [rows(s) for s in specs]
    while live:
        for src in list(live):
            for _ in range(BLOCK):
                line = next(src, None)
                if line is None:
                    live.remove(src)
                    break
                gz.stdin.write(line)
    gz.stdin.close()
    if gz.wait() != 0:
        sys.exit("gzip failed")


if __name__ == "__main__":
    main()
