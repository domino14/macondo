# inferlab analysis

Scripts that read `inferlab` JSONL. Each takes file paths.

- `analyze.py run.jsonl` — one run: replay vs the original log, per-round
  proposal quality (`-trace`), outcome by whether the truth was measured, and
  the calibration overfit gap.
- `compare.py base.jsonl v1.jsonl ...` — several runs over the same positions:
  outcome per variant and paired gains against the first.
- `bylen.py base.jsonl variant.jsonl` — the paired gain broken down by leave
  length, for judging a global change everywhere it applies. Pass `logged` as
  the base to compare against the run the positions came from: a replay with
  `-seed original` reproduces it exactly, so it never needs replaying.
- `ordering.py run.jsonl` — one `-probe` run: is the imputed posterior in the
  right order around the true leave.
- `ordercmp.py a.jsonl b.jsonl ...` — the ordering diagnostics side by side.
  The row that decides a change is "head above truth, imputed": it has to change
  sign, since measured the head sits about 100x below the truth.
