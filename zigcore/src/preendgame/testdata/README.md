# Pre-endgame golden files

The Go engine's pre-endgame results on a fixed set of positions, which the
Zig pre-endgame's tests (`zig build test`) check without Go.

They are written by `TestWritePEGGoldens` in `zigcore/crosscheck`
(`peg_golden_test.go`, which lists the cases):

```
go test ./zigcore/crosscheck/ -run TestWritePEGGoldens -crosscheck.peggoldens $PWD/zigcore/src/preendgame/testdata
```

Rewrite them only when the Go engine's pre-endgame is meant to change its
results, and say why in the commit.

The `manual-` cases are the examples of the pre-endgame manual
(`docs/manual/preendgame.md`). Most cases are small enough for every test
run; the larger ones are checked only with `-Dpeg-full=true` (see
`src/preendgame/peg.zig`), best in ReleaseFast.

## Format

One file a case, line by line:

- `cgp <position>`
- `options plies <n> id <0|1> maxtilesleft <n> nested <n> tiebreak <0|1> cutoff <0|1>`:
  the endgame plies, iterative deepening, `max-tiles-left`,
  `max-nested-depth`, the spread tiebreak and the early cutoff. Every case
  is solved on one thread.
- `only <moves>` (optional): only these moves are solved.
- `opprack <tiles>` (optional): the opponent's rack, known in whole or in
  part.
- `eventuality <tiles>` (optional): only this ordering of the bag is
  solved (its tiles in the order they are drawn), as Go's eventuality mode
  solves it.
- `play <move> points <p> [spread <s>]`: a play's points (wins plus half
  the draws, each bag ordering weighted by its count) and, when the
  tiebreak worked it out, its total spread over the orderings. `play <move>
  cut` is a play the early cutoff stopped, whose results are partial.
- `outcome <tiles> <win|draw|loss> <count>`: the play's result when the
  bag holds these tiles, in the order they are drawn, and how many of the
  orderings of the unseen tiles that is.
- `best <move>`: the first play of the Go engine's result.

Moves are in UCGI notation (`zigcore/UCGI.md`).
