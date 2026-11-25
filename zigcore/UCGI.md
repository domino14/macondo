# UCGI: Universal Crossword Game Interface (draft 0)

A line-based protocol between a crossword game engine and an interface,
modeled on chess's UCI. Status: draft, expected to change while the Zig core
is built. Feedback from other engine authors (MAGPIE, wolges) is welcome
before it is frozen.

## Transport

- One command or response per line, UTF-8, ending in `\n`.
- **Native:** the interface starts the engine as a child process and talks
  over stdin/stdout. The engine writes nothing else to stdout; logs go to
  stderr.
- **WASM:** the engine runs in a Web Worker; each line is one `postMessage`
  string in each direction.
- **Socket (optional):** the same lines over TCP or WebSocket.
- Tokens are separated by spaces. A value that contains spaces is the rest of
  the line (as with `position cgp ...`) or is quoted.

## Session

```
> ucgi
< id name MacondoZig 0.1
< id author ...
< option name threads type spin default 1 min 1 max 512
< option name hash-mb type spin default 256 min 1 max 65536
< option name data-path type string default ./data
< ucgiok
> setoption name threads value 8
> isready
< readyok
> quit
```

- `isready` is answered only when all earlier commands, including slow ones
  such as loading a lexicon, have finished.
- The engine answers unknown commands with `error unknown-command <name>` and
  carries on.

## Setting up a position

```
> position cgp <CGP string>
```

CGP (Compact Game Position, see `cgp/`) carries the board, racks, scores,
scoreless turns and options such as the lexicon and letter distribution. The
engine loads lexicon data on first use, or ahead of time with:

```
> load lex CSW24 dist english
```

Data files are found under the `data-path` option, laid out as in Macondo's
`data/` directory: `letterdistributions/<name>`,
`lexica/gaddag/<lexicon>.kwg` and `.klv2` (`super-<lexicon>.klv2` on the
SuperCrosswordGame board), and `strategy/<lexicon>/preendgame.json`. A
lexicon without leave values of its own uses those of the lexicon it is
derived from (CSW19 uses CSW24's, as in Macondo), and failing that plays
by score alone.

## Searching

All `go` commands run in the background. While a search runs the engine
accepts `stop`, `isready` and `quit`; any other command gets
`error busy`. Every search ends with exactly one `bestmove` line, also after
`stop`.

```
> go static                       best play by static equity
> go sim plies 2 iterations 5000  Monte Carlo sim (also: time <ms>, stop-confidence <pct>, bai)
> go infer                        rack inference for the last opponent play
> go endgame plies 6
> go preendgame
> go auto                         choose the algorithm by game state, as the bot does
> stop
```

`go static` answers at once with the best move by static equity (score,
leave value, and the opening, pre-endgame and endgame adjustments):

```
< info move 8D.QUIZ score 84 equity 102.5
< bestmove 8D.QUIZ
```

Progress, as often as the engine likes:

```
< info nodes 182734 nps 4100000 time 44 depth 5 value 23 pv 8H.QI 9G.ZA
< info sim iterations 1200 play 8D.QUIZ win 0.71 equity 41.2 stderr 1.3
< bestmove 8D.QUIZ
```

`info` lines are key/value tokens. A key the interface does not know is
skipped. Long structured payloads (full sim tables, inference rack lists) use
`info json <one-line JSON>`.

## Move notation

Moves use the coordinates and letters of GCG: `8D.QUIZ` (row first means
horizontal, column first means vertical), `.` for a played-through tile,
lowercase for a blank, `exch.ABC` for an exchange and `pass` for a pass.

## Debug and test commands

Used by the cross-check harness; engines may omit them.

```
> gen                    list every legal move for the player on turn, one per line
< move 8D.QUIZ score 84 equity 102.5
< genok count 4211
> crosssets              every square's cross-set data, both directions
< cs 7 8 h set 3fffffffffffffff score 0 anchor 1 lext 7ffffff rext 3fffffffffffffff
< crosssetsok
> leave AEST             leave value of the tiles, from the lexicon's leave file
< leave 18.48235
> autoplay games 100 seed 42
```

`gen` lists the moves best first by static equity (a pass, listed only
when there is no tile play, gets equity -50); an engine without an
equity model may leave `equity` out and list them by score. `crosssets` prints one `cs <row> <col> <h|v>` line per square and direction
(rows and columns from 0): the cross-set and the left and right extension
sets as hex bit masks (bit n = machine letter n), the cross-score, and
whether the square is an anchor. `leave` prints the value exactly as stored
(a 32-bit float), from the leave values the position's lexicon is played
with (0 if it has none). Equities are printed so that they read back as
the same 64-bit float.

## Errors

Errors are one line, `error <token> [detail]`, and the engine carries on.
Tokens so far: `unknown-command`, `unknown-option`, `bad-value`,
`bad-setoption`, `bad-position`, `bad-cgp <reason>`, `bad-load`, `bad-name`,
`bad-tiles`, `bad-go`, `rack-too-big`, `no-position`, `data-not-found <path>`,
`data-read-failed <path>`, `data-invalid <path>`, `line-too-long`.

## NN evaluation callback (optional)

For engines that hand neural-network evaluation to the interface:

```
< evalreq id 7 model <name> n 40 <base64 feature data>
> evalresp id 7 <base64 output data>
```

## To decide

- Exact sim and inference `info` fields.
- Whether multiple named positions or searches can run at once.
- How lexicon data is passed to a WASM engine (bytes through postMessage,
  or URLs the worker fetches).
