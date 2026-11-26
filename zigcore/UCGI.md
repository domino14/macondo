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
< option name word-maps type check default true
< option name rack-tables type combo default auto var auto var true var false
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
by score alone. The best-play search uses the lexicon's word map,
`lexica/<lexicon>.wmp` (MAGPIE's format), on boards it is made for (15x15);
without the file the engine makes it from the word graph in memory, which
takes a second or two the first time. `setoption name word-maps value
false` turns them off.

The CGP's `lm` opcode gives the move that led to the position, by the
player not on turn, in the open-protocols notation: through tiles in
parentheses (`lm 8G.S(CAT)TER 10;`), `ex.ABC` or `ex.3`, `pass`, and
optionally its score. Inference reads the opponent from it. The engine
checks it against the board: its new tiles must be there, the word whole,
the play touching what was on the board before it, and the score (when
given) right; otherwise `position` fails with `BadLastMove`. Through tiles
written as `.` are accepted too. A record that leaves through tiles
unmarked reads as if the move placed them, and is rejected only if that
reading breaks one of those checks.

With word maps, the best-play search can also use a rack info table,
`lexica/<lexicon>.<leaves>.zrit`: what it works out for each full rack before
it looks at the board (leave values, the best exchange, which words the
rack's subracks can make), precomputed for every rack (MAGPIE's rack info
table, with exact leave values). It makes searches faster, not different.
`setoption name rack-tables value auto` (the default) uses one if it is there
and was made from the lexicon's word graph and leave file; `true` also makes
one if there isn't (about 20 seconds on every core and 2.3 GB for English)
and saves it; `false` never uses one.

## Searching

All `go` commands run in the background. While a search runs the engine
accepts `stop`, `isready` and `quit`; any other command gets
`error busy`. Every search ends with exactly one `bestmove` line, also after
`stop`.

```
> go static                       best play by static equity
> go sim plies 2                  Monte Carlo sim until stop (also: iterations, time <ms>, stop <confidence>)
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

`go sim plies <n> plays <n> seed <n> [iterations <n>] [time <ms>]` sims the
`plays` best plays by static equity (default 15), rolling each out `plies`
plies (default 2) with best static plays against a random opponent rack,
once an iteration, until `stop` (or for `iterations` iterations, or `time`
milliseconds). It runs on the `threads` option's workers. The game is seeded
with SHA-256 of `seed` (default 1) and 0, both big-endian u64s, at turn 0,
and the bag starts in machine-letter order, so on one thread the same command
gives the same numbers as Macondo's single-threaded sim of the same position
with that seed (on several, the same numbers in distribution). About once a
second it writes a progress line, and it ends with one line per play, best
first by win probability (then equity), and `bestmove`:

```
< info nodes 182734 nps 410000 time 1000 rollouts 61200 best 8D.QUIZ win 0.71 equity 41.2
```


```
< info sim iterations 1000 play 8D.QUIZ win 0.71 equity 41.2 stderr 1.3
< bestmove 8D.QUIZ
```

With `stop <90|95|98|99|999>` the sim is a best-arm identification run, as
Macondo's elite bot runs its sims: each play gets 128 rollouts, then the
rollouts go mostly to the leader and its closest challenger, and the sim
stops when the leader is told apart from every other play with that
confidence (999 is 99.9%), when it has `bestcap` rollouts (default 10000),
when the sim has spent `cap` rollouts per play (default 1000), or when the
game is decided (or at `stop` or `time`). `iterations` doesn't apply. The
rollouts it makes, and so its result, are the same on any number of threads.
With `cycle 1` (the elite bot's setting) each sample is a bag cycle: one
shuffle of the unseen tiles, rotated a rack at a time so that every unseen
tile starts on the opponent's rack once, and the sample's reward is the mean
of its rollouts.
Each play's line gives its own
rollouts, the best-arm pick comes first, and a line before them says why the
sim stopped (`threshold`, `cutoff`, `sample-limit`, `best-arm-limit`) and how
many rollouts it made:

```
< info sim status threshold rollouts 4213
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
> autoplay games 100 seed 42 [pairs] [moves]
< move 0 8D.QUIZ score 84 equity 102.5
< game 0 score 412 388 turns 24
< autoplayok games 100
> minisim plies 2 plays 10 iterations 200 seed 7
< info sim iterations 192 play 8D.QUIZ win 0.61 equity 12.5 stderr 1.9
< bestmove 8D.QUIZ
```

`minisim` runs the inference's mini-sim on the position: the top plays by
static equity, simmed on one thread for at most `iterations` iterations
under Macondo's classic Stop99 rule (checked every 64 iterations; it drops
plays it rules out and stops early), seeded as `go sim` is.

`autoplay` plays games between static players (best move by static
equity) on the position's board layout and lexicon. Game i is seeded with
SHA-256 of the master seed and i, both as big-endian u64s, which is how
Macondo's autoplay seeds its games, so the same command plays the same
games in both engines; with `pairs`, games 2i and 2i+1 share pair i's seed
and draw from a paired bag (each seat from its own end). `moves` lists each
move before the game's result.

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
