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
scoreless turns and options such as the lexicon and letter distribution,
written `-lex CSW24 -ld english` (the spec's form, MAGPIE's) or in the
deprecated form `lex CSW24; ld english;`; the engine reads both. The
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
parentheses (`-lm 8G.S(CAT)TER 10`), `ex.ABC` or `ex.3`, `pass`, and
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

All `go` commands but `go static`, `go infer` and `go endgame` run in the background. While a search runs the engine
accepts `stop`, `isready` and `quit`; any other command gets
`error busy`. Every search ends with exactly one `bestmove` line, also after
`stop`.

```
> go static                       best play by static equity
> go sim plies 2                  Monte Carlo sim until stop (also: iterations, time <ms>, stop <confidence>)
> go infer                        rack inference for the last opponent play
> go endgame plies 6
> go preendgame                  pre-endgame until done (also: stop, time <ms>)
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

With two tiles or fewer in the bag (once the opponent holds a full rack:
a CGP without the opponent's rack has those tiles in the bag too), a
rollout that empties it stops there
and its win is a quick result-only endgame search's (2 plies, as Macondo's
sim endgame leaves): win, loss or tie for the player on turn, with the
spread at the empty bag as its equity. Only endgames within 60 points are
searched, each once a sim; a search that runs past a million positions
leaves the rollout to play on. `leaves 0` turns this off.

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

With `infer <budget>` the sim first reads the opponent's leave as `go infer`
does with that budget (an `info infer` line gives the read, or says why it
was skipped), then deals their racks as a leave drawn from the posterior by
weight and the rest from the bag, as Macondo's elite bot sims after
inference (bag cycling is then off). A position whose CGP has a last move
counts as turn 1 (the position before it as turn 0) and one without as turn
0; the turn goes into the sim's seeds as the game's turn number goes into
Macondo's.
Each play's line gives its own
rollouts, the best-arm pick comes first, and a line before them says why the
sim stopped (`threshold`, `cutoff`, `sample-limit`, `best-arm-limit`) and how
many rollouts it made:

```
< info sim status threshold rollouts 4213
```

`go infer [budget <n>] [seed <n>]` reads what the opponent kept after the
position's last move (the CGP's `lm`), as Macondo's rangefinder does for its
elite bot: for a candidate leave, a mini-sim of the position before the move
(the opponent's ten best plays by static equity, and the one they made, 2
plies, up to 200 iterations under Macondo's classic Stop99 rule) gives the
likelihood of their play, a softmax with temperature `tau` over the plays'
log-odds of winning. With few enough leaves (at most `budget`) every one is
measured; otherwise 40% of the budget measures leaves drawn from the prior,
a model of how sub-leaves lift the likelihood imputes the rest, and up to six
rounds of measurement drawn from that posterior refine it. An exchange is
read from leave values alone: the prior times exp(0.072 (value - best
value)). `budget` (default 200) is the number of mini-sims, and `seed` seeds
it as `go sim` is seeded, the position taken as turn 1 and the one before the
move as turn 0, so on one thread the read is the one Macondo's rangefinder
makes on one thread in budget mode. `tau` fixes the temperature (default: by
bag size, Macondo's fitted schedule, doubled for a one-tile play with 21 or
more in the bag) and `taucap` caps it (default 0.1, the elite bot's);
`iterations`, `plays`, `plies` and `rounds` shape the mini-sims and the
refinement. It ends with every leave the opponent could have kept and its
weight relative to the heaviest, in Macondo's order, then `inferok`; with
`trace 1`, each mini-sim's leave and likelihood come first, in the order
measured:

```
< info infer draw round 0 leave EIRS likelihood 0.41
< info infer leaves 3469 kept 3 tau 0.023 sims 200 measured 141 enumerated 0
< info infer leave ?AB weight 0.00085
< inferok
```

`go infer` answers before the engine reads the next command. A position
without a last move, or whose last move was a pass, a phony or a bingo, or
with an empty bag, gets `error infer` and the reason.

`go endgame plies <n>` solves the endgame (the bag must be empty, so both
racks are known) to n plies on the `threads` option's threads, deepening iteratively, with a
transposition table of `hash-mb` (at least 2^24 entries). After each depth
it writes the value (the change in the spread of the player on turn), the
positions searched, whether the value is exact, and the line of play, a `|`
where the search's moves end and its estimate of how the game finishes
begins:

```
> go endgame plies 4
< info endgame depth 1 value 14 nodes 63 exact 0 pv 1G.VIG. | 2J..O.DI G7.B...
< info endgame depth 2 value 11 nodes 1806 exact 0 pv 1G.VIG. 4A.HOER | G7.B...
< info endgame depth 4 value 11 nodes 3752 exact 0 pv 1G.VIG. 4A.HOER 13K.B..
< bestmove 1G.VIG.
```

(Depth 3 is skipped: a search to a fixed depth deepens by one ply at first,
then two at a time, ending on the depth asked for; `alldepths` deepens one
ply at a time.)

A value is exact when no line behind it stopped short of the game's end
(at a position valued by the greedy playout, or at a table entry that
depended on one): no deeper search can change it. The search then stops
there, and reports the value at the depth asked for. Two kinds of guess
don't count against it: a playout whose first move goes out ends the game
there, so its value is a sure minimum for the player who moves, and proves
a cutoff it reaches; and at a cutoff, the moves searched before the one
that cut off no longer matter. A player whose only move is a pass, after a
scoring turn, passes without the pass taking a ply (MAGPIE's forced-pass
bypass), which changes the values of searches that stop short of the end
of the game. `nosureout`, `noprecisepure` and `noforcedpass` turn these
off.

Two scoreless turns in a row end the game in the search, as in Macondo's
endgame mode. `firstwin` searches only whether the player on turn wins
(the window -1, 1); `noid`, `noheur` and `nolists` turn off iterative
deepening, the greedy playout that values the positions the search stops
at, and the incremental move lists. `go endgame` answers before the engine
reads the next command.

`go preendgame` solves a pre-endgame (one to six tiles in the bag; the
player on turn holds a full rack, and the opponent's rack in the CGP counts
as unseen) as Macondo's pre-endgame solver does. Each play is valued for
every ordering of the tiles the player on turn can't see: the opponent,
who knows what is in the bag, tries every reply, and one that beats us
makes that ordering a loss (one that ties, a draw); when it is our turn
again with tiles in the bag, we choose a play across the orderings we
can't tell apart, and that ordering's result is the worst among the plays
tied for best. A play's points are its wins plus half its draws, each
ordering counted as often as the unseen tiles can make it. The endgames
are result-only searches to `plies` plies (default 3), deepening
iteratively from one (`noid`: at once). Options:

- `maxtilesleft <n>`: only plays leaving at most n tiles in the bag
  (default 1; -1 for all);
- `nested <n>`: how many levels of our turns with tiles still in the bag
  are solved (default 1; -1 for all); past them we play our best move by
  score plus leave value;
- `notiebreak`: don't tell the plays tied for first that empty the bag
  apart by spread (their total spread over the orderings, from
  full-window endgames; the best 20 by score);
- `cutoff`: stop solving a play once it has more losses than the best play
  that empties the bag can still have; `skiploss`: at its first loss (its
  line then says `cut 1`);
- `only <move>` (repeatable): solve only these plays; `avoid <move>`
  (repeatable): plays the cutoffs never stop;
- `opprack <tiles>`: tiles known to be on the opponent's rack;
- `time <ms>`: stop after this many milliseconds;
- `outcomes`: each play's result for every ordering, by the tiles in the
  bag in the order they are drawn; `plays <n>`: write only the best n;
- `eventuality <tiles>`: solve for this one ordering of the bag (all its
  tiles, in the order they are drawn) and explain the best play's result
  (use with `only` to choose the play): our play and draw; the opponent's
  reply that decides it (the first that beats us, else the first that
  ties) and how many replies were tried to find it; then at each of our
  turns with tiles in the bag, within `nested`, the nested pre-endgame: our
  best play across the orderings we can't tell apart, each ordering's
  verdict (`actual 1` the one the bag holds), and the opponent's deciding
  reply to that play.

It runs in the background on the `threads` option's threads, with a
transposition table of `hash-mb`, and gives the same results on any number
of threads (but for which plays the cutoffs stop). After each endgame depth
but the last it writes the best play so far,
`info preendgame depth <d> endgames <n> time <ms> best <move> points <p>`;
at the end, the depth of its results, the endgames searched, the time and
the number of orderings of the unseen tiles, then the plays and `bestmove`:

```
> go preendgame opprack AACEISU outcomes plays 1
< info preendgame depth 1 endgames 624 time 89 best N3.STYE points 1
< info preendgame depth 2 endgames 1253 time 674 best N3.STEY points 1
< info preendgame plies 3 endgames 1880 time 847 orderings 1
< info preendgame play N3.STEY points 1 spread 10
< info preendgame outcome N3.STEY bag Y win count 1
< bestmove N3.STEY
```

`stop` (or the `time` limit) ends it with the results of the last depth
finished, marked `stopped 1` (if no depth was finished, the results so far,
which are partial), and without an eventuality's explanation:

```
> go preendgame plies 5 time 3000 plays 2
< info preendgame depth 1 endgames 4396 time 707 best N10.SEXTON points 2
< info preendgame plies 1 endgames 4396 time 3000 orderings 8 stopped 1
< info preendgame play N10.SEXTON points 2 spread -128
< info preendgame play 13J.EXONS points 2 spread -136
< bestmove N10.SEXTON
```

An eventuality (Macondo's manual's example, CSW21: 13M P(AH) when we draw
the G and the opponent the R, leaving the L):

```
> go preendgame only 13M.P.. eventuality GRL noid
< info preendgame plies 3 endgames 2469 time 9010 orderings 1
< info preendgame play 13M.P.. points 0
< info preendgame eventuality play 13M.P.. outcome loss rack ?ANNOPY opprack AEEEINS bag GRL
< info preendgame eventuality drawn rack ?AGNNOY bag RL
< info preendgame eventuality reply 10J..A. tried 9 opprack EEEINRS bag L
< info preendgame eventuality nested depth 1 rack ?AGNNOY opprack EEEINRS bag L unseen EEEILNRS best 1H.ANONYm verdict loss
< info preendgame eventuality nested depth 1 bag E count 3 win
< info preendgame eventuality nested depth 1 bag I count 1 win
< info preendgame eventuality nested depth 1 bag L count 1 loss actual 1
< info preendgame eventuality nested depth 1 bag N count 1 win
< info preendgame eventuality nested depth 1 bag R count 1 win
< info preendgame eventuality nested depth 1 bag S count 1 win
< bestmove 13M.P..
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
> play 8D.QUIZ rack AEFNST?
< playok score 32
> takeback rack ADIQSUZ
< takebackok
> cgp
< cgp 15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 ADIQSUZ/AEINRST 0/0 0
```

`setoption name zobrist-file value <path>` loads the keys the endgame
hashes positions with from a file in the format of Macondo's Zobrist.Dump
(JSON), so that a search can be compared with Macondo's node for node;
otherwise the engine makes its own.

`play` plays a tile play for the player on turn as a game does: the tiles
go on the board, the cross-sets are brought up to date incrementally rather
than worked out again, the tiles come off the mover's rack (those it
holds), and the turn passes; nothing is drawn, and `rack` gives the mover's
rack afterwards. `takeback` takes the last move back (the CGP's `lm`, or
the last `play`), as inference does; `rack` gives the rack of the player
then on turn, which the position doesn't know. `cgp` writes the position's
first four CGP fields. The cross-checks use them to reach boards other
than by loading a CGP.

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
