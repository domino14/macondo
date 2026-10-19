# Puzzles

- [Back to Manual](/macondo/manual)
- [Back to Main Page](/macondo)

Two commands. `puzzlegen` finds puzzles in games or in fresh bot self-play and
writes them to a file; `puzzle` steps through that file in the shell.

## Quick start

Generate a file of puzzles:

```
puzzlegen selfplay -bot COMMON_WORD_PLUS_TWOS_BOT -numgames 200 -filter "tag=NON_BINGO and tag=CEL_PLUS_TWOS and score>=30 and score_advantage>=10" -max 20 -out puzzles.jsonl
```

Then step through them:

```
puzzle open puzzles.jsonl
```

That filter asks for non-bingo plays worth 30 or more that beat the next-best
score by 10, with an answer built from common words. It is a reasonable default
to vary from.

Commands go on one line. The shell has no line continuation, so a trailing
backslash is an error rather than a wrap.

## Two vocabularies

The words on the board and the words in the answer are controlled separately.

- **The board** is set by `-bot`, and only for the `selfplay` source.
- **The answer** is set by a tag in `-filter`. Left unconstrained, it can be any
  word in the game lexicon.

`COMMON_WORD_PLUS_TWOS_BOT` restricts both players to the common-word list
(ECWL) plus every two-letter word in the game's own lexicon. ECWL has 66
two-letter words; NWL23 has 107. The bot admits all 107, but none of the 450
three-letter words NWL23 adds over ECWL. Measured over five games:

| Bot | Board words | Twos ECWL lacks | Outside ECWL+2s |
|---|---|---|---|
| `COMMON_WORD_PLUS_TWOS_BOT` | 212 | 36 | 0 |
| `LEVEL4_COMMON_WORD_BOT` | 184 | 0 | 0 |
| `HASTY_BOT` (default) | 201 | 21 | 74 |

The plain common-word bot is clean but sparse; HastyBot is dense but fills the
board with words like DOWIE, YIRTH and PAGOD. About 17% of the twos bot's board
words are twos a pure common-word bot could not have played.

No new lexicon file is involved. ECWL is a subset of NWL/CSW, so every word the
hybrid permits is already in the game lexicon; the bot generates with the full
lexicon and rejects any play forming a word outside the hybrid, the same way the
existing common-word bots work.

## puzzlegen

Examines every position in a game, finds the best play, and keeps the positions
matching the filter.

### Sources

`puzzlegen selfplay` — fresh bot-vs-bot games. The only source that takes `-bot`.

`puzzlegen woogles <id>...` — games from Woogles, by ID.

`puzzlegen xt <id>...` — games from Cross-tables, by ID.

`puzzlegen gcg <path>...` — local `.gcg` files.

### Options

`-bot CODE` — *selfplay only.* Both players. Default `HASTY_BOT`; use
`COMMON_WORD_PLUS_TWOS_BOT` for readable boards.

`-numgames N` — *selfplay only.* Default 1.

`-max N` — stop after N matches, part-way through a game if necessary. Use this
instead of guessing at `-numgames`.

`-filter EXPR` — which positions to keep. Omit to keep all of them.

`-out FILE` — append matches as JSON Lines. Read by `puzzle open`.

`-show SECTIONS` — what to print per match: `line` (default), `board`, `cgp`,
`answer`, `stats`, `all`.

`-seed S` — *selfplay only.* Any string. The same seed replays the same games,
and the seed is recorded in every puzzle.

`-lexicon NAME` — default `NWL23`.

`-equity-margin F` / `-score-margin F` — thresholds for `EQUITY` and `POINTS`.
Both default to 10.

`-eqloss-limit N` — abandon a game after this much cumulative equity loss.
Default 1000, and automatically unlimited for common-word bots, which lose
equity to a full-lexicon evaluator by design. Setting it by hand for a
common-word run will throw away every game.

`-gcgdir DIR` — also save each source game in full, including the moves after
the puzzle position. Contains the answer; it is for studying a game, not for
solving. The spoiler-free copy is the `gcg` field in `-out`.

## Filters

A boolean expression over tags and stat fields.

```
tag = NAME          has this tag
tag != NAME         does not
FIELD >= VALUE      also <=  >  <  =  !=

joined with and / or / not, grouped with parentheses
```

Case-insensitive throughout. A tag is always written `tag=BINGO`, never bare.
Stat fields are named directly: `score>=30`.

The bot governs the board; only a tag governs the answer. A filter on score
alone will return answers like GRILSE and ROQUE on otherwise readable boards.

## Every tag

Ten tags, each usable as `tag=NAME` or `tag!=NAME`. A position can carry
several.

### Kind of question

| Tag | Fires when | Typical use |
|---|---|---|
| `EQUITY` | The best play beats the second-best by the equity margin. | One unarguable answer. |
| `POINTS` | The answer is also the top scorer, by the score margin. | "Find the biggest play." Implied by `score_advantage>=10`. |

### Shape of the answer

| Tag | Fires when | Typical use |
|---|---|---|
| `BINGO` | All seven tiles are used. | Bingo practice. |
| `ONLY_BINGO` | A bingo, and no other bingo exists from that rack. | Unique placement and word. |
| `BLANK_BINGO` | A bingo using a blank. | Seeing through a blank. |
| `BINGO_NINE_OR_ABOVE` | A bingo nine letters or longer, so it plays through board tiles. | Through-play bingos. |
| `NON_BINGO` | Not a bingo. | Scoring plays, hooks, overlaps, blocks. |
| `POWER_TILE` | The answer plays a tile worth more than 6 — J, Q, X, Z. | Big-tile placement. |

### Vocabulary of the answer

| Tag | Fires when | Typical use |
|---|---|---|
| `CEL_ONLY` | Every word the answer forms is in ECWL. | No two-letter knowledge required. |
| `CEL_PLUS_TWOS` | Every word is in ECWL, or is a two-letter word in the game lexicon. | The usual choice. |

The two nest: `CEL_ONLY` always implies `CEL_PLUS_TWOS`. Each is a ceiling on
the vocabulary, not a statement that a two was used — `15A PRIZES` earns both
and uses no unusual two.

Answers in the gap between them are parallel plays whose crossword is a two
ECWL lacks:

| Answer | Words formed | Not in ECWL |
|---|---|---|
| `7C VEX` | VEX, XI, EM | `EM` |
| `M3 JUNIOR` | JUNIOR, ER, PO | `PO` |
| `M2 NoMINEE` | NOMINEE, RE, OE | `OE` |
| `E11 AT` | AT, ZA, UT | `ZA`, `UT` |

To collect only those: `-filter "tag=CEL_PLUS_TWOS and tag!=CEL_ONLY"`

## Recipes

Add `and tag=CEL_PLUS_TWOS` to any of these to constrain the answer as well.

```
Scoring play, clearly better than the alternatives
tag=NON_BINGO and score>=30 and score_advantage>=10

Overlap — a word laid alongside another, making 2-letter crosswords
words_formed>=3 and max_cross_word_length=2 and score>=30

Hook — hanging a play off an existing word
words_formed=2 and longest_hooked_word_length>=3 and score>=30

Extension — growing a word already on the board
longest_extended_word_length>=3 and (dws_covered>=1 or tws_covered>=1)

Power tile on a letter bonus, making a crossword
tag=NON_BINGO and max_fresh_tile_face_value_on_letter_bonus_with_crossword>=8

Bingos with no alternative
tag=ONLY_BINGO and words_formed>=2

One-point rack, still find 25
total_rack_tile_score<=7 and score>=25 and score_advantage>=10
```

A demanding filter over 200 games can return nothing. Loosen the numbers before
adding games.

## puzzle

Loads each puzzle as the shell's current game, so `s`, `gen` and `sim` work on
it. Replaces whatever game was loaded, as `load` does.

| Command | |
|---|---|
| `puzzle open <file>` | Open a file, show puzzle 1. |
| `puzzle answer` | Reveal the answer. |
| `puzzle info` | Everything in the record except the answer: all tags, game, turn, seed, CGP, all 22 stats. |
| `puzzle list` | Index the file, marking the current puzzle and the kept ones. |

```
> puzzle open puzzles.jsonl
Opened puzzles.jsonl: 4 puzzles.
Puzzle 1 of 4   EQUITY POINTS   (+3 tag(s) in `puzzle info`)
  NWL23  ·  english  ·  seed:p_pkdfz…  turn 15  ·  seed demo2
```

Then the board:

```
   A B C D E F G H I J K L M N O     ->              player1  CEIPRSZ  154
   ------------------------------                    player2           104
 1|=     '       =       '     = |
 2|  -       "       "       -   |   Bag + unseen: (52)
 3|    -       '   '       -     |
 4|'     -       '       B     ' |   ? ? A A A A B D D E E E E E E F G G G H
 5|      E X           - L       |   I I I I J L L M N N N N O O O O O O P Q
 6|  "     I F       "   U   "   |   R R R S S T T T T U U V
 7|    '     A H   Y O U R '     |
 8|M A L I C   A W A I T '     = |
 9|    '   A I D E '       '     |
10|  "     W " R     "       "   |   Turn 0:
11|        S   O K     -         |
12|'     -     N E       -     ' |
13|    -       ' Y '       -     |
14|  -       E V E N T       -   |
15|=     '       D       '     = |
   ------------------------------
```

Every word here is one an ordinary player knows — MALIC, AWAIT, KEYED — because
the bot was not allowed to play anything else. The answer is `15A PRIZES` for
83: 27 across the triple with the Z on a double letter, plus `ES` below
`EVENT`.

`gen` lists the top plays from the position, which checks an answer against the
engine rather than against the stored one and shows what second-best gives up.

### Moving between puzzles

| Command | |
|---|---|
| `puzzle next` · `puzzle n` | Next puzzle. |
| `puzzle prev` · `puzzle p` | Previous puzzle. |
| `puzzle 7` | Puzzle 7. |
| `puzzle goto 7` | Same, written out. |

Puzzles are in generation order — game by game, turn by turn. `open` always
starts on the first; there is no shuffle. Position is not kept between shell
sessions.

### Keeping the good ones

Generate a few hundred, browse them, and copy the keepers into a second file.
That file is itself a puzzle file — `puzzle open` reads it back.

| Command | |
|---|---|
| `puzzle keep <file>` | Append the current puzzle to `<file>` and remember it as the destination. |
| `puzzle keep` | Append to the remembered file, or to a name derived from the source. |
| `puzzle unkeep` | Take the current puzzle out of the collection it is in. |
| `puzzle kept` | Which file, and how many are in it. |

```
> puzzle open puzzles.jsonl
> puzzle keep favorites.jsonl
Kept puzzle 1 of 200 → favorites.jsonl (1 kept).
> puzzle n
> gen 5                       # look at the top plays before deciding
> puzzle n                    # not that one; moving on keeps nothing
> puzzle keep                 # same file, no need to name it again
Kept puzzle 3 of 200 → favorites.jsonl (2 kept).

> puzzle open favorites.jsonl # the collection, on its own
```

`puzzle list` marks what you have kept:

```
puzzles.jsonl: 4 puzzles   (* kept in favorites.jsonl)
  *   1  turn 15  EQUITY POINTS (+3)
->    2  turn  1  POINTS (+4)
  *   3  turn 17  EQUITY POINTS (+3)
      4  turn 10  EQUITY POINTS (+3)
```

Keeping the same puzzle twice does nothing — a puzzle is identified by its game
and turn, so a file built from several sources cannot collect duplicates.
Reopening an existing collection picks up what is already in it.

With no destination ever named, the derived name advances the round rather than
stacking suffixes:

| Browsing | `puzzle keep` with no destination goes to |
|---|---|
| `puzzles.jsonl` | `puzzles-kept.jsonl` |
| `favorites.jsonl` | `favorites-kept.jsonl` |
| `puzzles-kept.jsonl` | `puzzles-kept2.jsonl` |
| `puzzles-kept2.jsonl` | `puzzles-kept3.jsonl` |

Keeping into the file you are browsing is refused; it would grow under you as it
is read.

### Reviewing a collection later

Opening `favorites.jsonl` in a fresh session puts you in a different mode: you
are reviewing a set rather than building one. `puzzle unkeep` then prunes *that*
file, and the puzzle leaves the browser with it.

```
> puzzle open favorites.jsonl
Opened favorites.jsonl: 20 puzzles.
> puzzle 7
> puzzle unkeep
Removed puzzle 7 from favorites.jsonl, the file you are browsing. 19 left.
```

### What's hidden

The header names the kind of question and the position's origin. Tags
describing the answer are counted rather than named, and the stats are withheld.

Held back: `BINGO`, `ONLY_BINGO`, `BLANK_BINGO`, `BINGO_NINE_OR_ABOVE`,
`NON_BINGO`, `POWER_TILE`, `CEL_ONLY`, `CEL_PLUS_TWOS`, together with the score,
`tiles_played` and `words_formed`.

Shown: `EQUITY` and `POINTS`, which describe the question rather than the
answer.

`puzzle list` withholds the same things. `puzzle info` shows all of it.

## The output file

One JSON object per line. `-out` appends, so a collection can be built across
several runs and browsed at once.

| Field | |
|---|---|
| `cgp` | The position in Crossword Game Position notation, opponent rack blanked. This is what `puzzle` loads. |
| `board` | The board as text, as `s` prints it. |
| `gcg` | The game up to the position, stopping before the answer. |
| `answer` | Coordinates, tiles, score, rack. |
| `tags` | All tags earned. |
| `stats` | All 22 numbers, under the names `-filter` uses. |
| `game_id` `turn` `seed` `lexicon` | Provenance. |

## Stat fields

The numeric half of the filter language: 22 measurements of the answer. Use any
of them in a filter as `field OP value`. The same names appear in `puzzle info`
and in the `stats` object of the output file.

### Scoring

| Field | Measures |
|---|---|
| `score` | Points the answer scores. |
| `equity_advantage` | Equity of the answer minus the second-best play. |
| `score_advantage` | Score of the answer minus the best of the rest. Negative when the answer is not the top scorer. |
| `tiles_played` | Tiles the answer takes off the rack. 7 is a bingo. |
| `top_score_play_tiles_played` | Same, for the highest-scoring play. |

### Word shape

| Field | Measures |
|---|---|
| `words_formed` | Words the answer makes, main word included. 1 means no crosswords; 3 or more means an overlap. |
| `main_word_length` | Length of the main word, through-tiles included. |
| `max_cross_word_length` | Longest crossword made. Set it to 2 for pure overlap plays. |
| `min_cross_word_length` | Shortest crossword made. |
| `longest_hooked_word_length` | Longest existing word touched perpendicularly by a new tile. |
| `longest_extended_word_length` | Longest existing word grown at its front or back. |

### Bonus squares covered

| Field | Measures |
|---|---|
| `tws_covered` `dws_covered` | Triple- and double-word squares covered. |
| `tls_covered` `dls_covered` | Triple- and double-letter squares. |
| `bonus_squares_covered` | All four kinds together. |

### Tile values

| Field | Measures |
|---|---|
| `max_rack_tile_score` | Face value of the best tile on the rack. |
| `total_rack_tile_score` | Face values of the whole rack added up. 7 or less means every tile is a one-pointer. |
| `max_played_through_tile_score` | Face value of the best board tile the answer plays through. |
| `max_fresh_tile_face_value_on_tls` | Best new tile placed on a triple-letter square. |
| `max_fresh_tile_face_value_on_dls` | The same, for a double-letter square. |
| `max_fresh_tile_face_value_on_letter_bonus_with_crossword` | Best new tile on a letter bonus that also makes a crossword — the X-on-a-DLS-forming-XI play, where the bonus counts twice. |

The same reference is in the shell: `help puzzlegen` and `help puzzle`.
