cgp stands for Crossword Game Position, and it is a file format to use when a position in a game is available, but not the full history that got us there. This can be used for puzzles and other analyses.

It is intended to only support the UTF-8 encoding.

## Description

Influenced by the FEN and EPD formats in Chess, a CGP record consists of several fields, separated by spaces, all on one line. It is meant to be concise and easily shareable for import into analysis programs.

The fields are as follows:

### 1. Tile placement

Tiles on the board are represented as the uppercase UTF-8 codepoint that represents them. Blanks on the board can be represented by the corresponding lowercase UTF-8 codepoint.

Each row of the board can then be represented by any number of letter characters, and each row should be separated by the `/` character.

In order to represent an unused space on the board, we use a number -- `1` means 1 empty space, `8` means 8 empty spaces in a row, etc.

Then, an empty 15x15 board looks like this:

`15/15/15/15/15/15/15/15/15/15/15/15/15/15/15`

A row with a D in the middle of it (the 8th position) would look like:

`7D7`

Note: for tiles that consist of multiple codepoints, such as CH in Spanish, these must be specified as `[CH]` (or `[ch]` lowercase), for example.

### 2. Racks for all involved players, in the order that they will next play.

Each rack is represented by its tiles, and racks are separated by a `/` symbol. A blank is represented by a `?`.

e.g.

`AB/AA?` means the player to play next has a rack of AB and the other player has a rack of AA? (? being the blank).

If one of the players' racks is only partially known, specify as many tiles as are known. If no tiles are known, leave that player's rack blank.

e.g.

`/ABCDEF` or `ABCDEF/`

Note that this supports any number of players.

### 3. Scores for both players

These are represented as regular numbers, separated by a `/` symbol. The order should be in the order of the racks.

e.g.

`357/289`

Note that this supports any number of players. Racks and scores would both have n-1 slashes for n-player games, starting with the next-to-move player and ending with the player who last moved

### 4. Number of consecutive zero-point turns

This number is the number of consecutive zero-point turns immediately prior to the position being shown. This should be a single number. We use this because in some implementations of crossword games, the game is over after six consecutive zero turns, for example.

e.g.

`5`

### 5. Any number of optional operations

An operation is composed of an opcode followed by zero or more operands and is concluded by a semicolon.

Multiple operations are separated by a single space character. If there is at least one operation present in a line, it is separated from the last (fourth) data field by a single space character.

## Opcodes

### bb (bingo bonus)

The number of bonus points a bingo is worth. Default to 50.

### bdn (board name)

The bdn opcode should be followed by the name of the board. For example:

`bdn CrosswordGame;`

If not specified, the implementer decides what its default board is. _The format attaches no special meaning to the board names that are provided in this opcode_. A future opcode might describe the actual board configuration square by square.

### cr (challenge rule)

The cr opcode should be followed by the name of the challenge rule. These are:

single double triple void 5pt 10pt

The implementer decides what the default challenge rule is.

### etl (exchange tile limit)

The etl opcode should be followed by the minimum number of tiles that must be in the bag to allow an exchange. Defaults to 7.

### ld (letter distribution)

The ld opcode should be followed by the name of the letter distribution. For example:

`ld english;`

If not specified, the implementer decides what its default letter distribution is. It is recommended that the letter distribution also contain the number of points for each tile.

### lex (lexicon)

The lex opcode should be followed by the name of the lexicon. For example:

`lex CSW21;`

### lm (last move)

The lm opcode is followed by the move, in the following format:

`n8 tiles`

where n8 is the coordinate - rows are numbered 1 to 15 and columns are lettered from A to O for a 15x15 board. Horizontal plays start with the numbered row, vertical plays start with the lettered column. For boards that are bigger than 26 columns, use "excel" row notation, i.e. Z, AA, AB, ..., AAA, AAB, ...

"Through" tiles, i.e., tiles already on the board, must be specified as a `.`.

If the move is an exchange, represent as `exchange tiles` if the tiles are known, or replace the tiles with a number of exchanged tiles.

If the move is a pass or a failed challenge, represent it as `pass`.

If the move is a successful challenge, represent it as `challenge n8 tiles pts` where tiles are the played tiles and pts is the number of pts the challenged play would have scored.

Note: The lm opcode can be specified multiple times. It should contain the last plays in the reverse order that they were played, by all players.

### mcnz (max number of consecutive zeroes)

Maximum number of consecutive zeroes until the game ends, for the given rule set. This should default to 6 if not specified.

### ti (timer increment)

The timer increment in milliseconds, if one exists.

e.g.

`ti 3000;` means that 3 seconds are added to player clocks after they move

### tmr (timer)

Timers can be specified with slashes between players. They must be specified as milliseconds remaining on each player's clock.

e.g.

`tmr 10000/-2500;`

means the player to move has 10 seconds on their clock, and the other player has gone overtime by 2.5 seconds.

### to (timer max overtime setting)

The amount of overtime before the game ends, in milliseconds.

e.g.

`to 60000;` means the game is over after 1 minute of overtime (in typical play, the person who went over loses 10 points for every minute of overtime).

## Example CGPs

1. From the 2018 US Nationals, game 30. Joel's opening rack was AENSTUU and he exchanged UU. Nigel's opening rack was AELNOQT. From his perspective:

`15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 AELNOQT/ 0/0 0 lex NWL18; lm exchange 2;`

Note: he exchanged QO instead of playing QAT!

2. Endgame from Maven paper (score is unknown, but it doesn't matter)

`5BERGS5/4PA3U5/2QAID3R5/3BEE3F2S2/1P1ET2VIATIC2/MA1TAW3c2H2/ES3IS2E2A2/AT1FOLIA4V2/LI1L1EX1E6/1N1O1D2N2Y3/1GNU2C1JETE3/2ER2OHO2N3/2O3GOY6/1INDOW1U7/4DORR7 IKLMTZ/AEINRU? 0/0 0 lex OSPD1;`
