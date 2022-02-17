cgp stands for Crossword Game Position, and it is a file format to use when a position in a game is available, but not the full history that got us there. This can be used for puzzles and other analyses.

It is intended to only support the UTF-8 encoding.

## Description

Influenced by the FEN and EPD formats in Chess, a CGP record consists of several fields, separated by spaces, all on one line. It is meant to be concise and easily shareable for import into analysis programs.

The fields are as follows:

1. Tile placement

Tiles on the board are represented as the uppercase UTF-8 codepoint that represents them. Blanks on the board can be represented by the corresponding lowercase UTF-8 codepoint.

Each row of the board can then be represented by any number of letter characters, and each row should be separated by the `/` character.

In order to represent an unused space on the board, we use a number -- `1` means 1 empty space, `8` means 8 empty spaces in a row, etc.

Then, an empty 15x15 board looks like this:

`15/15/15/15/15/15/15/15/15/15/15/15/15/15/15`

A row with a D in the middle of it (the 8th position) would look like:

`7D7`

2. Racks for all involved players, in the order that they will next play. Each rack is represented by its letters, and racks are separated by a `/` symbol. A blank is represented by a `?`.

e.g.

`AB/AA?` means the player to play next has a rack of AB and the other player has a rack of AA? (? being the blank).

If one of the players' racks is only partially known, specify as many letters as are known. If no letters are known, leave that player's rack blank.

e.g.

`/ABCDEF` or `ABCDEF/`

3. Scores for both players, represented as regular numbers, separated by a `/` symbol. The order should be in the order of the racks.

e.g.

`357/289`

4. Number of consecutive zero turns immediately prior to the position being shown. This should be a single number. We use this because in some implementations of crossword games, the game is over after six consecutive zero turns, for example.

e.g.

`5`

5. One or more "opcodes" with optional arguments; each opcode would be separated by `;` characters and optional spaces.

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

n8 runes

where n8 is the coordinate - rows are numbered 1 to 15 and columns are lettered from A to O for a 15x15 board. Horizontal plays start with the numbered row, vertical plays start with the lettered column.

Runes are just the letters on the rack. "Through" tiles, i.e., tiles already on the board, do not need to be specified and can be specified as a `.`.

If the move is an exchange, represent as `exchange runes` if the runes are known, or replace the runes with a number of exchanged tiles.

If the move is a pass or a failed challenge, represent it as `pass`.

If the move is a successful challenge, represent it as `challenge n8 runes pts` where runes are the played letters and pts is the number of pts the challenged play would have scored.

Note: The lm opcode can be specified multiple times. It should contain the last plays in the reverse order that they were played, by all players.

### mcnz (max number of consecutive zeroes)

Maximum number of consecutive zeroes until the game ends, for the given rule set. This should default to 6 if not specified.

## Example CGPs

1. From the 2018 US Nationals, game 30. Joel's opening rack was AENSTUU and he exchanged UU. Nigel's opening rack was AELNOQT. From his perspective:

`15/15/15/15/15/15/15/15/15/15/15/15/15/15/15 AELNOQT/ 0/0 0 lex NWL18; lm exchange 2`

Note: he exchanged QO instead of playing QAT!

2. Endgame from Maven paper (score is unknown, but it doesn't matter)

`5BERGS5/4PA3U5/2QAID3R5/3BEE3F2S2/1P1ET2VIATIC2/MA1TAW3c2H2/ES3IS2E2A2/AT1FOLIA4V2/LI1L1EX1E6/1N1O1D2N2Y3/1GNU2C1JETE3/2ER2OHO2N3/2O3GOY6/1INDOW1U7/4DORR7 IKLMTZ/AEINRU? 0/0 0 lex OSPD1;`
