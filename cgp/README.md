cgp stands for Crossword Game Position, and it is a file format to use when a position in a game is available, but not the full history that got us there. This can be used for puzzles and other analyses.

It is intended to use ASCII characters only, but it is not meant to be human-readable.

### Description
Influenced by the FEN and EPD formats in Chess, a CGP record consists of several fields, separated by spaces, all on one line. It is meant to be concise and easily shareable for import into analysis programs.

The fields are as follows:

1. Tile placement

In crossword games, which can be played in many languages, some alphabets might have more or fewer than 26 letters, letters might also be digraphs, or not be in the ASCII character set at all. It is therefore desirable to map each letter to a number like:

```
1 A
2 B
...
```

and so on. Since these early numbers are not easily representable in ASCII, we add 0x20 (32) to each number, and represent the letters as such:

```
33 A  !
34 B  "
35 C  #
...
58 Z  :
```

We will call this the "CGP rune representation".

Blanks on the board can be represented by the corresponding number, plus 48. 
We can represent up to 42 different tiles. (The character 127 is a DEL and thus not easily representable, and we want to reserve ASCII 124 for row separators).

So for example, a blank B would be 34 + 48 = 82 = the character R 
(again, this is not meant to be human-parsable).

Each row of the board can then be represented by any number of letter characters, and each row should be separated by the `|` character (ASCII 124).

In order to represent an unused space on the board, we use a form of run-length encoding. We use the ~ character (ASCII 126) to represent one or more spaces together as follows:

- Use ~ by itself to represent one single space.
- Use the ~ character twice to encode any runs of two or more spaces, followed by the number of spaces. Note that the number needs to be encoded the same way the letters are encoded, i.e. in "CGP rune representation": ASCII 34 is 2, ASCII 35 is 3, etc.

Thus an empty 15x15 board would look like:

`~~/|~~/|~~/|~~/|~~/|~~/|~~/|~~/|~~/|~~/|~~/|~~/|~~/|~~/|~~/`

A row with a Z in the middle of it (the 8th position) would look like:

`~~':~~'`

2. Racks for all involved players, in the order that they will next play. Each rack is represented in CGP rune representation (see part 1), and racks are separated by a | symbol. A blank is represented by a ~ (since it is not assigned at this time).

e.g.

`!"|!!~`  means the player to play next has a rack of AB and the other player has a rack of AA? (? being the blank).

3. Scores for both players, represented as regular numbers, separated by a `|` symbol. The order should be in the order of the racks.

e.g.

`357|289`

4. Number of consecutive zero turns immediately prior to the position being shown. This should be a single number. We use this because in some implementations of crossword games, the game is over after six consecutive zero turns, for example.

e.g.

`5`

5. One or more "opcodes" with optional arguments; each opcode would be separated by | characters and optional spaces. 

### Opcodes

**bb** (bingo bonus)

The number of bonus points a bingo is worth. Default to 50.

**bdn** (board name)

The bdn opcode should be followed by the name of the board. For example:

`bdn CrosswordGame`

If not specified, the implementer decides what its default board is. *The format attaches no special meaning to the board names that are provided in this opcode*. A future opcode might describe the actual board configuration square by square.

**etl** (exchange tile limit)

The etl opcode should be followed by the minimum number of tiles that must be in the bag to allow an exchange. Defaults to 7.

**ld** (letter distribution)

The ld opcode should be followed by the name of the letter distribution. For example:

`ld english`

If not specified, the implementer decides what its default letter distribution is. The letter mapping (CGP rune representation) will need to match your letter distribution. It is recommended that the letter distribution also contain the number of points for each tile.

**lm** (last move)

The lm opcode is followed by the move, in the following format:

n8 runes

where n8 is the coordinate - rows are numbered 1 to 15 and columns are lettered from A to O for a 15x15 board. Horizontal plays start with the numbered row, vertical plays start with the lettered column.

Runes are in CGP rune representation. "Through" tiles, i.e., tiles already on the board, do not need to be specified and can be specified as a `~`.

If the move is an exchange, represent as `exchange runes` if the runes are known, or replace the runes with as many `~`s as there were exchanged tiles.

If the move is a pass or a failed challenge, represent it as `pass`.

Note: The lm opcode can be specified multiple times. It should contain the last plays in the reverse order that they were played, by all players.

**mncz** (max number of consecutive zeroes)

Maximum number of consecutive zeroes until the game ends, for the given rule set. This should default to 6 if not specified.


