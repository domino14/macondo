This is an explanation of the layout and data structures used in this app.

## alphabet

The `alphabet` module contans a few data structures.

### Alphabet

The `Alphabet` struct defines an alphabet. Essentially, an alphabet is used to convert user-visible runes like `'A'` into something called a `MachineLetter`. A `MachineLetter` is just a byte representation of this letter. For example, for a standard English alphabet, `A` should map to `MachineLetter(0)` and `Z` should map to `MachineLetter(25)`.

### LetterDistribution

**Note: consider moving this to a game-specific module**

A `LetterDistribution` has a deeper correspondence to game logic. It contains the distribution of letters in a crossword game (how many As, Es, etc there are) and what their individual point values are.

### Bag

**Note: consider moving this to a game-specific module**

A `Bag` also corresponds to game logic. It contains an array of `MachineLetter`s, that are found in a crossword game bag, i.e. tiles. These tiles can be drawn, shuffled, etc.

### Rack

**Note: consider moving this to a game-specific module**

A `Rack` also corresponds to game logic. It contains essentially an array of tiles in a player's rack, but in a bit-array sort of format. For example if the player has ABCEEH on their rack it would look like:

`1, 1, 1, 0, 2, 0, 1, 0, 0, ...`

The first element corresponds to A, and so forth.

### Word

This is only used in github.com/domino14/word_db_server now. We should just move it there.
