mlproducer is a producer of data for the purposes of training an ML model to learn how to play a crossword board game.

It has two components:

- turn_scanner scans a file with turns, one per row. This file can be produced using the macondo `autoplay` command. It is saved by default to `/tmp/autoplay.txt`.

- game_assembler assembles games and does the hard work of actually turning it into features for a model. A more in-depth description below.

#### Responsibilities for game_assembler

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Maintain a map[GameID]gameState:                                    │
│    • board (15×15 rune / byte array)                                   │
│    • racks for both players                                            │
│    • scores, bag, turn number                                          │
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. FeedTurn(t Turn) steps                                               │
│    a. Fetch or create gameState                                         │
│    b. Apply t.Play to the board                                         │
│       • if "(exch …)" remove tiles from rack, draw same count from bag  │
│       • else lay tiles, update bag                                      │
│    c. Compute *post-move* feature tensor                                │
│       • 60 binary planes (letters + premiums + cross-checks)            │
│       • 56 scalar features                                              │
│    d. Append {feature, scoreDiff} to the game’s sliding window          │
│    e. When window ≥ horizon+1, emit a training vector                   │
│       • x = feature(t)                                                  │
│       • y = spread gain after horizon plies  (or other label)           │
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. When TilesRemaining == 0 AND both racks empty → game over            │
│    • Flush remaining window positions                                   │
│    • Delete state from map to free memory                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Motivation and functionality

It helps to think about how exactly we want to use this model for inference. The basic idea is to replace the simple static evaluator (score + leave) with something that takes more parameters into account.

Imagine we are faced with a position:

```
   A B C D E F G H I J K L M N O     ->             Player_1  EEEIILZ  336
   ------------------------------                   Player_2           298
 1|C     '       =       '     =
 2|O -   T O Y       "       -      Bag + unseen: (16)
 3|m I R A D O R   '       -
 4|F     -   D A B     P U G H '    A A A C E E I I I I L N N Q R S
 5|I       -   G O O E Y       V
 6|T "       X I     M A L T H A
 7|    '       '   '       '   N
 8|=     '     G U M     ' O W N
 9|    '       ' P E W     D O E
10|  "       "       E F   D O R    Turn 20:
11|    K U N A   J   B E V E L S    Player_2 played 4K PUGH for 28 pts from a
12|'     T U R R E T s   - S   '    rack of GHPU
13|    -       ' A '       T
14|  -       "   N   "       -
15|=     '       S       '     =
```

In this position we wish to evaluate all of the candidate plays for Player_1. The inference would work as follows:

- For every candidate play:
    - Place the play on the board
    - Evaluate the board position with the following parameters:
        - The candidate play on the board
        - All of the cross-sets for this new board
        - Our leave
        - The score of the play
        - The bag state, prior to drawing replacement tiles (we don't know what we're going to draw!)
- Sort all plays by evaluation and display/pick highest one

#### Training set

Therefore the training set must match whatever we are inferring. We should train with a set of game positions where the side that just played hasn't yet drawn replacement tiles. Let's imagine that HastyBot played M1 ELE(G)I(T) above, the highest equity play. We then train this position. There's a good chance the opponent's play next turn was a high scoring bingo. So the evaluation we should give this position will be the spread _gain_ after N turns, which is likely to be more negative after ELEGIT than after a play like `14F ZINE`.

The training vector for this position would contain:
- Board vectors for board + cross sets containing ELEGIT
- Our rack leave (EIZ)
- The unseen tiles prior to drawing (16 tiles: `A A A C E E I I I I L N N Q R S`)
- Spread after play is made
- The last opponent move tiles marked as "just played".

The predictor would be the spread gain after N turns, and maybe the game's final spread.