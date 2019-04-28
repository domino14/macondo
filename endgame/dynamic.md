The dynamic programming endgame algorithm as described in Sheppard's paper.

```
Nathan Benedict: Turn 14
   A B C D E F G H I J K L M N O   -> Nathan Benedict          RR      365
   ------------------------------     JD                       LN      510
 1|G R R L       d     Q '     =|   TRACKING
 2|  E       J   E   G I     -  |    LN  2
 3|  M M   L A ' N Y E     -    |
 4|'   O B I     O O N   -     '|
 5|    K I P     T   I F        |
 6|  W E T A "   I   T O     "  |
 7|S O ' C     ' V ' O B   '    |
 8|U T   H E D G E   R   ' V   =|
 9|I   '     U '   '       E    |
10|  "       P       F A A N "  |
11|    C   - E N T A Y L E D    |
12|W   H O A S   A       - I   '|
13|E   I       ' T '       N    |
14|D A Z E D "   O   "     g -  |
15|S     X I     U     U R S A E|
   ------------------------------
```

Generated moves for Nathan:

(GI)RR 5, (TO)RR 4, (E)RR 3, (A)R 4, R(OON) 4, (TO)R 3, (U)R 2, R(E) 2, (O)R 2, (E)R 2

For JD:

L(U)N 3, N(E) 5, (A)N 4, (A)L 4, (E)N 4, (I)N(TO) 4, N(OON) 4, L(OON) 4, (GI)N 4, (AL)L 3, (TO)N 3, (U)N 2, N(U) 2, N(E) 2, N(U) 2


```
We can statically compute the values of the racks if N = 0 (i.e., if the
opponent goes out before we move). That is simply −2 times the sum of the
tiles. For N > 0 we use dynamic programming.
For every move that can be played out of each rack we compute the score of
that move plus the evaluation of the remaining tiles assuming that N − 1
turns remain. The value of a rack in N turns is the largest value of that
function over all possible moves.
```

    Play            N   Formula

    (GI)RR 5        1   5 + val(null, 0) = 5
    (TO)RR 4        1   4 + val(null, 0) = 4
    (E)RR 3         1   3 + val(null, 0) = 3
    (A)R 4          7   4 + val(R, 6) = null ?
    R(OON) 4        7   4 + val(R, 6) = null ?

    (A)R 4          6   4 + val(R, 5) = null ?
    .
    .
    (A)R 4          2   4 + val(R, 1) = 4 + 4 = 8
    R(OON) 4        2   4 + val(R, 1) = 4 + 4 = 8
    (A)R 4          1   4 + val(R, 0) = 4 + (-2) = 2
    R(OON) 4        1   4 + val(R, 0) = 4 + (-2) = 2
    (TO)R 3         1   3 + val(R, 0) = 3 + (-2) = 1
    (U)R 2          1   2 + val(R, 0) = 2 + (-2) = 0
    R(E) 2          1   0
    (O)R 2          1   0
    (E)R 2          1   0

    This table is filled in recursively by going through plays above,
    and filling in the 0 case manually.

    Rack    N   Value
    RR      0   -4
    R       0   -2
    RR      1   max (table above where N = 1) = 5
    R       1   max (table above where N = 1 and we play one R) = 4
    RR      2   max (table above where N = 2) = 8
```

Interpretation of Maven paper:

```
It is impractical to build a table of all possible racks that we hold versus all possible racks that our opponent holds, as there are up to 128 possible racks for each side. But fortunately we do not need the full set. Simply knowing what one side can achieve within N turns (for N = 0, . . . , 7) is sufficient.
```

Strange that this is "impractical" as 128 racks isn't that many, even 20+ years ago? But I think the point is that we only need to keep track of the main rack itself plus the N = 0 to 7. So we have a lot of extraneous entries in our table?

```
For every move that can be played out of each rack we compute the score of that move plus the evaluation of the remaining tiles assuming that N − 1 turns remain. The value of a rack in N turns is the largest value of that function over all possible moves.
The beauty of this algorithm is that we can statically determine how many turns it will take a given rack to play out. If we wish to know whether DERTTT will play out in N moves, we simply check to see whether there are 0 tiles left if DERTTT has N moves to play.
```

The very last sentence refers to the first sentence. The score of that move plus the evaluation of the remaining tiles with N-1 turns. So if DERTTT has N moves to play, the evaluation of the remaining tiles would not exist since the remaining tiles would be 0? Should our table keep track of how many tiles are left for each N?

```
Now assume that MAVEN moves first, holding DERTTT and the opponent has EJVY. What is the outcome? MAVEN starts by computing the result assuming that DERTTT will play eight moves (and EJVY will play seven moves). That gives us one sequence which we will minimax to determine the right answer.
```

DERTTT will play eight moves (two are possibly passes). What is the sequence we are minimaxing here? Based on the fact that we only have two tables with the full racks (DERTTT and EJVY only, no subracks), and descending Ns, the minimax algorithm is simple enough. Every node for player 2 that is at or below EJVY(7) is a child of the DERTTT(8) node. (?)
So we are only minimaxing one ply deep. I think.