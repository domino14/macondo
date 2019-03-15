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
