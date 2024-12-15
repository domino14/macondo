### Game situation
We are losing the game by 40 points. Our rack is ACDEPQU and there are 16 tiles unseen to us (so 7 in our opponent's rack, and 9 in the bag).

### Simulation results:

Play                Leave         Score    Win%            Equity
12K QU(ID)          ACDEP         28       34.89±3.30      17.32±3.22
12K QA(ID)          CDEPU         28       28.25±3.08      13.02±3.25      ❌
 3A QUA(t)          CDEP          24       28.19±3.10      14.26±3.31      ❌
 3B QA(t)           CDEPU         22       27.44±3.39      8.19±3.84       ❌
12J EQU(ID)         ACDP          30       25.10±4.33      10.21±4.25      ❌

Iterations: 1297 (intervals are 99% confidence, ❌ marks plays cut off early)

### Simulation details:

**Ply 1 (Opponent)**
Play                Leave             Win%    Mean   Stdev Bingo %   Iters
---------------------------------------------------------------------------
12K QU(ID)          ACDEP            34.89  35.045  21.910  25.289    1297
12K QA(ID)          CDEPU            28.25  31.848  20.403  21.395    1290
 3A QUA(t)          CDEP             28.19  28.700  21.960  22.412    1285
 3B QA(t)           CDEPU            27.44  28.704  22.137  22.537    1025
12J EQU(ID)         ACDP             25.10  39.532  22.833  37.947     643

**Ply 2 (You)**
Play                Leave             Win%    Mean   Stdev Bingo %   Iters
---------------------------------------------------------------------------
12K QU(ID)          ACDEP            34.89  45.216  21.540  21.434    1297
12K QA(ID)          CDEPU            28.25  36.725  20.522  16.357    1290
 3A QUA(t)          CDEP             28.19  37.471  20.726  15.564    1285
 3B QA(t)           CDEPU            27.44  37.108  20.965  15.902    1025
12J EQU(ID)         ACDP             25.10  40.378  19.244  19.907     643

### Detailed stats for the winning play (12K QU(ID)):

### Opponent's next play
Play                Score    Count    % of time
 2B RI(T)           19       75       5.78
 2B RU(T)           19       50       3.86
J12 EUOI            21       48       3.70
 2B OU(T)           19       40       3.08
K10 RO(Q)UE         28       40       3.08
O10 OOSE            38       40       3.08
O12 SLO(T)          28       39       3.01
O12 SLU(T)          28       35       2.70
J10 LIEU            21       35       2.70
K10 GU(Q)IN         30       30       2.31
12G ILLI(QUID)      19       27       2.08
 4F LE(V)O          22       24       1.85
J10 GIE             23       23       1.77
 1A (COXA)L         14       19       1.46
13H LIRI            16       18       1.39
Bingo probability: 25.29%

### Our follow-up play
Play                Score    Count    % of time
15G PREAD(JUST)     57       195      15.03    # Potential setup
 5D (S)CAP(A)       28       106      8.17
 2B PI(T)           27       101      7.79
 5D (S)COP(A)       28       100      7.71
 2B PU(T)           27       59       4.55
 B8 (R)EPLACED      88       42       3.24
 5D (S)COP(A)E      30       41       3.16
 5D (S)PIC(A)E      30       40       3.08
 B8 (R)ESPACED      80       30       2.31
 5A LAP(S)ED        32       28       2.16
 5D (S)PIC(A)       28       25       1.93
 J9 CAPERED         81       20       1.54
 J9 PEASCOD         81       14       1.08
J10 CLEG            28       12       0.93
 J9 CLASPED         81       12       0.93
Bingo probability: 21.43%