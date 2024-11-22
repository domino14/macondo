### Game situation
We are losing the game 46-66. Our rack is ABCEEQU and there are 80 tiles unseen to us (so 7 in our opponent's rack, and 73 in the bag).

### Simulation results:

Play                Leave         Score    Win%            Equity
 5E BE(R)CEAU       Q             44       48.39±0.94      -4.90±2.03
 E6 QU(E)BEC        AE            38       46.63±1.02      -8.60±2.21      ❌
 E4 QUEB(E)C        AE            38       46.05±1.42      -9.67±3.09      ❌
 F4 CABE(R)         EQU           29       43.83±2.28      -14.53±4.99     ❌
 5E BA(R)QUE        CE            34       43.52±2.54      -16.79±5.93     ❌
Iterations: 3593 (intervals are 99% confidence, ❌ marks plays cut off early)

### Simulation details:

**Ply 1 (Opponent)**
Play                Leave             Win%    Mean   Stdev Bingo %   Iters
---------------------------------------------------------------------------
 5E BE(R)CEAU       Q                48.39  47.140  21.468  26.468    3593
 E6 QU(E)BEC        AE               46.63  45.631  22.093  25.021    3589
 E4 QUEB(E)C        AE               46.05  48.007  21.540  26.423    1809
 F4 CABE(R)         EQU              43.83  50.503  22.656  27.900     638
 5E BA(R)QUE        CE               43.52  54.476  26.568  26.989     641

**Ply 2 (You)**
Play                Leave             Win%    Mean   Stdev Bingo %   Iters
---------------------------------------------------------------------------
 5E BE(R)CEAU       Q                48.39  49.680  19.683   3.312    3593
 E6 QU(E)BEC        AE               46.63  46.687  23.432  26.553    3589
 E4 QUEB(E)C        AE               46.05  47.001  22.384  25.207    1809
 F4 CABE(R)         EQU              43.83  56.152  21.241  11.129     638
 5E BA(R)QUE        CE               43.52  52.192  26.286  28.549     641

### Detailed stats for the winning play (5E BE(R)CEAU):

### Opponent's next play
Play                Score    Count    % of time
 F4 R(E)Z           63       36       1.00
 F4 V(E)ZI(R)       70       24       0.67
 F4 S(E)Z           63       23       0.64
 F4 F(E)Z           66       21       0.58
 6E OW(E)           31       20       0.56
F10 HOA             37       15       0.42
 F4 B(E)Z           65       14       0.39
 F3 GE(E)Z          65       13       0.36
 F8 (R)ADIO         29       13       0.36
F10 GIO             25       12       0.33
 F2 VIN(E)W         40       12       0.33
F10 KOA             43       12       0.33
 F4 Y(E)Z           66       12       0.33
 F3 JE(E)Z          71       12       0.33
 7I JOW             38       11       0.31
Bingo probability: 26.47%

### Our follow-up play
Play                Score    Count    % of time
F10 QI              67       764      21.26
10F Q(I)            62       329      9.16
F10 QIN             73       263      7.32
10F Q(I)N           32       132      3.67
 9C QI              30       127      3.53
F10 Qi              65       90       2.50
 I7 Q(A)T           23       61       1.70
 7C QAT             33       59       1.64
 9C QIN             33       42       1.17
 J8 SUQ             48       41       1.14
F10 QiN             71       40       1.11
11D WAQ(F)          38       40       1.11
 J2 TAL(A)Q         36       39       1.09
 F4 R(E)Z           63       23       0.64
 6B QADI            38       23       0.64
Bingo probability: 3.31%