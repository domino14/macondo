Rough inference algorithm:

The last play made kept a certain set of tiles. Let's say the play used {X} tiles, keeping {Y} tiles.

We wish to know what tiles show up in {Y} most frequently (and least frequently).

We can pretend to be our opponent at the time they made the play. We iteratively set their rack to {X} + {R}, where {R} are random tiles from the bag, enough to fill their rack. Note that even though our rack is unseen to them, {R} cannot include our rack, since we know they don't have those letters.

We then generate moves statically. If their move is in the top Z moves (or we can set an equity cutoff), then {R} is a good estimate for what they kept.

We save {R} in memory, and generate a few thousand of these.

Then, when we do sims, we can set their partial rack to each {R}, iteratively as well. Once we run out of {R} racks we can probably stop the sim. If we don't have enough {R} racks then it might mean their rack was just not easy to estimate, and we can just choose random racks. Another alternative is just to cycle through the different {R}s over and over again, drawing different tiles for the remainder of the racks.

We can also figure out the distribution of {R}s relative to what is left in the bag. If the different racks have many more Ss than would be expected by chance, then we can show this to the user. If they have fewer Js than would be expected by chance, we can show this to the user, etc.

### Notes about how to potentially do stuff

- Probability dist for all possible leaves
- Bayes formula

X = The event that opp kept specific rack leave. P(EIRST) for example
Y = The event that the opp made the specific play

Given that they made that specific play, what is the prob that they kept this rack?

Use softmax after each 100-iteration sim during inference.

Can smooth out softmax if it's likely we're playing an opp that won't see every play.

The denominator of Bayes doesn't matter since it's applied to all plays.

How to calculate P(X): in the remaining tiles in the bag, how many ways can I generate that specific leave?


The number of ways you can generate the specific leave / numbre of ways that many tiles can be in teh bag.


Example:

   A B C D E F G H I J K L M N O               CésardelSolar            52
   ------------------------------    ->            KarlHigby  ELPRUUX   19
 1|=     '       =       '     = |
 2|  -       "       "       -   |   Bag + unseen: (82)
 3|    -       '   '       -     |
 4|'     -       '       -     ' |   ? ? A A A A A A A A B B C C D D D E E E
 5|        -           -         |   E E E E E E E F F G H H I I I I I I I J
 6|  "       "       "       "   |   K L L L M N N N N N N O O O O O O O P R
 7|    '       '   '       '     |   R R R S S S S T T T T T T U U V V W W Y
 8|=     '       O M E G A     = |   Y Z
 9|    '       G R I D     '     |
10|  "       Q I     "       "   |   Turn 3:
11|        -           -         |   CésardelSolar played 10F QI for 34 pts fro
12|'     -       '       -     ' |   m a rack of EEEIQWY
13|    -       '   '       -     |
14|  -       "       "       -   |
15|=     '       =       '     = |
   ------------------------------

We want P(LUX leave | PURGE was the play).

Bayes:

This is equal to P(PURGE was teh play | LUX leave) * P(LUX leave) / P(PURGE was the play)

We can assume the denominator is a constant across all leaves.

```
macondo> sim -plies 5
Simulation started. Please do sim show and sim details to see more info
macondo> {"level":"info","starting-node-count":0,"message":"nodes"}
macondo> sim stop
Play                Leave         Score    Win%            Equity
 K5 PUR(G)E         LUX           23       36.96±3.89      -17.16±8.68
 K6 UR(G)E          LPUX          12       34.85±4.07      -22.54±9.36
11G PULER           UX            20       34.55±3.99      -22.97±9.46
11G PURL            EUX           12       32.64±4.22      -28.90±10.56
 K6 LU(G)ER         PUX           13       32.56±4.04      -27.47±9.56
11G PUL             ERUX          11       32.52±3.97      -28.55±9.52
 I8 (MI)XUP         ELRU          16       32.29±4.36      -29.57±10.98
11G PUR             ELUX          11       32.11±3.91      -30.47±9.88
 9L XU              ELPRU         19       31.41±3.99      -30.99±9.41
11G PURE            LUX           12       31.37±3.88      -31.34±9.87
11G PULE            RUX           12       31.33±3.98      -32.37±10.30
(exch LUU)          EPRX          0        31.17±4.10      -32.40±10.15
 L6 LU(A)U          EPRX          4        31.06±4.12      -32.88±10.57
 7E PURL            EUX           10       30.99±3.97      -31.33±9.44
 J7 R(ED)UX         ELPU          15       30.59±4.02      -33.02±9.70
 K5 PLU(G)          ERUX          14       30.42±3.86      -33.74±9.50
 9L LUX             EPRU          13       29.87±4.44      -37.89±11.72
 7L LUX             EPRU          13       29.72±4.37      -37.24±11.20
 K6 LU(G)E          PRUX          12       29.58±3.59      -34.65±9.02
 L5 PUL(A)          ERUX          6        29.50±3.86      -35.44±9.29
 I5 PLU(MI)ER       UX            12       29.36±4.04      -36.94±10.42
 J5 PUL(ED)         ERUX          10       29.14±3.91      -36.76±9.54
 J5 LUR(ED)         EPUX          8        29.13±4.09      -37.71±10.45
 L3 PLEUR(A)        UX            16       28.99±3.66      -36.57±9.34
 K6 LU(G)           EPRUX         4        28.78±3.95      -38.37±10.17
 7F PUL             ERUX          9        28.73±3.79      -38.21±9.70
 K6 PU(G)           ELRUX         6        28.36±3.67      -37.90±9.19
 J5 RUL(ED)         EPUX          8        28.16±3.83      -39.83±10.27
 L6 PR(A)U          ELUX          6        28.15±4.05      -39.35±10.14
 J4 PURE(ED)        LUX           11       28.09±3.94      -40.61±10.47
 J6 RU(ED)          ELPUX         7        27.94±3.95      -39.11±9.95
 J5 URP(ED)         ELUX          10       27.27±3.96      -42.56±10.27
 7L LUXE            PRU           14       27.26±4.10      -46.94±13.01
 7L PURL            EUX           11       26.39±3.80      -45.77±10.75
 J4 PURL(ED)        EUX           11       26.23±3.60      -44.34±9.63
 7L PUL             ERUX          10       25.86±3.74      -46.26±10.14
 7L PULE            RUX           11       25.72±3.75      -50.16±12.45
 9L LUXE            PRU           14       25.43±3.70      -49.04±11.22
 7L PURE            LUX           11       25.22±3.82      -51.91±12.62
 7L PUR             ELUX          10       24.77±3.82      -48.70±10.04
Iterations: 157 (intervals are 99% confidence, ❌ marks plays cut off early)
```

to figure out P(PURGE was teh play | LUX leave) we can use Softmax with the above data.

Let's imagine PURGE was 0.76 after doing softmax, and URGE was 0.2, and the others were 0.04 across all the other ones. So then that P was 0.76 (for example).

Then, what is P(LUX is leave) (naive probability of rack)?

(4c1 * 4c1 * 1c1) / (85c3)

Then multiplying these numbers together gives me a relative probability.

Then we can use these probabilities to sample racks for inferences.

#### Next steps
- Recursion?

P(X | Y,Z) = (P(Y | X,Z) * P(X | Z)) / C

- Individual tiles that are more or less likely:

Can we do the above process but only with 1-tile leaves?

- Tree-based logic to expedite which candidate rack leaves to consider?

- As the simulator improves, the quality of this inferencing will also improve.

- Train neural net to try to estimate what the score potential of a given play is on future turns. (Sum of pts scored in next turns). That can be a board dynamics measure.

We can create a 3D space of (pts, leave, board_dynamic) scores.

Then we can find the "maximal coverage" -- so that we can have as few plays as possible cover as much of this 3D space as possible. We can generate up to K unique plays. Then we can cut off plays that are bad and sample other parts of this 3D space as the sim progresses.