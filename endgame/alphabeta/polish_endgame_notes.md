The game in the following file was sent to me by a player in the Woogles Discord.

`../../gcgio/testdata/polish_endgame.gcg`

It is an interesting endgame for several reasons:

- It involves both players being stuck with tiles
- It breaks the iterative deepening endgame algorithm (as of Dec 17, 2021). It only solves it correctly with iterative deepening off.
- There are a relatively small number of moves available each turn for both players, which should make it easier to test our minimax / alpha-beta pruning / iterative deepening / etc algorithms to make sure they work properly.
- It is in Polish (OSPS44) and involves cool looking words.

#### The position

At turn 46, we are in an endgame, and player 1 is winning 304-258 (+46)

The best sequence appears to be the following, on turn 46:

```
Best sequence:
1) N7 ZG.. (+12)
2) M1 ŻM.. (+13)
3) (Pass)  (+0)
4) 6L .I   (+2)
5) B8 ZU.  (+6)
6) 9A K.   (+5)
7) (Pass)  (+0)
8) (Pass)  (+0)
(and pass until game ends, as both are stuck)
```

This results in a total spread gain of 5 points from the point of view of the first player (ptf1559)
Final score: 315-264 (+51)

Iterative deepening seems to be broken if invoked on turn 47, after N7 ZG(OŃ) (note: it is not actually broken, see below). It results in this sequence, from the point of view of player 2:

```
Best sequence:
1) M1 ŻM.. (+13)
2) (Pass)  (+0)
3) (Pass)  (+0)
```

Let's try to invoke ID semi-manually and test out a few sequences.

### Why pass on move 3?

First of all, why is Pass the best move after `M1 ŻM..`?

Player 1 is winning 316-271 (+45) here, after `M1 ŻM..`.

There are only three possible tile-playing moves for player 1:

```
7M B(Z)U 10
7M B(Z)Z 8
B8 ZU(P) 6
```

Let's examine these 1 by 1.

#### 7M B(Z)U 10

Then:

1. B8 KÓ. (9)
2. O6 H. (6)
3. 8B .I (3)
4. (Pass) (0)
5. 8B ..Ź (21)

Final score 332-308 (+24)

Basically, this allows opponent to set himself up and go out! This is a net (-21) spread for P1.

#### 7M B(Z)Z 8

Then:

1. O5 KÓ. (8)
   (why not the same tiles at B8? Because then opp can play. `O5 KÓ.` blocks all of opp's moves)
2. Forced pass
3. 1L I. (7)
4. Both players pass until the end

Final score 317-277 (+40)

This is a net (-5) spread for P1. Not as bad as BZU.

#### B8 ZU(P) 6

Then:

1. 9A K. (5)
2. (Pass) (no other moves playable)
3. 1L I. (7)
4. (Pass)
5. (Pass)

Final score 315-269 (+46)

This is a net (+1) spread for P1.

#### Pass

Then:

1. 6L .I (2)

   This blocks B(Z)U, so it's better than `1L I.` (7), even though the latter scores 5 more pts. But if we don't block BZU, wouldn't we get a massive advantage as in the analysis for 7M B(Z)U above?

   We get a lot of that massive advantage by saving our `I` for the KÓP/KI/KIŹ setup.

   If instead we played KÓP (saving our I) then the opp can block the I setup with (K)U, then rack up a ton of points with KUB, WU, ZWU while we only have one other place for the I.

2. B8 ZU. (6) (only available move)
3. 9A K(U) (5) (only available move)
4. (pass until game over)

Final score (315-264) (+51)

This is a net (+6) for Player 1, so Pass is best.

### Let's look at move 4

After P1's pass on move 3, the best move for P2 on move 4 is 6L .I (2). This is explained above.

What if P2 passes to end the game? (Pretend game ends after 2 instead of 6 passes for the purposes of the endgame).

First of all, if P2 takes the proper line of `6L .I (2)`, and the game continues as in the case above, then the final score is 315-264, so a total spread of (-51) as seen from P2's point of view.

If P2 instead passes, the game ends 305-254, which is also a total spread of (-51). So from P2's perspective, passing to end the game is just as good as playing `6L .I (2)`! However, we know that the game doesn't actually end after two turns. If P2 passes, P1 should pass back, as it's still P1's best move, and so forth.

If we plug this into macondo, and plug in a pass for P2 after P1's pass, then do `endgame 14`, this results in:

```
Best sequence:
1) (Pass)
2) (Pass)
3) (Pass)
4) (Pass)
5) (Pass)
6) (Pass)
7) (Pass)
8) (Pass)
9) (Pass)
10) 1L I.
11) B8 ZU.
12) 9A K.
13) (Pass)
14) (Pass)
```

It is generating a long series of spurious passes. This is likely due to a bug where it thinks the game is already over (remember, the endgame ends after 2 passes for simplicity). So this bug should be fixed.

### Changes

- We will value not passing over passing if both nodes are otherwise equal. This may increase the chance of an opponent mistake.
- We may need to change the scoreless turns behavior at the beginning of the endgame. We may want to set it to 0 no matter what?
