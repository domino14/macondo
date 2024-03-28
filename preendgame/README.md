There is no set definition for pre-endgame, but it is commonly understood as when it is no longer possible to exchange; i.e. when there are between 1 and 6 tiles in the bag, inclusive. Once the bag is empty, we have reached the endgame.

Some people consider the pre-endgame to start even earlier.

Just as it is possible to exhaustively solve an endgame, it is also possible to exhaustively solve a pre-endgame. It of course becomes exponentially more difficult the more tiles are unseen.

We should establish a convention for pre-endgames. Consider the situation where there are 2 tiles in the bag, and we make a 1-tile move. We then draw 1 of 9 possible tiles. Our opponent could then solve the 1-in-the bag pre-endgame for 1 of 8 possible tiles.

Let's say opponent has a pre-endgame that wins almost all the time, but if our opponent solves that pre-endgame wrongly, and makes a suboptimal play, but draws the right tile needed to win, we should still mark that 1/9 possibility as a LOSS for us. The naive way would assume that our opponent would always make their best play, but we should always be pessimistic (this isn't advice for life, just for crossword board game pre-endgames).

The approach for pre-endgame changes depending on how many tiles are unseen.

### 1-in-the-bag

If there is 1 tile in the bag, there are up to eight different possibilities for what that 1 tile can be. The player on turn (heretofore referred to as we/us) therefore sees 8 different racks for their opponent.

We should first generate all of our possible plays. Then, for every play, we can draw each of the 8 tiles successively and solve all of the endgames from our opponent's perspective. Every play is guaranteed to use at least 1 tile (see exception below). Therefore, every endgame is known and we can have an accurate count of won/lost/drawn games for every move and every possible tile draw. We can then sort by win % or similar.

We should also evaluate the possibility of passing. If we pass, we can first examine the opponent passing back (which automatically ends the game, with our 2-pass simplification):
    - Give them all 8 possible racks and try passing back.
    - Track wins/losses/draws from it on a tally.

If the opponent doesn't pass back, then we must try every possible play from their perspective:

- For every combination of 7 tiles they could have (all 8 of them):
    - Generate all plays (except passes, which we just tried above).
    - For each generated play:
        - Keep a tally of wins, draws, and losses from our perspective for this play
        - Make the play
        - Solve the endgame from OUR perspective
        - If there is a loss for us, mark this combination as a loss for us and break early.
            - Note that this isn't a guaranteed loss, because the opponent doesn't have to make this play. But a potential loss should still count as a loss for the purposes of win % and sorting.
        - If there is a draw or win, continue.
    - If there is a draw, mark the game as a draw from our perspective, only if there is no loss.
    - Otherwise, mark this game as a win from our perspective.

It might be illuminating to step through a PASS example.

#### Example: Our first move is a Pass

Our rack is `<R>`, our spread relative to our opponent is `<S>`, and there is one tile in the bag. Unseen to us is `AACEISUY`, and we decide to pass.

First, we solve the easier "opp passes back" case, and this ends the game. We solve it for each of the 8 possible opponent racks (in this case there's actually 7, but `ACEISUY` happens twice).

Let's say 6 of those 8 possible games result in a win for us, 1 is a tie, and 1 is a loss. Let's pretend Y in the bag is a loss for us, and C in the bag is a tie.

Since Y in the bag is a loss for us if the opponent passes back, we don't need to consider any more Y-in-the-bag cases. Even if every single other Y-in-the-bag case when the opponent makes a non-passing play are all wins for us, and even though the opponent may miss his win and not pass back, the fact that it's possible for us to lose should still mark Y as a potential loss.

C in the bag is a tie for us if the opponent passes back. We still need to consider other cases, until we at least find a loss (and stop searching, marking C in the bag as a loss), or get to the very end of the search. If we're at the end of the search, and the worst we found for us is a tie, then we mark C in the bag as a tie.

AAEISU in the bag are wins for us if the opponent passes back. We still need to consider other cases as above, until we find a tie or loss. Only if we don't find a single tie or a loss can we be assured that AAEISU in the bag are guaranteed wins.


We can then generate all possible plays with `AACEISUY` from our opponent's perspective, that use AT MOST 7 of those 8 tiles. Note: we can take a shortcut here, and just consider all plays with `AACEISU`, since we already know we have a potential loss if the Y is in the bag, so there's no need to analyze that case. However, for the purposes of this example, let's forget about that temporarily and generate all plays with `AACEISUY`.

Let's imagine that we analyze the play `CAUSEY`. We make it so the opponent plays `CAUSEY` and then solve the endgame from our perspective. Our rack `<R>` is known and fixed (since we passed to start), so we know no matter what that after `CAUSEY` our opponent's rack is `AI`. So we can perfectly solve this endgame.

It happens that `CAUSEY` causeys us to lose. We then enumerate which racks the opponent could have had to play `CAUSEY`:

- AACESUY (I in the bag)
- ACEISUY (A in the bag)
- ACEISUY (A in the bag, again)

And racks the opponent could not have had:
- AAEISUY (C in the bag)
- AACEIUY (S in the bag)
- AACEISY (U in the bag)
- AACISUY (E in the bag)
- AACEISU (Y in the bag)

So, we know that I in the bag and A in the bag are losses for us if we decide to pass to start the endgame. This means we can stop analyzing those cases as well, and just mark them as losses.

We then analyze `SAUCY`. Before even starting analysis, we see that opponent could only make SAUCY if:

- I is in the bag
- A is in the bag (2X)
- E is in the bag

We already know that I and A in the bag are losses for us. We analyze SAUCY only to determine if E in the bag is also a loss for us. If it is, we mark that down so that we can avoid extra computation for other future plays.

We then analyze `YUCAS`. We can immediately skip that analysis, since it doesn't give us any new information about which letters in the bag lose.

And so on. So it is often possible to exit early.

### 2-in-the-bag

If there are 2 tiles in the bag, there are up to (9 * 8)/2 = 36 different possibilities for what those 2 tiles can be. We see 36 different racks for our opponent.

We proceed as above in generating all our possible plays. Then:

- For every play that uses 2 or more tiles, we draw each of the 36-tile combinations successively, and solve all of the endgames from our opponent's perspective.
- For every play that uses 1 tile, we draw each of the up to 9 possible tiles we can draw. (Loop for each of these 9)
    - Then, it's opp's turn. We assign them all 8 possible draws
    - For each of the 8 possible combinations opp can have:
        - Gen all plays (including pass)
        - For each generated play:
            - Keep a tally of wins/losses/draws etc (see above)
            - Make the play
            - Solve endgame from our perspective and update tally.
        - If opp passes:
            - Solve 1-in-the-bag pre-endgame for US, and check if any of our "best moves" make us draw/lose. Our best moves are the ones that have the most wins (draws are half a point). We assume we would never play a move that would be worse.
    - If any of these result in a loss for us, then that 1 out of 72 eventuality (order of tile draws matters) gets marked as a loss.
- For a pass, then:
    - It's opp's turn, We assign all 36 possible racks for them and calculate W/L/D if they pass back.
    - We generate all possible non-pass plays for them with their 36 racks.
        - Make the play from their perspective
        - Solve the 1-in-the bag pre-endgame for us if they don't empty the bag, or solve the endgame if they do. Use the "best moves" heuristic from above for pre-endgame.
            - 1-in-the-bag is tricky. You must check what happens if they draw 2 tiles in either order. Might be easier to do the loop of 9 and loop of 8.

### Speed improvement ideas

- We typically care about the winningest play.
    - We should sort plays initially by some metric: score or short sim performance
    - Let's say we examine a play and it wins 6/8 endgames
    - If we then examine a play and it's already lost 3 endgames, we don't need to examine that play any further since it's going to lose.
    - When we queue jobs we should have a shared structure with best performers so that we can determine inside the job whether we should quit early.


3/4/24 - Notes on first implementation of generic N-in-the-bag pre-endgame

## Rough pseudocode description, leaving out optimizations:

### Main thread:

- Initialize N independent endgame solvers, where N is the number of threads
- Start N job processors
- Generate all moves available for the player we're solving for
- For every play:
    - Queue up a job containing the play and the unseen tiles
- Wait until all processors are done

### Job processor for play M:
- If we have found more losses than any analyzed move's losses, then quit analyzing this move
- Generate bag-length permutations of all unseen tiles. (i.e. if you have 11 tiles unseen, then
   we create 11P4 permutations of length 4 -- these are all the 4-letter possible draws).
- For every permutation P:
    - If play M's found losses has more losses than any analyzed move's losses, then quit analyzing this move (I guess we check again)
    - Draw a rack for our opponent, leaving the permutation P in the bag to draw from
    - Call Solve(M, M, P)

#### Solve(M, M, P)

function Solve(M, moveToMake, P):
    if HasSomeLoss(M):
        # If M already has a loss, stop analyzing it
        return

    if bag.IsEmpty() or game.IsOver():
        if game.IsOver():
            finalSpread = game.SpreadFor(PlayerWeAreSolvingFor)
        else if bag.IsEmpty():
            finalSpread = endgameSolver.Solve(game)

        if M empties the bag:
            add win, draw, or loss to M for permutation P, depending on finalSpread
        else:
            add unfinalized win, draw, or loss to M for permutation P, depending on finalSpread

        return

    PlayMove(moveToMake)

    if not bag.IsEmpty() and not game.IsOver():
        generate all plays for player on turn
        for each of those plays nextPlay:
            Solve(M, nextPlay, P)
            if player on turn is the player we are solving for:
                if M.OutcomeFor(P) == WIN:
                    break

    else:
        # bag is empty, solve endgames
        Solve(M, nil, P)

    UnplayLastMove()

## Comments to read

https://www.cross-tables.com/annotated.php?u=37031

https://www.cross-tables.com/annotated.php?u=29275

https://www.cross-tables.com/annotated.php?u=37033

"I just asked for the analysis of 6C FEN in the case you pick NS.

It took 18 hours and it says that NS+JS is a guaranteed win, NS+SJ is a guaranteed draw and everything else is a possible loss (NS+AA, NS+AJ, NS+JA, etc.).

That's wrong, NS+JS and NS+SJ should both be possible losses.
If Noah passes with AADDIOR then Joshua's objectively best move is 15B SEALANT, which he should play, and then Noah wins the endgame."

## New pseudocode description

### Main thread:

- Initialize N independent endgame solvers, where N is the number of threads
- Start N job processors
- Generate all moves available for the player we're solving for
- For every play:
    - Queue up a job containing the play and the unseen tiles
- Wait until all processors are done

### Job processor for play M:
- If we have found more losses than any analyzed move's losses, then quit analyzing this move
- Call Solve(M, M)

#### Solve(M, M)

function Solve(M, moveToMake):
    if HasSomeLoss(M):
        # If M already has a loss, stop analyzing it
        return

    if bag.IsEmpty() or game.IsOver():
        if game.IsOver():
            finalSpread = game.SpreadFor(PlayerWeAreSolvingFor)
        else if bag.IsEmpty():
            finalSpread = endgameSolver.Solve(game)

        if M empties the bag:
            add win, draw, or loss to M for permutation P, depending on finalSpread
        else:
            add unfinalized win, draw, or loss to M for permutation P, depending on finalSpread

        return

    PlayMove(moveToMake)

    if not bag.IsEmpty() and not game.IsOver():
        generate all plays for player on turn
        for each of those plays nextPlay:
            Solve(M, nextPlay, P)
            if player on turn is the player we are solving for:
                if M.OutcomeFor(P) == WIN:
                    break

    else:
        # bag is empty, solve endgames
        Solve(M, nil, P)

    UnplayLastMove()