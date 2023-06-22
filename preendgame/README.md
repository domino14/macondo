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
        - If there is a draw or win, continue.
    - If there is a draw, mark the game as a draw from our perspective, only if there is no loss.
    - Otherwise, mark this game as a win from our perspective.

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
