# Bot interactions

The bot can be thought of as a function from `GameHistory -> GameEvent` (in
practice it is from `BotRequest -> BotResponse` to take care of error handling,
network details etc, but the core function is to process a `GameHistory` and
return a single event to the server). From the server's point of view a player
vs bot game consists of a loop of the form:

```
while game is not over
  --b-> game-history
  <-b-- move-event
  game-history = game-history + move-event
  --p-> game-state
  <-p-- move-event
  game-history = game-history + move-event
```

Where `-->` means `send to`, `<--` means `receive from`, and `b` and `p` are
the bot and player respectively. The `move-event` needs to be validated by the
server before it can be applied to the game; for simplicity that part has been
left off.

The bot is assumed to be stateless; at every turn it is sent an entire game
history, including a current rack, which its AI engine can process however it
likes to figure out the best next move.

## Messages

The full list of messages is:

#### 1. Bot has a move to play

- Bot sends `TILE_PLACEMENT_MOVE`
- Server places the move on the board, and updates the game state
- Server draws for the bot and updates the game history with the move and the new rack

#### 2. Bot wants to exchange

- Bot sends `EXCHANGE`
- Server draws for the bot and updates the game history with the exchange and the new rack

#### 3. Bot wants to pass

- Bot sends `PASS`
- Server updates the game history with the pass and the new (same as current) rack

#### 4. Bot wants to challenge

- Bot sends `CHALLENGE`
- Server validates the player's move, and does one of

  - Valid move:
    - Update the game history with the challenge score adjustment
    - If DoubleChallenge, add a `PASS` for the bot's next turn and set player-on-turn back to the player

  - Invalid move:
    - Update the game history with the retracted word
    - Remove the word from the board
    - Add a `PASS` for the player to the game history
    - Set the player-on-turn back to the bot

Note that the bot never needs to **respond** to a challenge, since it always
gets a game history that ends with the bot as player-on-turn and a rack to play
with. If the player challenges the bot, the server handles all the events in
the game history until it's time for the bot to play again.

## Error handling

If the player makes an invalid move, the server will send an error message to
the front end, which can display the appropriate things to the user and ask for
the move to be redone. It is less clear what to do if the bot makes an
erroneous move; since the bot is a function from `Game History -> Move Event`,
sending it the same game history again will presumably generate the same
(invalid) move. The best course would likely be to have the bot forfeit the
game, and return the validation error back to the bot for its logs, and some
sort of "bot was badly programmed; ending the game" message to the player's
front end.

This move validation (or lack thereof) is the key difference between a bot and
an interactive player client.

## Networking

Note that both the `BotRequest` and `BotResponse` are protocol buffers sent
over a `NATS` message queue; while this particular bot happens to share a lot
of board and game representation code with the `liwords` server it should be
possible to write a bot in whatever language you choose, using the same
messages for communication.
