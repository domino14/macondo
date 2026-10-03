# The 54M NWL23 games behind nwl23 / nwl23s / nwl23st

Everything the 57% model learned from is these games. They are kept, not
archived here (32 GB gzipped): `~/data/nwl23{a,b,c}.txt.gz` (turn logs, one
row per move) and `~/data/games-nwl23{a,b,c}.txt` (one row per game).
Checksums in `DATA-MD5SUMS`. Their autoplay configs and logs are in `logs/`.

## How they were generated

Three runs of Macondo's autoplay, on this machine, with the shell built from
commit `9e66225d` (2026-09-24, "Quick endgame: use the (-1, 1) root window")
for half a, and from `52dd40a2` (2026-09-25, the Zobrist rack-count fix that
ended half a) for halves b and c. The generation code has not changed since
in any way that affects these bots (verified 2026-09-27: 600 seeded
HastyBot-vs-HastyBot games are byte-identical under `0986c24f` and `9bfd80f3`).

    bin/shell autoplay -botcode1 RANDOM_BOT_WITH_TEMPERATURE -botcode2 HASTY_BOT \
      -lexicon NWL23 -quickendgame 2 -quickendgamemargin 60 -quickendgamecap 250 \
      -numgames <N> -threads 16 -block true -experimentid nwl23<half>

(`pytorch/run-fresh.sh` at those commits, `LEXICON=NWL23`.) Saved configs
`logs/nwl23{a,b,c}.config.json`; a `player2` with no botCode is HASTY_BOT.
No master seed: every game drew its tiles from the unseeded RNG, so the
exact games are reproducible only from the kept logs, not by rerunning.

| half | started | games | note |
|---|---|---|---|
| a | 2026-09-24 11:48 | 19,298,831 | asked for 27M; crashed at 19.3M on a seven-of-a-kind rack (Zobrist table size), fixed in `52dd40a2` |
| b | 2026-09-25 00:34 | 27,000,000 | |
| c | 2026-09-25 18:51 | 7,701,169 | top-up to 54M total |

Total 54,000,000 games. About 435 games/s on 16 threads.

## The two bots

- `RANDOM_BOT_WITH_TEMPERATURE` (seat 1 in even games, seat 2 in odd ones,
  the runner alternates): generates the top 50 plays by HastyBot static
  equity (leave values from the NWL23 KLV, opening/pre-endgame/endgame
  adjustments) and, while the bag holds more than 60 tiles, samples one from
  softmax(equity / 1.0 point); with 60 or fewer it plays the top play. Hard-
  coded in `ai/bot/bot_player.go` at those commits (top 50, T=1.0, bag > 60).
  This is the label-contaminating diversifier that the sampled-openings
  batch (`open`) replaces.
- `HASTY_BOT`: the top play by static equity.
- Both: when the bag is empty and |spread| <= 60, moves come from a quick
  2-ply negamax with the (-1, 1) root window (result-only), greedy playout
  at the leaves, 250 ms cap per move; otherwise the static play.

Lexicon NWL23, English letter distribution, standard board, classic
variant; no game pairs, no handicap, no random openings.

## From games to training rows

`bin/mlproducer` (commit `3449d13f`) replays each game; half of all games are
replayed transposed (`shouldTranspose`, by game-ID hash). With
`-labeler result -per-game -endgame-plies 2` it draws one turn per game
uniformly from 1..30, skips draws that land on an empty bag, and emits the
post-move position (board after the move, mover's leave, opponent's rack
back in the pool) with labels: mover's final result (WDL), spread to the
end, the opponent's next-move bingo flag and score, and the four placement
planes. Draws are random and unseeded, so a rescan picks different turns;
the 36,400,365-row cache used by nwl23s is not archived (102 GB) but any
rescan is the same distribution. (`nwl23`, the 53.13% comparison, was a
separate scan of the same games with the same flags before the planes
existed.)
