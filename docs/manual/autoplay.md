# Autoplay

- [Back to Manual](/macondo/manual)
- [Back to Main Page](/macondo)

## What does autoplay do?

Autoplay pits two Macondo bots against each other for many games. In
its basic use case, without any arguments, `autoplay` will use two
"exhaustive leave" players - i.e. computer players that use 1-to-6 tile
leave values to calculate equity.

The leave values can be found in `./data/strategy/default_english/leaves.idx`.

See [make_leaves_structure](/macondo/manual/make_leaves_structure.html) for how
this file was created.

## Options

`-logfile foo.txt` will log games to the given file. Note that another file will be created, `games-foo.txt` in this case, with very basic metadata
about each game (the final score and who went first). `foo.txt` will contain more-in depth turn-per-turn data.

`-lexicon CSW19` uses the CSW19 lexicon, for example

`-letterdistribution norwegian` uses the norwegian letter distribution, for example

`-leavefile1 filename.idx` sets the first bot's leavefile to `filename.idx`. Note that the `filename.idx` must be located inside the `./data/strategy/<lexicon>` directory in order to be found.

`-leavefile2 filename.idx` sets the second bot's leavefile to `filename.idx`.

## Bot types

Right now only two bots are supported: `exhaustiveleave` and `noleave`.
The `noleave` bot uses no leave values, so it effectively becomes a greedy bot (it maximizes score every turn).

## Starting and stopping

Using any `autoplay` command will start the games. Every 1000 games, a log will be printed to screen.

To STOP autoplay cleanly, type in `autoplay stop`. It will take a couple of seconds.

## Analyzing log files

A very simple analyzer can be accessed with the `autoanalyze` command. This command's only argument is the `games-foo.txt` file that was created with the `-logfile` option.

## Examples

### Player1 is an exhaustive-leave bot, Player2 is a greedy bot:

`autoplay exhaustiveleave noleave`

Leave it running for a bit, then do an `autoplay stop` and then an `autoanalyze /tmp/games-autoplay.txt`:

```
Games played: 20111
exhaustiveleave-1 wins: 12967.0 (64.477%)
exhaustiveleave-1 went first: 10169.0 (50.564%)
Player who went first wins: 10971.0 (54.552%)
exhaustiveleave-1 Mean Score: 434.577545  Stdev: 59.587155
noleave-2 Mean Score: 399.649893  Stdev: 58.249322
```

The stats above show that a bot that uses leave values wins nearly 2/3 of its games against a greedy bot.

### Both players are exhaustive-leave bots:

`autoplay` (this is the default, and uses default Macondo values)

### Player1 uses a special set of leave values, Player2 uses the default set:

`autoplay exhaustiveleave exhaustiveleave -leavefile1 quackleleaves.idx`

**Note:** The file `quackleleaves.idx` in this case must be in your `./data/strategy/<lexicon>/` directory. You can put it in the special `default_english` lexicon to make it apply to both NWL18 and CSW19 games.

Analysis:

```
Games played: 84648
exhaustiveleave-1 wins: 42923.0 (50.708%)
exhaustiveleave-1 went first: 42425.0 (50.119%)
Player who went first wins: 47021.0 (55.549%)
exhaustiveleave-1 Mean Score: 435.499787  Stdev: 58.757773
exhaustiveleave-2 Mean Score: 434.291584  Stdev: 59.496489
```

This analysis shows that the bot that used Quackle leaves won around 50.7% of its games against Macondo. A binomial calculator shows that the chance
that this is a fluke is around 0.00002.

Note: Macondo values can now beat Quackle values around 50.6 or so % of the time, after fine-tuning our values. You can see the `/notebooks` directory of this repo for more details.

We don't have a simming player working yet, but I think the difference might be even bigger once simulation is involved.

### Different bot levels

`autoplay -botcode1 LEVEL4_PROBABILISTIC -botcode2 HASTY_BOT -numgames 20000`

This plays 20000 games between a Level 4 probabilistic bot and a HastyBot. "Probabilistic" bots miss more plays than HastyBot randomly; the weaker, the more plays it misses.

You can see `macondo.proto` in the source for a list of the different bots.

`autoplay -lexicon FRA20 -letterdistribution french -botcode1 LEVEL1_PROBABILISTIC -botcode2 HASTY_BOT`

In order to specify a non-English lexicon, you must also specify the letter distribution, as in the example above.
