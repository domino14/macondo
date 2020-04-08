# Macondo

## What is Macondo?

Macondo will some day be a world-class Scrabble and generic Crossword Game AI.

But for now, it is in **pre-alpha**, so download and use at your own risk. It does no machine learning and its values are not as good as Quackle's yet. The interface is minimal. In short, it should just be used for **research**. Please expect things to break!

Quick tutorial video: [https://youtu.be/07Dpa-oTTFY](https://youtu.be/07Dpa-oTTFY)

## Features it has:

- Simple static move evaluation engine, using exhaustive leave values. Our values are not as good as Quackle's yet.
- Multi-core Monte Carlo simulation. Should be a bit faster than Quackle.
- An exhaustive endgame solver using minimax + alpha-beta pruning + iterative deepening. It should be able to solve every complex endgame. However, depending on the complexity, it can be extremely slow (think many hours if not days). We will work on speeding this up.

  - For the large majority of endgames, it should be able to find a solution in a few seconds.

- A very simple opening-move placement heuristic value
- GCG load and navigation
- Very fast import from cross-tables
- Automatic game playing (on my 2017 iMac it can play ~70 games per second against itself)
- Command-line driven with a simple shell

## Features it's missing that Quackle has:

- A proper GUI, and all that entails.
- Any lexica besides NWL18 (although adding support for other lexica will not be too difficult)
- Pre-endgame heuristic values
- 1 and 2 in the bag pre-endgame solver (1-PEG and 2-PEG)
- A "championship player"
- Builds for anything other than Macs

## Features we will add in the future:

- Heat maps
- Automatic inferencing
- Exact pre-endgame results (which tile draws win and how, etc)
- Graphs of score distributions per simmed play
- Machine learning! (taratantara!)
- And more!

## How to install

1. Navigate here: [https://github.com/domino14/macondo/releases](https://github.com/domino14/macondo/releases)

2. Download the latest file that looks like `macondo-darwin.tar.gz`. Right now it is **Mac-only**. If you are savvy with compilers, you can build your own Windows or Linux version with `go`. Untar it to your desired directory.

3. You will need a lexicon gaddag. The default lexicon right now is NWL18, but we will add more in the near future. You can build a gaddag using the make_gaddag executable found in the tarball.

   - Usage: `./make_gaddag -filename NWL18.txt`
   - The lexicon can be downloaded from NASPA if you're a NASPA member.
   - The command above will generate a file named out.gaddag
   - Move this file to your ./data/lexica/gaddag/NWL18.gaddag in your untarred directory

4. Try opening the `macondo` executable. Your Mac might complain, especially if you're using Catalina. You can open it anyway by going to your Settings -> Security & Privacy -> General and click "Allow anyway" next to macondo. I swear it's not a virus.

## How to use

1. Type in `help` for commands

2. Note that many commands are very primitive. I did not want to expand this shell interface too much as I expect the actual interface to be a GUI. As such there are things that you probably shouldn't do, like sim a position and generate other positions at the same time.

3. See this quick video tutorial. Note that this can quickly go out of date. I'll make another tutorial when a GUI is ready.

[https://youtu.be/07Dpa-oTTFY](https://youtu.be/07Dpa-oTTFY)
