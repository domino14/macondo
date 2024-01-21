# Macondo

## What is Macondo?

Macondo will some day be a world-class Crossword Game AI.

But for now, it is in **beta**, so download and use at your own risk. It does no machine learning and it doesn't have all of Quackle's parameters yet. The interface is minimal. In short, it should just be used for **research**. Please expect things to break!

Tutorial video: [https://youtu.be/MOyzyKskr_4](https://youtu.be/MOyzyKskr_4)

- [The Macondo team](/macondo/team.html)
- [Change Log](/macondo/changelog.html)
- [CGP File Format](https://github.com/woogles-io/open-protocols/tree/master/cgp#readme)
- [Manual / Documentation](/macondo/manual)
- [How it works](/macondo/howitworks.html)
- [Why the name Macondo?](/macondo/name.html)
- [Acknowledgements](/macondo/acknowledgements.html)

## How to install

1. Navigate here: [https://github.com/domino14/macondo/releases](https://github.com/domino14/macondo/releases)

2. Download the latest file for your operating system.

3. Try opening the `macondo` executable. If you are using Mac OS, your system might complain. You can open it anyway by going to your Settings -> Security & Privacy -> General and click "Allow anyway" next to macondo. I swear it's not a virus.

## Features it has:

- Simple static move evaluation engine, using exhaustive leave values. Values
  are slightly better than Quackle's.
- Multi-core Monte Carlo simulation. Should be a bit faster than Quackle.
- An exhaustive endgame solver using minimax + alpha-beta pruning + iterative deepening + multithreading. It should be able to solve every complex endgame. However, depending on the complexity, it can be slow (think several hours). We will work on speeding this up.

  - For the large majority of endgames, it should be able to find a solution in a few seconds.

- A fully exhaustive 1-in-the-bag pre-endgame solver
- A very simple opening-move placement heuristic value
- GCG load and navigation
- Very fast import from cross-tables, [Woogles](https://woogles.io), and GCG files
- Automatic game playing (on my 2017 iMac it can play ~70 games per second against itself)
- Command-line driven with a simple shell

## Features it's missing that Quackle has:

- A proper GUI, and all that entails.
- Pre-endgame heuristic values for simulations (in our experiments, though, we've found that Quackle's values did not really provide a benefit)
- 2 in the bag pre-endgame solver (Quackle's is not fully exhaustive but still decent)

## Features we will add in the future:

- Heat maps
- Graphs of score distributions per simmed play
- Machine learning! (taratantara!)
- And more!

## How to use

1. Open the executable. If you wish your default lexicon to be CSW21, you
   must set the environment variable `DEFAULT_LEXICON` to `CSW21`. Otherwise, it defaults to NWL20. In this case you can set your lexicon by typing in `set lexicon CSW21` into the macondo prompt.

2. Move some `*.kwg` files for your desired lexicon to the `./data/lexica/gaddag` folder. You can find kwg files at [https://github.com/woogles-io/liwords/blob/master/liwords-ui/public/wasm](https://github.com/woogles-io/liwords/blob/master/liwords-ui/public/wasm)

3. Type in `help` for commands

4. Note that many commands are very primitive. I did not want to expand this shell interface too much as I expect the actual interface to be a GUI. As such there are things that you probably shouldn't do, like sim a position and generate other positions at the same time.

5. See this quick video tutorial. Note that this can quickly go out of date. I'll make another tutorial when a GUI is ready. [https://youtu.be/MOyzyKskr_4](https://youtu.be/MOyzyKskr_4)

6. See also this video that demonstrates how to use the "infer" command in depth:

[https://www.youtube.com/watch?v=oa4gXVVvfyc](https://www.youtube.com/watch?v=oa4gXVVvfyc)