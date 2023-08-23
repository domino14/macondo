# Manual

- [Back to Manual](/macondo/manual)
- [Back to Main Page](/macondo)

## How to create a leaves structure

You must first create a CSV that contains every possible 1 to 6 tile
leave value in your lexicon. e.g:

```csv
?,30
A,2
B,-1
..
..
?ABCD,15
..
..


```

Then clone the [wolges](https://github.com/andy-k/wolges) repo and install Rust on your computer. Andy has created a small, fast leaves file structure, but it requires a different repo and language to create. If anyone wants to help translate it to Go, please make a PR!

You can then follow the instructions in the wolges README to make a klv2 file. For example:

```
cargo run --release --bin buildlex -- english-klv2 leaves.csv leaves.klv2
```

- This will create a file named `leaves.klv2`. Copy this file to the
  `./data/strategy/<lexicon>/` directory for your desired lexicon. If you
  wish it to be loaded by default, you must rename it to
  `leaves.klv2`. Otherwise, it can be used in experiments with its original
  name. See the [Autoplay](/macondo/manual/autoplay.html) documentation.

## How to create a GADDAG file

A GADDAG is a data structure that makes move generation very fast. Macondo needs GADDAGs in order to generate moves for a given lexicon.

You first need a text file for your lexicon. See the wolges README instructions for how to make a .kwg file (this is a Kurnia Word Graph, which encodes a GADDAG). For example:

```
cargo run --release --bin buildlex -- english-kwg CSW21.txt CSW21.kwg
```

You can then move this kwg to the appropriate `data` sub-directory.

You can download some gaddags from the liwords repo; see https://github.com/domino14/liwords/blob/master/liwords-ui/public/wasm.

