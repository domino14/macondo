# Manual

- [Back to Manual](/macondo/manual)
- [Back to Main Page](/macondo)

## make_leaves_structure

You must first create a CSV that contains every possible 1 to 6 tile
leave value in your lexicon. e.g:

```
A,2
B,-1
..
..
ABCD?,15
..
..
```

- Usage: `./make_leaves_structure -filename superleaves.csv`
- This will create a file named `data.idx`. Copy this file to the
  `./data/strategy/<lexicon>/` directory for your desired lexicon. If you
  wish it to be loaded by default, you must rename it to
  `leaves.idx`. Otherwise, it can be used in experiments with its original
  name. See the [Autoplay](/macondo/manual/autoplay.html) documentation.
