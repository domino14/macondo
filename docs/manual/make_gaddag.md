# Manual

- [Back to Manual](/macondo/manual)
- [Back to Main Page](/macondo)

## make_gaddag

- Usage: `./make_gaddag -filename NWL18.txt`
- The command above will generate a file named out.gaddag
- Move this file to your `./data/lexica/gaddag/NWL18.gaddag` in your Macondo download.

You can replace NWL18 with another desired lexicon.

If you wish to use other lexica, you will need to also
change the environment variable `DEFAULT_LETTER_DISTRIBUTION` to other values: `spanish`, `polish`, `german`, `norwegian`, `french`. There will be more in the future.

Then, you can change the `DEFAULT_LEXICON` environment variable to your
desired lexicon (make sure to use the same capitalization schema). For
example if your gaddag file is named OSPS42.gaddag, the name of the lexicon
is `OSPS42`.

Alternatively, you can just use `set lexicon OSPS42 polish` inside the Macondo shell, for example.
