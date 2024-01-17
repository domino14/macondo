# Manual

- [Back to Manual](/macondo/manual)
- [Back to Main Page](/macondo)

# Tips 

## How to use other languages

NWL20 is currently the default lexicon of Macondo.

If you wish to use other lexica, you will need to also
change the environment variable `DEFAULT_LETTER_DISTRIBUTION` to other values: `spanish`, `polish`, `german`, `norwegian`, `french`, `catalan`, `polish`. There will be more in the future.

Then, you can change the `DEFAULT_LEXICON` environment variable to your
desired lexicon (make sure to use the same capitalization schema). For
example if your gaddag file is named OSPS42.gaddag, the name of the lexicon
is `OSPS42`.

Alternatively, you can just use `set lexicon OSPS42 polish` inside the Macondo shell, for example.
