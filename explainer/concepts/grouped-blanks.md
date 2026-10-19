---
id: grouped-blanks
priority: 120
when: has_grouped_followup
---
**Plays that can be made more than one way.** Some plays can be made in more
than one way, because the blank can stand in for different tiles. Those appear
as one grouped row followed by indented `-` lines giving each individual way:

```
 1H (Z)WIEBACK      {B|C|?}       116-134  165      6.72
  -  1H (Z)WIEBAcK  {B}           134      70       2.85
  -  1H (Z)WIEbACK  {C}           125      68       2.77
  -  1H (Z)WIEbAcK  {?}           116      27       1.10
```

The grouped row's "% of time" is the chance of making that play **by any
route**. That is the number to quote and the number to judge a setup by - never
one indented line's percentage, which is only a share of it. The score column
shows the range across the routes.

In the Needed Draw column, `|` means "or", separating whole alternative draws.
`{B|C|?}` means drawing the B, *or* the C, *or* the second blank each let you
make the play; `{EI|E?|I?}` would mean drawing both E and I, or E plus the
second blank, or I plus the second blank. A `-` alternative is a route that
needs no draw at all, and a trailing `...` just means the list was too wide for
the column - every alternative is spelled out on the indented lines.

The grouped row is written in capitals because no single notation is correct
for all of its routes. If you want to name one specific route, use its exact
indented play string, with the lowercase letter showing where the blank goes.
