---
id: reading-data
priority: 30
when: always
---
**Reading the data.** Candidates are sorted by win%, decreasing, so the first
one is the play being recommended. A figure like `25.5±3.20` is a mean of 25.5
with a 99% confidence interval of ±3.20; when two plays' intervals overlap
heavily, the difference between them is not established. A play's score already
includes the bingo bonus if it was a bingo.

The "next two plies" table is what the simulation saw happen after each
candidate: ply 1 is the opponent's reply and ply 2 is our own next turn. Mean
and Stdev are the mean score and its standard deviation for that ply; Bingo% is
how often that player bingoed on that turn.

The follow-up tables sample what each side actually played. Most rows carry a
low percentage. A row that stands out well above the rest is the interesting
one.

Anything stated in a **verdict** section has already been checked against the
board by the engine. Trust it over your own reading of the tables, and over
your own arithmetic.
