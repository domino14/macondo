---
id: winpct
priority: 20
when: always
---
**Win percentage.** Win% can only be estimated by simulation. We run a
truncated Monte Carlo sim: we look a fixed number of plies ahead rather than to
the end of the game, then look the resulting score difference and bag count up
in a table to get an estimated win%, and average that over every iteration.

Win% is the metric that decides which play is best - more important than
equity. Do not explain that a play is best because it has the highest win%.
Explain *why* it has the highest win%.
