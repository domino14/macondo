Optimization notes for the alpha beta minimax algorithm.

TestSolveStandard2 (vs Joel)
- Number of expanded nodes: 59081  (when using old, simple EndgameHeuristic)
~0.73s roughly on MBP.

TestSolveOther3 (vs Joey)
- Number of expanded nodes: 14557067 (when using old, simple EndgameHeuristic)
~156 - 192 s roughly on MBP (big variation here :/ )


-----

No sorting at all:

TestSolveStandard2:
- 182449 expanded nodes, ~3 seconds


-----

SheppardSort:

TestSolveStandard2:
- Number of expanded nodes: 61909
About 4 times slower...