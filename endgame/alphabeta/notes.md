    Optimization notes for the alpha beta minimax algorithm.

TestSolveStandard2 (vs Joel)

- Number of expanded nodes: 59081 (when using old, simple EndgameHeuristic)
  ~0.73s roughly on MBP.
  ~0.50s roughly on iMac.

TestSolveOther3 (vs Joey)

- Number of expanded nodes: 14557067 (when using old, simple EndgameHeuristic)
  ~156 - 192 s roughly on MBP (big variation here :/ )

---

No sorting at all:

TestSolveStandard2:

- 182449 expanded nodes, ~3 seconds

---

SheppardSort:

TestSolveStandard2:

- Number of expanded nodes: 61501
  Slightly slower (~0.71s on iMac)

TestSolveOther3 (vs Joey)

- Number of expanded nodes: 2088170 (7x fewer!!!)
  ~37 seconds roughly on iMac

TestSolveComplex has more nodes (407080 vs 272205 and is more than twice as slow)

TestSolveOther2 has about the same number of nodes (11657284 vs 11740313) and is also more than twice as slow (about 2.2x) .. :(
