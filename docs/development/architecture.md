This software should support several use cases:

### Annotation

Allow for the annotation of games from other sources (live games or ISC, etc). We should allow GCG import as well as export. Allow users to specify partial or unknown racks. Allow generation of plays, simming, endgame/preendgame, and more.

### Play against computer

Allow for playing against the computer. Self-explanatory. The computer should have different players available (simming player, static player).

For a use case where this software connects online, it should have an API where it is given the opponent's last play, and its current rack, and it should submit its best play after some time. Perhaps the time limit can be given as well.

### Serve as backend for an online game

The software should be importable as a backend for a Crossword Game engine. For example, for calculating scores, dealing out tiles, adding bonuses, etc. Keeping track of a game and saving/loading it into a database.

---

Annotation and play against computer are a bit at odds with each other because the latter starts an interactive game. Annotation should probably be treated mostly as if it was an actual game, so that the annotator doesn't have to do things like, e.g. add the end-rack bonus.
