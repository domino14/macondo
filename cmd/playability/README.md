# Word playability

How much is knowing a word worth? For any set of words S,

    playability(S) = expected equity HastyBot gives up per game
                     if S is removed from its lexicon and nothing else changes

At one turn the loss is a **regret**: E(best move) − E(best move that forms no word in S). "Forms" means the main word and every cross word, so hooks count. AXE gets credit when the best play hooks an E onto AX. The per-game value is the average regret per turn times the number of turns per game.

- **Word value:** S = {w}. STANE is worth 0.47 per 1000 games; it is the best play's word only 1,133 times in 10M games, and each time losing it costs about 4.2 points, the same as losing the whole AENST rack.
- **Alphagram value:** S = all anagrams of the alphagram. This is what a study list wants.

**Units:** HastyBot static equity points per game; `analyze.py` reports per 1000 games.

## Why filtering is exact
A move is legal under lexicon L∖S exactly when it is legal under L and forms no word in S. So one full move list, generated with the whole lexicon and filtered by `FormedWords`, gives the answer a pruned word graph would. Regret is nonzero only when the best move forms a word in S, so each turn only needs work for the 1–3 words the best move makes.

## Self-play (`gen`)
- HastyBot plays itself with the full move list each turn. A test in check mode confirmed the list's top move matches HastyBot's own top-play path: 0 mismatches.
- Each formed word's regret, and its alphagram's regret, is accumulated per word.
- Records the main-word share (`a_main`) and a split by bag phase (`a_ph0`..`a_ph3`: empty bag, 1–20, 21–50, 51+).
- `PLAY_BOT` (e.g. `NO_LEAVE_BOT`) swaps the bot, for diagnostics only.

## Why there is no re-deal estimator for long words
An earlier version also had a synthetic estimator: re-deal saved turns so the mover's rack holds (most of) a long word, weighted by the exact draw probability. It was removed on Oct 18 because:
- at 10M games self-play already covers 99.9% of 7-8 letter racks, 98.6% of 9s and 85% of 10s;
- for 9s it read only 0.3-0.4 of self-play, because 61% of 9-letter hits are hooks onto an 8 already on the board (AERATION→AERATIONS), which a re-dealt rack doesn't model;
- its error bars were unreliable for racks with few hits, so combining it moved some well-measured power-tile racks by thousands of places (BCEEKSZ #14,310 with it vs #10,716 without).

It is in the git history (commit c91e87f2 and earlier) if long words ever need it.

## Scoring (`score`, `analyze.py`)
1. `score` writes `words.tsv` and `alphas.tsv` for every word in the lexicon (combos, hooks, self-play columns) and `meta.tsv`.
2. `analyze.py` gives each self-play estimate a standard-error floor of one unresolved 20-point hit.
3. It then shrinks toward a prior in log space. The prior is fitted on well-measured alphagrams using:
   - length
   - log(combos), and log(combos) × length
   - anagram count
   - hooks
   - JQXZKS counts
   - vowel share
4. Alphagrams with no evidence (most of 11+) land on the prior, which leans on probability. `prior_share` in the output says how much of each value is prior.

## Pipeline: lexicon in, ranked study list out
```
cmd/playability/run-pipeline.sh CSW24          # from the main checkout; data in ~/data/playability/CSW24
cmd/playability/query.py ~/data/playability/CSW24/final/ranks.tsv 7 201 400
```
`final/ranks.tsv` has one row per alphagram: `len rank alphagram playability lo95 hi95 prior_share prob_rank n_best words`. Ranks are distinct and run 1..N within each length; ties go to the more probable rack, then alphabetically. Every step resumes. On a busy 16-core machine, 1000 shards (10M games) took 18 h.

## Running the steps by hand (resumable; run from the main checkout, which has ./data)
```
playability gen -lexicon NWL23 -out $D/gen -shards 1000 -games-per-shard 10000 -workers 3   # skips finished shards
cmd/playability/status.sh
playability score -lexicon NWL23 -gen $D/gen -out $D/score
python3 cmd/playability/analyze.py $D/score $D/final
```
Seeds are fixed per shard, so a re-run reproduces a missing shard exactly.

## Known limits
- **One turn, static equity.** It misses setups, board openness, defence and challenges. Two follow-ups:
  - sim a sample of high-regret turns
  - measure regret in FastMLBot's win value instead of equity, as a collaborator suggested (about 10× slower; deferred)
- **Marginal values.** Each value assumes every other word is known. A learner version would take regret against a reduced base lexicon, using the same filter.
- **Expert boards.** The board distribution is HastyBot's own.

## Pilot (20k games, NWL23)
- Turns per game: 22.9.
- **QI** is first at 4.5 pts/game: best on 10,246 turns, 5,654 of them as the main word. The next tier is the 2-letter E words (RE/ER, ET/TE, EN/NE, DE/ED), mostly as cross words, then ZA and JO.
- **AENST** (STANE family): best on 9 turns in 458k, 0.0015 pts/game. (The pilot suggested each word in it was about 0; the full run says otherwise, see above.)
- **EOLIPILE:** 1 self-play hit.

## Full run (10M games, NWL23, finished Oct 17)
- 229M turns, 22.92 turns/game, 0 mismatches against HastyBot's top play. Output: `~/data/playability/final/playability.tsv`.
- **Self-play coverage** (alphagrams seen at least once): 7s and 8s 99.9%, 9s 98.6%, 10s 84.8%, 11s 49%, 12s 29%, 13s 11%, 14s 2.5%, 15s 1%.
- **Top:** QI 4.48 pts/game, then the 2-letter E words, ZA, JO. 7s: AEINRST 51 per 1000 games; 8s: ADEIINOT 58; 9s: AEILNORST 3.5.
- **Examples:** EOLIPILE is #9460 by probability among 8s but #3392 by playability. AENST is worth 2.1 per 1000 games; STANE alone 0.47.
- Words 11+ that self-play never saw sit on the prior (prior_share near 1), which ranks them by probability.

## Measured vs prior, by length (NWL23, 10M games)
The result tables are not committed; regenerate them with the pipeline.

How much of each length is measured vs prior (rank correlation of playability with probability, median `prior_share`):

| len | rho vs probability | prior_share | never the top play |
|---|---|---|---|
| 7-8 | 0.94-0.95 | 0.01 | 0% |
| 9 | 0.79 | 0.09 | 1% |
| 10 | 0.87 | 0.42 | 15% |
| 11-12 | 0.95-0.98 | 0.72 | 51-71% |
| 13-15 | 0.995-1.000 | 0.72 | 89-99% |

9s and 10s are carried by self-play (the top 10s, e.g. RERADIATED, have hundreds of hits). From 13 up the ranks are essentially probability order.

## Comparison with a count-based list (Oct 17-18)
An outside list counts how often each word is the top play (exact ties split 1/N) and sums words into racks. Against its raw NWL23 counts (Spearman within length):
- Rack level 0.90-0.97 for 2-9 letters; within their top 1,000, 0.48-0.81 for 4-9.
- Most of the top-1,000 gap is in the raw counts, not the method: their NWL23 counts behave as if leaves mattered less than for HastyBot (TUITION ~1.0x our count, AEINRST 0.12x; their TWL18 counts were a flat 0.10x). The cause is unknown; the author reports a HastyBot-like eval. `PLAY_BOT=NO_LEAVE_BOT` in `gen` was used to test this.
- Method alone (our count vs our regret, same games): 0.94-0.99 overall, 0.68-0.89 in the top 1,000. Regret drops vowel dumps (AUDIO, OWIE) and 8+letter hook 9s, and raises power-tile racks (QUIZ, QWERTY).
- The regret gap per hit is not noise: split-half r 0.91-1.00 at 1M games for 2-8 letters.
