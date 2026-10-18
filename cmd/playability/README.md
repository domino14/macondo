# Word playability

How much is knowing a word worth? For any set of words S,

    playability(S) = expected equity HastyBot gives up per game
                     if S is removed from its lexicon and nothing else changes

At one turn the loss is a **regret**: E(best move) − E(best move that forms no word in S). "Forms" means the main word and every cross word, so hooks count. AXE gets credit when the best play hooks an E onto AX. The per-game value is the average regret per turn times the number of turns per game.

- **Word value:** S = {w}. STANE is worth almost nothing, because ANTES, NATES, ETNAS or NEATS usually plays in the same spot for the same score.
- **Alphagram value:** S = all anagrams of the alphagram. This is what a study list wants.

**Units:** HastyBot static equity points per game; `analyze.py` reports per 1000 games.

## Why filtering is exact
A move is legal under lexicon L∖S exactly when it is legal under L and forms no word in S. So one full move list, generated with the whole lexicon and filtered by `FormedWords`, gives the answer a pruned word graph would. Regret is nonzero only when the best move forms a word in S, so each turn only needs work for the 1–3 words the best move makes.

## Estimator A: self-play (`gen`)
- HastyBot plays itself with the full move list each turn. A test in check mode confirmed the list's top move matches HastyBot's own top-play path: 0 mismatches.
- Each formed word's regret, and its alphagram's regret, is accumulated per word.
- Records the main-word share (`a_main`) and a split by bag phase (`a_ph0`..`a_ph3`: empty bag, 1–20, 21–50, 51+).
- About 0.2% of turns are saved as positions for estimator B.

## Estimator B: conditioned re-deals (`synth`)
Self-play almost never produces most 8–15 letter words. B estimates the same per-turn expectation by importance sampling over saved turns.
- **Positions:** a saved turn keeps the mover's leave from its previous move. Only the draw on top of that leave is re-dealt, out of the mover's unseen pool (everything not on the board and not in its leave).
- **Strata** are disjoint bands of "the rack covers c of the word's letters":
  - stratum 0 is the full word (c = min(len, 7))
  - stratum j is exactly j short, e.g. RODENTS through a board O
- **Exact probability:** grouping tiles into word letters, blank and other makes the draw probability `p` exactly computable (`playability.NewCondDraw`).
- **Estimate per stratum:** mean(p) over M random saved turns, times mean(regret) over K turns drawn from those M in proportion to p. Each of the K turns is re-dealt conditionally on the band.
- **Move generation:** top-10 with shadow pruning, falling back to the full list only if no top-10 move avoids the word. The pilot gave identical results with this.

## Combining (`score`, `analyze.py`)
**Since Oct 17, B is not used:** the pipeline and `analyze.py` take self-play only. At 10M games self-play covers 99.9% of 7-8 letter racks, and B's error bars were unreliable for racks with few B hits, so combining moved some power-tile racks by thousands of places (BCEEKSZ #14,310 with B vs #10,716 without). The steps below describe the old combination.

1. `score` writes `words.tsv` and `alphas.tsv` for every word in the lexicon (combos, hooks, A and B columns) and `meta.tsv`.
2. `analyze.py` combines A and B by inverse variance. A's standard error gets a floor of one unresolved 20-point hit.
3. It then shrinks toward a prior in log space. The prior is fitted on well-measured alphagrams using:
   - length
   - log(combos), and log(combos) × length
   - anagram count
   - hooks
   - JQXZKS counts
   - vowel share
4. Alphagrams with no evidence (most of 9+) land on the prior, which leans on probability. `prior_share` in the output says how much of each value is prior.

## Pipeline: lexicon in, ranked study list out
```
cmd/playability/run-pipeline.sh CSW24          # from the main checkout; data in ~/data/playability/CSW24
cmd/playability/query.py ~/data/playability/CSW24/final/ranks.tsv 7 201 400
```
`final/ranks.tsv` has one row per alphagram: `len rank alphagram playability lo95 hi95 prior_share prob_rank n_best words`. Ranks are distinct and run 1..N within each length; ties go to the more probable rack, then alphabetically. Every step resumes. On a busy 16-core machine, 1000 shards (10M games) took 18 h.

## Running the steps by hand (resumable; run from the main checkout, which has ./data)
```
cmd/playability/run-gen.sh 3 1000   # 1000 shards x 10k games; skips finished shards
cmd/playability/run-synth.sh 2      # waits for shards 0..59 (positions), then B; skips finished chunks
cmd/playability/status.sh
playability score -gen $D/gen -synth $D/synth78,$D/synth9p -out $D/score
python3 cmd/playability/analyze.py $D/score $D/final
```
Seeds are fixed per shard and per chunk, so a re-run reproduces a missing shard exactly.

## Known limits
- **One turn, static equity.** It misses setups, board openness, defence and challenges. Two follow-ups:
  - sim a sample of high-regret turns
  - measure regret in FastMLBot's win value instead of equity, as a collaborator suggested (about 10× slower; deferred)
- **Marginal values.** Each value assumes every other word is known. A learner version would take regret against a reduced base lexicon, using the same filter.
- **Expert boards.** The board distribution is HastyBot's own.

## Pilot (20k games, NWL23)
- Turns per game: 22.9.
- **QI** is first at 4.5 pts/game: best on 10,246 turns, 5,654 of them as the main word. The next tier is the 2-letter E words (RE/ER, ET/TE, EN/NE, DE/ED), mostly as cross words, then ZA and JO.
- **AENST** (STANE family): best on 9 turns in 458k, 0.0015 pts/game. Each word in it is about 0.
- **EOLIPILE:** 1 self-play hit; B gives 0.003–0.004 pts/game.
- **A vs B:** on the high-probability 7s, B agrees with A within about 2 SE. The 8s varied between runs because the pilot only had 4.6k saved turns.

## Full run (10M games, NWL23, finished Oct 17)
- 229M turns, 22.92 turns/game, 0 mismatches against HastyBot's top play. Output: `~/data/playability/final/playability.tsv`.
- **Self-play coverage** (alphagrams seen at least once): 7s and 8s 99.9%, 9s 98.6%, 10s 84.8%, 11s 49%, 12s 29%, 13s 11%, 14s 2.5%, 15s 1%.
- **B vs A:** 7s 0.93, 8s 0.93-0.97 on well-sampled alphagrams, so B is scaled by 1.078 and 1.041. For 9s, B reads only 0.3-0.4 of A, because 61% of 9-letter hits are hooks onto an 8 on the board (AERATION→AERATIONS), which B doesn't model. B is therefore not used for 9+.
- **Top:** QI 4.48 pts/game, then the 2-letter E words, ZA, JO. 7s: AEINRST 51 per 1000 games; 8s: ADEIINOT 58; 9s: AEILNORST 3.5.
- **Examples:** EOLIPILE is #9460 by probability among 8s but #3392 by playability. AENST is worth 2.1 per 1000 games; STANE alone 0.47.
- Words 11+ that self-play never saw sit on the prior (prior_share near 1), which ranks them by probability.
