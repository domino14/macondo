# Calibrating the inference softmax temperature (tau) from real games

## Background

The rangefinder (`rangefinder/`) infers an opponent's kept tiles after they make a
play. For each candidate rack consistent with the played tiles, it runs a shallow
sim (SimpleSimmer: 2 plies, 200 iterations, top-10 static plays plus the observed
play) and weighs the rack by the likelihood that a rational player holding it would
have chosen the observed play:

```
P(play | rack) = softmax( logit(win prob) / tau )
```

over the shallow sim's candidate set, where `logit(p) = ln(p/(1-p))` converts win
probabilities to log-odds (undoing the implicit sigmoid so softmax gets the
unbounded inputs it is designed for; see `rangefinder.Logit` and
`rangefinder.SoftmaxOverLogOdds`).

The temperature **tau** is on the log-odds scale and controls how strictly we
assume the observed play is the best available one. Small tau → the player is
assumed to (nearly) always pick the shallow sim's top choice; large tau → all
candidates become similarly likely, and the observed play carries little
information about the rack.

## Why fit tau on BestBot's own moves

Tau is *not* purely a model of player skill — it also absorbs the mismatch between
the evaluator used inside inference and the true decision process of the player
being modeled. BestBot chooses its moves with a **5-ply** sim, while the inference
mini-sim is only **2-ply** (it has to run hundreds of times per inference, so it
must stay cheap). Even for a perfectly rational player, the 2-ply evaluation
disagrees with the 5-ply one on real positions, and tau must be wide enough to
account for that approximation noise. Fitting tau against BestBot's actual choices
directly measures exactly this: *given the shallow sim's view of a position, how
peaked is the distribution of what BestBot really plays?*

## Maximum likelihood estimation

The key observation is that in a **finished** game the mover's rack is known at
every turn (Woogles GCGs record the full rack on each event line), so no rack
inference is needed to evaluate the likelihood model itself:

1. Replay the game to just before a BestBot turn. The position, BestBot's rack,
   and the move actually chosen are all known.
2. Run the exact shallow sim the rangefinder uses (`SimpleSimmer.GenAndSim`,
   2 plies, 200 iterations, top-10 static candidates **plus the played move** —
   the `addedMove` argument guarantees the chosen move is always in the candidate
   set even when static generation wouldn't rank it top-10).
3. Record the **log-odds vector** of the candidates' win probabilities and the
   index of the move actually played.

Given N such positions, the log-likelihood of tau is

```
LL(tau) = Σᵢ log softmax( logits⁽ⁱ⁾ / tau )[targetᵢ]
```

The per-position logits are tau-independent, so they are extracted once
(expensive: one mini-sim per position) and the 1-D search over tau is instant.
There is no closed form for a softmax-temperature MLE, so we evaluate LL on a
log-spaced grid, refine the maximum with a parabolic fit in log(tau), and report
a ~95% profile-likelihood confidence interval (ΔLL ≤ 1.92, Wilks).

### Position filters

A position qualifies only where the deployed inference (and BestBot's Monte Carlo
decision-making) actually operates:

- **≥ 2 tiles in the bag** pre-move. BestBot uses Monte Carlo only there: with 1
  tile in the bag it switches to the pre-endgame solver and with 0 to the endgame
  solver, so those choices are not softmax-of-sim shaped at all.
- **Tile placements of 1–6 tiles** by default. Seven-tile plays give inference no
  information (`PrepareFinder` returns `ErrNoInformation` — there is no leave to
  infer), and exchanges go through a different likelihood path
  (`inferSingleExchange`); both are recorded and can be included with flags.
- Plays challenged off the board are skipped (the recorded move became a pass).

## Pipeline (`cmd/taufit`)

```
taufit -mode fetch   -max-games 2000        # metadata + GCGs → taufit-data/
taufit -mode extract -workers 14 -max-games 2000   # shallow sims → ingredients.jsonl
taufit -mode fit     -curve-csv curve.csv   # grid-search MLE + breakdowns
```

- **fetch** paginates `game_service.GameMetadataService/GetRecentGames` for
  BestBot on woogles.io (unauthenticated), keeping NATIVE, classic-variant,
  `SIMMING_BOT` games in a single lexicon (default **NWL23** — tau should be
  refit per lexicon), and downloads each GCG. Resumable/append-only.
- **extract** replays each game (`game.NewFromHistory` positions the game before
  event *i* with the true rack from the event), runs the shallow sim at each
  qualifying BestBot turn, and emits one JSONL record per position with the
  log-odds vector, target index, and context fields (tiles in bag, tiles played,
  opponent rating, …). `-player human` exists for later fitting of a human tau.
- **fit** applies the filters, batch-evaluates LL(tau) across the grid, and
  reports the MLE, CI, comparisons against the current default and a
  uniform-choice baseline, per-bucket MLEs (tiles in bag, tiles played, opponent
  rating), and the rank histogram of the played move within the shallow sim.

The likelihood math is shared with the engine (`rangefinder.SoftmaxOverLogOdds`,
`rangefinder.Logit`, `rangefinder.MovesAreTheSame`), so the fitted tau plugs
directly into `RangeFinder.SetTau` with no model drift.

## Results (2000 games / 16,028 positions, NWL23, 2026-07-24)

- Overall MLE **tau\* = 0.119** (95% CI [0.118, 0.122]); a 1000-game subsample
  gave 0.118, so the estimate is stable. The shipped default 0.05 is far too
  sharp — its mean log-likelihood (−2.05) is barely better than a uniform guess
  over the candidates (−2.31), vs −1.66 at tau\*.
- Strong phase dependence of the 2-ply/5-ply mismatch:

  | tiles in bag | n | tau\* | top-1 agreement |
  |---|---|---|---|
  | 46+ | 7667 | 0.026 | 72.9% |
  | 31–45 | 2749 | 0.040 | 73.9% |
  | 11–30 | 3790 | 0.095 | 65.1% |
  | 2–10 | 1822 | 0.92 | 46.3% |

  The shallow sim is an excellent proxy for BestBot early in the game and
  degrades sharply as the pre-endgame approaches — suggesting a phase-dependent
  tau (or deeper mini-sims late) rather than a single constant.
- Tau\* falls monotonically with tiles played (1 tile: ≈0.97; 6 tiles: ≈0.057):
  the 2-ply sim most misjudges short fishing/setup-style moves.
- BestBot's move was the shallow sim's top choice 68.4% of the time, and in its
  top three 90.6% of the time.

## Caveats and future work

- The fitted tau is specific to the (evaluator, player) pair: 2-ply/200-iteration
  mini-sim vs BestBot-with-5-ply. Changing mini-sim depth/iterations, the leave
  file, or the win-prob model shifts the right tau; refit after such changes.
- One global tau is a compromise across phases; the per-bucket table above is the
  starting point for a tiles-in-bag-dependent tau in the rangefinder.
- Human opponents need their own (larger) tau — extract with `-player human`.
  Rating-bucketed fits would give a skill-dependent temperature.
- Other lexica (CSW24 etc.) should be refit with `-mode fetch -lexicon ...`.
