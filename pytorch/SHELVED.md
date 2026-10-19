# Shelved experiments

Ideas from the value-net program (Sep-Oct 2026) that were built, measured and
did not help. Their code has been removed so the repository only shows what
the shipped models use; it is all in git history. Each entry gives the
result, where the code lived, and what it would take to try again. The
full narrative is in `experiments.md`, every run is in `RUNS.md`.

To bring code back, find the commit that removed it and check out the path
from its parent:

    git log --diff-filter=D --oneline -- pytorch/ising_fit.py   # the removing commit, C
    git checkout C^ -- pytorch/ising_fit.py boardenergy/         # restore what you need

All results are 100,000 game pairs vs HastyBot (95% interval about ±0.18)
unless noted. The baseline is `macondo-tf` (streamopen) at **57.65%**.

## Rollout labels (generation 1)

Label each position with the mean of 16 two-ply rollouts, the net scoring
each line's end, instead of a real game result.

- **Result:** 52.03%, against 53.19% for the same-era net on real labels.
  The labels trained smoothly and lost at the board.
- **Why it failed (likely):** each rollout dealt the opponent a uniformly
  random rack; real racks are kept leaves plus a draw, so every rollout was
  optimistic for the mover. Also 9.6M labelled positions against 163M.
- **Code:** `cmd/mlproducer/rollout.go` (`-labeler rollout`, `-plies`,
  `-rollouts`, `-sample`, `-labels-out`), `run-gen1.sh`.
- **To retry:** deal the opponent their real rack at ply 1 (the log has it)
  or sample it from the inference posterior; label far more positions. The
  last version also fixed a deal-order bug (the opponent was dealt short
  with a nearly empty bag).

## Ising board-shape energy

A pairwise maximum-entropy (Ising) model of which squares are occupied,
fitted per tile count (Witteveen & Bauer, "Statistical mechanics for
Scrabble", arXiv 2605.00813). Its energy measures how unusual a board shape
is.

- **Fit:** sane, and energy is real geometry (a null test against permuted
  parameters passes): high-energy boards have less extreme outcomes.
- **In play, both uses were null:**
  - energy-chosen extra candidates for the net (the 10 lowest- and 10
    highest-energy plays outside the equity top 50): 57.44%;
  - energy terms in the sim's end-of-line win model, 2-ply sim head to
    head with and without them: 49.85% ± 0.88 (2,500 pairs).
- **Code:** `boardenergy/` (Go), `ai/bot/mlenergy.go`
  (`MACONDO_ML_ENERGY_EXTRA`), `pytorch/ising_*.py`,
  `data/strategy/default/ising/`.
- **Untried:** the explorer bot it was built for, generating training games
  steered toward unusual board shapes.

## Fitted end-of-line win model for sims

A logistic model of P(win) from spread and unseen tiles, fitted on our
NWL23 games, to replace the win-percentage table at the end of each
simulated line (`-simleafwin1/2 base`).

- **Result:** 2-ply sim head to head, fitted model vs table: 50.79% ± 0.84
  (2,500 pairs, 1.8 SE). Suggestive, not significant; the table stayed.
  **The most promising thing on this list**: a longer match could settle it.
- **Code:** `boardenergy/leafwin.go`, `simLeafWin` in `ai/bot/mlenergy.go`,
  `Simmer.SetLeafWin` in `montecarlo/`, the autoplay `sim_leaf_win` option
  (proto field 18, now reserved), the fitter `pytorch/ising_leafwin.py`, the
  parameters in `data/strategy/default/ising/leafwin-v1.json`.

## Sim distillation

Fine-tune the net to rank candidates the way a 5-ply sim does. 1.26M
positions had every candidate simmed on a borrowed 192-thread machine
(about a week).

- **Result:** four losses, none better than the starting net:
  listwise 57.48%, rebalanced listwise 57.18%, pairwise 56.97%, top-4
  57.52%. Ranking a full list of 50 by a noisy sim teaches noise.
- **Code:** the ranking losses and candidate-group loading in
  `training.py`, `simlabel frames`, `build-groups-full.sh`,
  `queue-simft*.sh`, `plan-sim-distill.md`.
- **Kept:** `cmd/simlabel select|sim|check`, which label positions with
  the sim and measure how often a bot agrees with it (the agreement-by-phase
  table in `experiments.md`, "Opening moves: streamopen vs the 5-ply sim").
  The labels are archived (`results/simlabels-5ply-nwl23/`).

## Ownership heads

Two extra per-square training targets: which empty squares the mover / the
opponent will cover before the game ends (KataGo's ownership head).

- **Result:** 57.65%, identical to the same recipe without them.
- **Code:** `mlproducer -ownership`, the ownership heads in `training.py`,
  `queue-own.sh`.

## MAGPIE's 6-layer student

`macondo-tf` distilled by John O'Laughlin into the same transformer with 6
layers instead of 8 (2.77M parameters vs 3.66M), for MAGPIE.

- **Result:** 55.80% vs HastyBot, 48.07% head to head vs `macondo-tf`, and
  only 6% faster net-vs-net generation here, where the GPU is not the
  bottleneck.
- **Code:** `import_magpie_student.py`, `queue-student.sh`; the checkpoint is
  at `~/data/student-t6-so-rack/`.
- **To retry:** worth it where the network is the expensive part (CPU,
  laptops, inside a search), probably as a smaller student still.

## Measured, nothing to remove

These were configuration choices, not code; the knobs that remain are
harmless or still useful.

- **Top 100 candidates instead of 50** (`MACONDO_ML_TOPN=100`): 57.64%.
  Kept as a knob (generation uses 16).
- **Transpose augmentation** (`--transpose-prob 0.5`): +0.30 over the same
  run without it (2.3 SE), not adopted; the producer already transposes half
  of all games. Kept as a trainer option.
- **Training on self-play games from scratch** (3M games only): 54.45%,
  against 55.39% for 3M HastyBot games, and it loses that head to head
  (49.64%). Self-play helps as a fine-tune (`results/selfplay/`), not as the
  only data.
- **More of the same data** (27M → 54M games): 53.10% → 53.13%.

## One-off drivers

About thirty `queue-*.sh` and `run-*.sh` scripts each ran one experiment,
with paths and models hard-coded. Their results are in `experiments.md`;
the reproduction notes in `results/` pin the commits they used. The
scripts that remain are the shipped recipe (`run-stream.sh`,
`run-openings.sh`, `train-fresh.sh`, `watch-tf-heads.sh`, `gen-nvn.sh`,
`mix-logs.py`, `copy_model.sh`, `build-engines.sh`) and general tools
(matches: `queue-ml-v-ml.sh`, `queue-ml-v-sim.sh`, `queue-simbench.sh`,
`pairs-stats.py`; benchmarks, export and parity checks, visualizers).
