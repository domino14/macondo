# selfplay: macondo-tf-selfplay (nvnft)

`macondo-tf` fine-tuned on its own games. Results (2026-10-16), 100,000 game
pairs each, NWL23, FastMlBot ranking HastyBot's top 50 plays, decided-game
rule on:

| match | paired win rate |
|---|---|
| nvnft vs HastyBot | 57.07% ± 0.18 |
| hastyft (control, below) vs HastyBot | 57.56% ± 0.18 |
| nvnft vs hastyft, head to head | 50.30% ± 0.16 |
| nvnft vs macondo-tf (streamopen), head to head | 50.37% ± 0.16 |

Self-play loses half a point against HastyBot but beats both its parent and
the same-size HastyBot-data control head to head. Shipped as
`macondo-tf-selfplay`; the narrative is in `experiments.md` ("Result: net-vs-net
A/B" and the two head-to-head sections after it).

## Recipe

1. **Games.** 3M net-vs-net games, `macondo-tf` (streamopen) against itself,
   sampled openings and quick 2-ply endgames as in the open sets:

       MODEL=macondo-tf CHUNKS=12 CHUNK_GAMES=250000 pytorch/gen-nvn.sh

   Output: `~/data/nvn/nvn-*.txt.gz`. The control arm's 3M fresh HastyBot
   games came from `GEN=1 TRAIN=0 TAG=hasty3m GAMES=3000000 run-openings.sh`.

2. **Mix** the new games 50/50 with the same number of turn rows from the
   head of `open2` (one of the two sets `macondo-tf` was trained on):

       R=$(zcat ~/data/nvn/nvn-*.txt.gz | grep -vc '^playerID,')
       python3 pytorch/mix-logs.py ~/data/mix-nvn.txt.gz "$HOME/data/nvn/nvn-*.txt.gz" ~/data/open2.txt.gz:$R

3. **Fine-tune** from the `macondo-tf` checkpoint for 11k steps (effective
   batch 2,048), lr 1e-4, 500 warmup steps, keeping the final weights:

       TAG=nvnft LOGS=~/data/mix-nvn.txt.gz VAL=val-streamopen.bin PASSES=7 ROWS_PER_PASS=4000000 \
       STEPS=11000 BATCH=128 ACCUM=16 GPU_MEM=5120 LR=1e-4 WARMUP=500 DEPLOY_FINAL=1 \
       TRAIN_EXTRA="--init-ckpt ~/data/results/streamed-stream-streamopen/model/streamopen/best-tf-streamopen.pt" \
         pytorch/run-stream.sh

   (`hastyft`, the control, is the same command on `mix-hasty.txt.gz`.) A pass
   yields ~4.0M positions, since only ~68% of games have a turn before the bag
   empties, hence 7 passes for 11k steps.

The checkpoint is archived at
`~/data/results/selfplay-nvnft/model/best-tf-nvnft-step11000.pt`. The driver
that ran both arms, `queue-nvn-ab.sh`, was removed with the other one-off
drivers; `SHELVED.md` says how to get it back from git history.
