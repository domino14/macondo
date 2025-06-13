A list of experiments and ideas.

### Base model

Base model (6/12/25)

Features:
- Board planes
- Is-a-blank plane
- Horizontal and vertical cross-set planes
- Unoccupied hotspot planes
- 27 Rack scalars (normalized by division by 7)
- 27 bag scalars (normalized by division by 20)
- Last move score (ours)
- Last move leave (ours)
- Tiles remaining in bag (normalized by division by 100)
- Our spread

Predictor:
- The win (-1, 0, 1 for loss, draw, win) after <= 5 plies.

Note: features are given immediately post-placing each move.

Training stats (on gfx card):

```
6.27TiB data
Training complete after 44923 steps. (batch_size = 2048, so ~92M positions total)
Best loss: 0.2199
Total time: 21983.80 seconds
```


```
Games played: 5343
HastyBot wins: 2840.0 (53.154%)
HastyBot Mean Score: 433.5289  Stdev: 60.2139
FastMlBot Mean Score: 423.0625  Stdev: 54.3519
HastyBot Mean Bingos: 2.0313  Stdev: 1.0436
FastMlBot Mean Bingos: 1.6697  Stdev: 0.9801
HastyBot Mean Points Per Turn: 37.6897  Stdev: 6.5587
FastMlBot Mean Points Per Turn: 37.1053  Stdev: 5.9275
HastyBot went first: 2672.0 (50.009%)
Player who went first wins: 2965.0 (55.493%)
```

Ideas to try:

- [ ] Inference with Python (faster than ONNX?) - would need server (use NVIDIA Triton actually)
- [x] Validation set (Graph loss/etc)
- [x] Train on transposed positions
- [x] Remove our last move/leave values
- [x] Normalize bag and rack by count of tile, not by fixed numbers
- [ ] Include opponent's last play
- [ ] Set target to win % after Monte Carlo simulation (need a lot of time to collect enough data)

### Add transposition (6/13/25)

- Randomly transpose around 50% of positions

(And validation set is now working - this should be part of any ML training)

```
  44500  train=0.2211  val=0.2224  4,470 pos/s
    ✓ checkpointed (best validation)
```

![](model2.png)

```
Games played: 5394
HastyBot wins: 2872.5 (53.254%)
HastyBot Mean Score: 433.6110  Stdev: 59.2063
FastMlBot Mean Score: 423.2827  Stdev: 54.7302
HastyBot Mean Bingos: 2.0365  Stdev: 1.0372
FastMlBot Mean Bingos: 1.6809  Stdev: 0.9764
HastyBot Mean Points Per Turn: 37.7055  Stdev: 6.4496
FastMlBot Mean Points Per Turn: 37.1014  Stdev: 5.9583
HastyBot went first: 2697.0 (50.000%)
Player who went first wins: 3011.5 (55.831%)
```

Not too much of a change. The data set is probably hard to overfit, and
transposition only really has effect on the board shape for the very first move or two.


### Remove last move and leave values (6/13/25)

```
macondo> autoanalyze /tmp/games-autoplay.txt
Games played: 921
HastyBot wins: 548.5 (59.555%)
HastyBot Mean Score: 435.6840  Stdev: 59.6269
FastMlBot Mean Score: 415.5244  Stdev: 54.4507
HastyBot Mean Bingos: 2.0879  Stdev: 1.0341
FastMlBot Mean Bingos: 1.5331  Stdev: 0.9798
HastyBot Mean Points Per Turn: 37.7080  Stdev: 6.4333
FastMlBot Mean Points Per Turn: 36.2686  Stdev: 6.0016
HastyBot went first: 461.0 (50.054%)
Player who went first wins: 518.5 (56.298%)
```

This model is clearly wrong and was stopped early. It for example prefers DUGS with a rack of DUGDUGS to start. We need last move and leave values. From Gemini:

> In summary: Keep the lastMoveScore feature. It's a low-cost, high-value piece of information that captures the game's dynamics, providing a richer context for the model without preventing it from recognizing that the spread is the most important factor.

Same for leave, essentially.

###