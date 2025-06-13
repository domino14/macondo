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
Games played: 2261
HastyBot wins: 1204.5 (53.273%)
HastyBot Mean Score: 434.4122  Stdev: 61.2653
FastMlBot Mean Score: 423.5997  Stdev: 54.2278
HastyBot Mean Bingos: 2.0411  Stdev: 1.0512
FastMlBot Mean Bingos: 1.6661  Stdev: 0.9765
HastyBot Mean Points Per Turn: 37.7376  Stdev: 6.5382
FastMlBot Mean Points Per Turn: 37.0873  Stdev: 5.9293
HastyBot went first: 1131.0 (50.022%)
Player who went first wins: 1252.5 (55.396%)
```

Ideas to try:

- [ ] Inference with Python (faster than ONNX?) - would need server
- [ ] Validation set (Graph loss/etc)
- [ ] Train on transposed positions
- [ ] Remove our last move/leave values
- [ ] Normalize bag and rack by count of tile, not by fixed numbers
- [ ] Include opponent's last play
- [ ] Set target to win % after Monte Carlo simulation (need a lot of time to collect enough data)
