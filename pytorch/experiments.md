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

- [x] Inference with Python (faster than ONNX?) - would need server (use NVIDIA Triton actually)
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

### Normalize bag by count in bag, not by fixed numbers (6/13/25)

This will give us a rough probability of how many of each tile we could be expected to draw. Bring back score/leave as well.

First experiment failed because there were several untenable bugs.

### Fix bugs, parallelize inference and training (6/15/25)

Adding parallelization to training introduced several bugs and surfaced other bugs that I hadn't known about. I'm about 95% confident the model is actually representing what it's supposed to (I'm more unconfident about exchanges, I'm almost 100% sure that it works fine for tile-play moves). This will need to be spot-checked.

Also added parallelization to inferences, fixed a few bugs around that, and tried Triton with GPU - it speeds up inference about 100X or more!

As a result of it all, we have the first model that barely beats HastyBot!

```
Games played: 31875
HastyBot wins: 15891.5 (49.856%)
HastyBot Mean Score: 446.5599  Stdev: 76.9168
FastMlBot Mean Score: 423.2252  Stdev: 67.2689
HastyBot Mean Bingos: 2.0627  Stdev: 1.0968
FastMlBot Mean Bingos: 2.0682  Stdev: 0.9830
HastyBot Mean Points Per Turn: 37.3685  Stdev: 6.3035
FastMlBot Mean Points Per Turn: 36.0960  Stdev: 7.6867
HastyBot went first: 15938.0 (50.002%)
Player who went first wins: 17987.5 (56.431%)
```

### Train on rand v rand (6/15/25)

Made a bot that plays a random play from the top 10 by equity. Let's train on it to see if anything good happens.

```
macondo> autoanalyze /tmp/games-autoplay.txt
Games played: 28759
HastyBot wins: 15410.5 (53.585%)
HastyBot Mean Score: 443.2089  Stdev: 76.3776
FastMlBot Mean Score: 413.8135  Stdev: 65.3490
HastyBot Mean Bingos: 2.0748  Stdev: 1.0865
FastMlBot Mean Bingos: 1.7034  Stdev: 0.9669
HastyBot Mean Points Per Turn: 37.6251  Stdev: 6.5083
FastMlBot Mean Points Per Turn: 35.7706  Stdev: 7.4254
HastyBot went first: 14380.0 (50.002%)
Player who went first wins: 15887.5 (55.244%)
```

Not very good. Let's use a softmax / temperature parameter.

### Train on hasty v rand-softmax (6/15/25)

Softmax chooser generates 50 top moves and picks from a prob distribution, using temperature = 1.0. When there are 60 or fewer tiles in the bag, it chooses a temperature of 0.0. This causes some early exploration, hopefully allowing the model to see that putting a vowel next to a 2LS in the beginning of the game is bad.

If that doesn't work we can add a feature - 15-size vector that has a 1 for "vowel next to 2LS" and train with that.

```
macondo> autoanalyze /tmp/games-autoplay.txt
Games played: 2835007
HastyBot wins: 1431715.5 (50.501%)
HastyBot Mean Score: 436.3040  Stdev: 58.8817
RandomBot Mean Score: 435.0373  Stdev: 58.9473
HastyBot Mean Bingos: 2.0475  Stdev: 1.0296
RandomBot Mean Bingos: 2.0432  Stdev: 1.0286
HastyBot Mean Points Per Turn: 38.3349  Stdev: 6.3499
RandomBot Mean Points Per Turn: 38.2267  Stdev: 6.3551
HastyBot went first: 1417504.0 (50.000%)
Player who went first wins: 1581291.5 (55.777%)
```

Hasty v Rand-Softmax is a very close matchup. Will it result in a better model?


BTW - verified that exchange vectors are being generated correctly, so we are now pretty sure our machine learning vectors are all accurate.


- Also step up validation size to 150,000

Results:

```
Games played: 39889
HastyBot wins: 19843.0 (49.746%)
HastyBot Mean Score: 441.5946  Stdev: 70.1797
FastMlBot Mean Score: 424.4761  Stdev: 61.1595
HastyBot Mean Bingos: 2.0284  Stdev: 1.0683
FastMlBot Mean Bingos: 2.0398  Stdev: 0.9785
HastyBot Mean Points Per Turn: 37.4837  Stdev: 6.3793
FastMlBot Mean Points Per Turn: 36.6343  Stdev: 6.9273
HastyBot went first: 19945.0 (50.001%)
Player who went first wins: 22233.0 (55.737%)
```


Better than our first model! This will be model 2.


### Add opponent's last play (6/16/25)

Was also using Hasty v Rand-softmax since we got good results with this.

```
macondo> autoanalyze /tmp/games-autoplay.txt
Games played: 25033
HastyBot wins: 12644.5 (50.511%)
HastyBot Mean Score: 444.1086  Stdev: 75.4048
FastMlBot Mean Score: 422.9002  Stdev: 65.0795
HastyBot Mean Bingos: 2.0470  Stdev: 1.0815
FastMlBot Mean Bingos: 2.0332  Stdev: 0.9780
HastyBot Mean Points Per Turn: 37.4520  Stdev: 6.2960
FastMlBot Mean Points Per Turn: 36.3930  Stdev: 7.4809
HastyBot went first: 12517.0 (50.002%)
Player who went first wins: 14080.5 (56.248%)
```

Doesn't do as well as without the last play :/

### Add a representation of our play on the board too (6/16/25)

Training set: Hasty v Rand-softmax, about 5M games.


### Train on rand-softmax v rand-softmax