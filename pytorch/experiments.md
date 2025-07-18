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
- [x] Include opponent's last play
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

```
Games played: 55086
HastyBot wins: 27457.5 (49.845%)
HastyBot Mean Score: 443.9589  Stdev: 76.4878
FastMlBot Mean Score: 422.5923  Stdev: 66.0006
HastyBot Mean Bingos: 2.0485  Stdev: 1.0908
FastMlBot Mean Bingos: 2.0383  Stdev: 0.9803
HastyBot Mean Points Per Turn: 37.3446  Stdev: 6.3467
FastMlBot Mean Points Per Turn: 36.2106  Stdev: 7.5318
HastyBot went first: 27543.0 (50.000%)
Player who went first wins: 30723.5 (55.774%)
```

Decent, about as good as our best model, or statistically indistinguishable.

A longer run:

```
Games played: 222714
HastyBot wins: 110070.5 (49.422%)
HastyBot Mean Score: 443.6442  Stdev: 76.6556
FastMlBot Mean Score: 422.8754  Stdev: 66.1699
HastyBot Mean Bingos: 2.0442  Stdev: 1.0915
FastMlBot Mean Bingos: 2.0432  Stdev: 0.9787
HastyBot Mean Points Per Turn: 37.3182  Stdev: 6.3709
FastMlBot Mean Points Per Turn: 36.2385  Stdev: 7.5451
HastyBot went first: 111357.0 (50.000%)
Player who went first wins: 125257.5 (56.241%)
```

Good model!

For CSW24:

```
macondo> autoanalyze /tmp/games-autoplay.txt
Games played: 251408
HastyBot wins: 125448.5 (49.898%)
HastyBot Mean Score: 466.2470  Stdev: 77.8985
FastMlBot Mean Score: 444.5643  Stdev: 68.0848
HastyBot Mean Bingos: 2.2917  Stdev: 1.1265
FastMlBot Mean Bingos: 2.2603  Stdev: 1.0144
HastyBot Mean Points Per Turn: 40.6773  Stdev: 6.7721
FastMlBot Mean Points Per Turn: 39.4990  Stdev: 8.0347
HastyBot went first: 125704.0 (50.000%)
Player who went first wins: 141351.5 (56.224%)
```

Note: the above model is the same (trained on NWL23). We can probably get better
results for a CSW24-specific model.


### Train on bogowin as target variable  (6/17/25)

```
Games played: 156888
HastyBot wins: 83533.5 (53.244%)
HastyBot Mean Score: 456.7695  Stdev: 73.5927
FastMlBot Mean Score: 400.8373  Stdev: 73.0475
HastyBot Mean Bingos: 2.0614  Stdev: 1.0645
FastMlBot Mean Bingos: 1.7964  Stdev: 1.0367
HastyBot Mean Points Per Turn: 36.2815  Stdev: 6.0308
FastMlBot Mean Points Per Turn: 32.9691  Stdev: 8.1514
HastyBot went first: 78444.0 (50.000%)
Player who went first wins: 87905.5 (56.031%)
```

Not as good, but shocking difference in average score. That FastMLBot wins 46.75%
of games is incredible with that difference in average score. Need to analyze these results further.

### Train with HastyBot v STEEBot

Hypothesis is that I want a bot that makes a few more "mistakes" so I can see more positions where it is worth it to sacrifice equity.

We will train on win-after-5 plies, bogowin target seems to be broken.

```
Games played: 34682
HastyBot wins: 18301.5 (52.769%)
HastyBot Mean Score: 441.5951  Stdev: 61.6206
FastMlBot Mean Score: 406.4064  Stdev: 58.2548
HastyBot Mean Bingos: 2.0274  Stdev: 1.0329
FastMlBot Mean Bingos: 1.7450  Stdev: 0.9795
HastyBot Mean Points Per Turn: 37.0319  Stdev: 6.3778
FastMlBot Mean Points Per Turn: 34.8631  Stdev: 6.3483
HastyBot went first: 17341.0 (50.000%)
Player who went first wins: 19528.5 (56.307%)
```

Not great. Back to softmax.

### Train with HastyBot v Softmax again

New softmax parameters: temperature 2 down to 0 tiles left in the bag.
Hasty beats old softmax bot ~50.6% of the time.
Hasty beats STEEBot more than 70% of the time.

So maybe we want to try something in between to see how it does?

Hasty beats softmax with temp 2 ~56% of the time, let's try this ::shrug::

```
Games played: 19719
HastyBot wins: 10217.5 (51.816%)
HastyBot Mean Score: 456.5801  Stdev: 80.9562
FastMlBot Mean Score: 394.4646  Stdev: 85.7891
HastyBot Mean Bingos: 2.0694  Stdev: 1.0804
FastMlBot Mean Bingos: 1.8179  Stdev: 1.0609
HastyBot Mean Points Per Turn: 36.9656  Stdev: 6.3370
FastMlBot Mean Points Per Turn: 33.1946  Stdev: 8.9318
HastyBot went first: 9860.0 (50.003%)
Player who went first wins: 11209.5 (56.846%)
```

Nope. (Interesting bot though. Such a low scoring average for still a good win ratio vs HastyBot)

How about temperature 1 down to 0 tiles left in the bag?. Hasty beats this one ~51.4% of the time. Maybe it'll work better?


`~/data/autoplay-softmax-v-hasty-4.txt`


```
macondo>     autoanalyze /tmp/games-autoplay.txt
Games played: 34725
HastyBot wins: 18179.0 (52.351%)
HastyBot Mean Score: 458.6027  Stdev: 79.7317
FastMlBot Mean Score: 395.1538  Stdev: 81.8634
HastyBot Mean Bingos: 2.0870  Stdev: 1.0820
FastMlBot Mean Bingos: 1.8313  Stdev: 1.0556
HastyBot Mean Points Per Turn: 36.8781  Stdev: 6.1157
FastMlBot Mean Points Per Turn: 33.2102  Stdev: 8.7180
HastyBot went first: 17363.0 (50.001%)
Player who went first wins: 19319.0 (55.634%)
```

Nope. Kind of worried that I messed something up now.

Let's try recreating the 50.6% bot, and then try training on different numbers of plies.

Ran 7.46M games (oops, too many) and saved to `~/data/autoplay-softmax-v-hasty-5.txt`. Then train.

I forgot about the `torch.tanh`! Let's try that again.

```
Games played: 213298
HastyBot wins: 104547.5 (49.015%)
HastyBot Mean Score: 439.9196  Stdev: 70.2616
FastMlBot Mean Score: 425.5268  Stdev: 61.6670
HastyBot Mean Bingos: 2.0277  Stdev: 1.0681
FastMlBot Mean Bingos: 2.0686  Stdev: 0.9677
HastyBot Mean Points Per Turn: 37.2882  Stdev: 6.5674
FastMlBot Mean Points Per Turn: 36.4596  Stdev: 6.8759
HastyBot went first: 106649.0 (50.000%)
Player who went first wins: 119615.5 (56.079%)
```

**This is great!** It's basically a 51% win model against Hasty! We save this as model 2 now. (The old model 2 became model 1 I think).


Trying the `torch.tanh` fix on the previous dataset (`~/data/autoplay-softmax-v-hasty-4.txt`),
basically training data of Hasty vs SoftmaxBot-Full (i.e., use temperature of 1 throughout a game):

```
Games played: 67951
HastyBot wins: 33735.0 (49.646%)
HastyBot Mean Score: 443.0799  Stdev: 74.4996
FastMlBot Mean Score: 423.1817  Stdev: 64.9303
HastyBot Mean Bingos: 2.0453  Stdev: 1.0812
FastMlBot Mean Bingos: 2.0581  Stdev: 0.9755
HastyBot Mean Points Per Turn: 37.3700  Stdev: 6.4462
FastMlBot Mean Points Per Turn: 36.2679  Stdev: 7.3339
HastyBot went first: 33976.0 (50.001%)
Player who went first wins: 38028.0 (55.964%)
```

More sensible, but not as good as the last model. Still, there's probably a bit
of a margin of error here.

### 3 plies

Look out 3 instead of 5 plies. Maybe there's too much noise at 5 plies and we
can learn better. Use the best dataset we have so far (`~/data/autoplay-softmax-v-hasty-5.txt`):

```
Games played: 209547
HastyBot wins: 105245.5 (50.225%)
HastyBot Mean Score: 440.6544  Stdev: 72.2606
FastMlBot Mean Score: 420.9047  Stdev: 62.7507
HastyBot Mean Bingos: 2.0426  Stdev: 1.0798
FastMlBot Mean Bingos: 1.9771  Stdev: 0.9552
HastyBot Mean Points Per Turn: 37.4824  Stdev: 6.7245
FastMlBot Mean Points Per Turn: 36.1746  Stdev: 6.9951
HastyBot went first: 104774.0 (50.000%)
Player who went first wins: 117327.5 (55.991%)
```

3 plies was too little.

### 4 plies? (6/20/25)

```
Games played: 206058
HastyBot wins: 101494.0 (49.255%)
HastyBot Mean Score: 442.6054  Stdev: 77.1187
FastMlBot Mean Score: 420.2734  Stdev: 68.0042
HastyBot Mean Bingos: 2.0443  Stdev: 1.0959
FastMlBot Mean Bingos: 2.0229  Stdev: 0.9677
HastyBot Mean Points Per Turn: 37.4071  Stdev: 6.6318
FastMlBot Mean Points Per Turn: 36.0002  Stdev: 7.5072
HastyBot went first: 103029.0 (50.000%)
Player who went first wins: 115859.0 (56.226%)
```

seems as we add more plies, we get better performance.

One more run of 5 plies, after fixing a potential endgame bug:

```
Games played: 189506
HastyBot wins: 93041.0 (49.097%)
HastyBot Mean Score: 439.0608  Stdev: 69.7367
FastMlBot Mean Score: 422.9370  Stdev: 63.3336
HastyBot Mean Bingos: 2.0248  Stdev: 1.0678
FastMlBot Mean Bingos: 2.0706  Stdev: 0.9645
HastyBot Mean Points Per Turn: 37.4331  Stdev: 6.5987
FastMlBot Mean Points Per Turn: 36.3162  Stdev: 6.9383
HastyBot went first: 94753.0 (50.000%)
Player who went first wins: 106246.0 (56.065%)
```


### Many plies

```
Games played: 656009
HastyBot wins: 317295.0 (48.367%)
HastyBot Mean Score: 438.3345  Stdev: 70.5534
FastMlBot Mean Score: 427.2371  Stdev: 62.2173
HastyBot Mean Bingos: 2.0331  Stdev: 1.0705
FastMlBot Mean Bingos: 2.0325  Stdev: 0.9657
HastyBot Mean Points Per Turn: 37.4935  Stdev: 6.4108
FastMlBot Mean Points Per Turn: 36.7779  Stdev: 7.0184
HastyBot went first: 328005.0 (50.000%)
Player who went first wins: 367239.0 (55.981%)
```

FastMLBot wins 51.63% of games against HastyBot!
(Saved as model `3`)

### Endgame fix

The NN bot is using its own implicit endgame algorithm. If it's behind or ahead by a lot it doesn't prioritize going out. It might even blow close endgames by not going out (I am not sure). We should consider either using the HastyBot endgame algorithm, or handcrafting an algorithm that prioritizes going out if it's possible.

Let's do this and run a couple of matches again to make sure they're not affecting the win rate. I don't think they are, but they seem to be affecting the average score.

(Implemented, in the 51.63% run above)

### More features

Train with three more features:
- V/C ratio in bag
- V/C ratio in rack
- Power tile count in bag

All of this can be deduced, but apparently it can be helpful to "pre-seed" the
NN with more features, even if a little redundant? Anyway, let's see if it works.

```
Games played: 255626
HastyBot wins: 123589.5 (48.348%)
HastyBot Mean Score: 436.4643  Stdev: 70.1158
FastMlBot Mean Score: 425.7528  Stdev: 61.0820
HastyBot Mean Bingos: 2.0229  Stdev: 1.0709
FastMlBot Mean Bingos: 1.9990  Stdev: 0.9579
HastyBot Mean Points Per Turn: 37.4571  Stdev: 6.4241
FastMlBot Mean Points Per Turn: 36.7565  Stdev: 6.9454
HastyBot went first: 127813.0 (50.000%)
Player who went first wins: 143352.5 (56.079%)
```

MLBot wins 51.65% of the time. Tiniest edge over last model, within margin of error, but since it's annoying to add/remove new features I'll just keep this
model. This will be our "final" static model (for now) so we can release this.


### Train on actual win instead of win after 5 plies

(See Many plies results above!)


- [ ] try adding a few more heads to stabilize model
- [ ] try bigger learning rates?
- [x] try looking out to 2 plies instead of 5 plies? 3 plies?
- [ ] try temperature varying from 1 to 0 linearly/in a curve depending on # tiles in bag?

### More features

Forgot to add opponent's last play _score_. This can be a good predictor of the goodness of tiles that they may have kept, and indeed, we get a little bump:

```
Games played: 170947
HastyBot wins: 82032.0 (47.987%)
HastyBot Mean Score: 436.8603  Stdev: 70.4650
FastMlBot Mean Score: 425.7584  Stdev: 61.4315
HastyBot Mean Bingos: 2.0230  Stdev: 1.0717
FastMlBot Mean Bingos: 2.0254  Stdev: 0.9621
HastyBot Mean Points Per Turn: 37.3172  Stdev: 6.4764
FastMlBot Mean Points Per Turn: 36.5636  Stdev: 6.9554
HastyBot went first: 85474.0 (50.000%)
Player who went first wins: 95373.0 (55.791%)
```

We passed the 52% barrier for FastMLBot.

Add the opponent's last play number of tiles as well. It's possible it doesn't "see" this from the last tile positions, as that is mostly spatial data. This feature can help it more with inference and the relationship between keeping good tiles and sacrificing points.

This training was accidentally interrupted about 3/4 of the way through. Let's test with this model anyway because most of the training game seems to come very early. Let's see if this is the case.

```
Games played: 220692
HastyBot wins: 106472.0 (48.245%)
HastyBot Mean Score: 436.1501  Stdev: 69.4948
FastMlBot Mean Score: 427.0988  Stdev: 60.2532
HastyBot Mean Bingos: 2.0240  Stdev: 1.0658
FastMlBot Mean Bingos: 2.0283  Stdev: 0.9532
HastyBot Mean Points Per Turn: 37.3779  Stdev: 6.4125
FastMlBot Mean Points Per Turn: 36.8260  Stdev: 6.8908
HastyBot went first: 110346.0 (50.000%)
Player who went first wins: 123721.0 (56.060%)
```

Sadly we can't be quite sure what happened here. Maybe we should train all the way to the end.

Let's change the opp exchange tiles to not be a one-hot encoded metric either. The number of tiles exchanged is not a "category".


```
Games played: 264786
HastyBot wins: 126707.5 (47.853%)
HastyBot Mean Score: 435.2447  Stdev: 68.3678
FastMlBot Mean Score: 426.5620  Stdev: 60.3480
HastyBot Mean Bingos: 2.0190  Stdev: 1.0708
FastMlBot Mean Bingos: 2.0050  Stdev: 0.9623
HastyBot Mean Points Per Turn: 37.3128  Stdev: 6.4913
FastMlBot Mean Points Per Turn: 36.7216  Stdev: 6.7899
HastyBot went first: 132393.0 (50.000%)
Player who went first wins: 148691.5 (56.155%)
```

That's better; around 52.15% for FastMLBot.

#### Add more history

Add the last two opponent plays (and our play in between). This should provide
a better signal for whether our opponent might be about to bingo. If this model
doesn't do well, can try taking away our "in-between" play. I'm not sure.

```
Games played: 329363
HastyBot wins: 158114.0 (48.006%)
HastyBot Mean Score: 436.2155  Stdev: 69.3642
FastMlBot Mean Score: 426.7953  Stdev: 60.7719
HastyBot Mean Bingos: 2.0239  Stdev: 1.0687
FastMlBot Mean Bingos: 2.0358  Stdev: 0.9584
HastyBot Mean Points Per Turn: 37.2891  Stdev: 6.4552
FastMlBot Mean Points Per Turn: 36.6793  Stdev: 6.8931
HastyBot went first: 164682.0 (50.000%)
Player who went first wins: 184749.0 (56.093%)
```

Try taking away our in-between play:

```
Games played: 218153
HastyBot wins: 105439.5 (48.333%)
HastyBot Mean Score: 437.6145  Stdev: 71.6119
FastMlBot Mean Score: 425.3666  Stdev: 62.4449
HastyBot Mean Bingos: 2.0286  Stdev: 1.0766
FastMlBot Mean Bingos: 2.0142  Stdev: 0.9610
HastyBot Mean Points Per Turn: 37.4438  Stdev: 6.4217
FastMlBot Mean Points Per Turn: 36.6436  Stdev: 7.1191
HastyBot went first: 109077.0 (50.000%)
Player who went first wins: 122300.5 (56.062%)
```

Try just adding a feature for "number of turns since last opp bingo" and go back
to single history?

```
Games played: 1001786
HastyBot wins: 480160.5 (47.930%)
HastyBot Mean Score: 435.5182  Stdev: 68.8356
FastMlBot Mean Score: 426.4469  Stdev: 60.3782
HastyBot Mean Bingos: 2.0197  Stdev: 1.0671
FastMlBot Mean Bingos: 2.0049  Stdev: 0.9600
HastyBot Mean Points Per Turn: 37.4524  Stdev: 6.5060
FastMlBot Mean Points Per Turn: 36.8241  Stdev: 6.8077
HastyBot went first: 500893.0 (50.000%)
Player who went first wins: 561082.5 (56.008%)
```

95% CI for this feature is (0.47832, 0.48028)
95% CI for last best model without this feature is (0.47663, 0.48043)


Probably should try taking away that feature again and see...

### Train on ML vs softmax

Can we try one of the players being a bot using the 52.1% model and the other
being softmax?

```
Games played: 309631
FastMlBot wins: 156112.5 (50.419%)
FastMlBot Mean Score: 424.4974  Stdev: 66.2355
HastyBot Mean Score: 442.3905  Stdev: 75.2271
FastMlBot Mean Bingos: 2.0268  Stdev: 0.9612
HastyBot Mean Bingos: 2.0461  Stdev: 1.0842
FastMlBot Mean Points Per Turn: 36.4770  Stdev: 7.6370
HastyBot Mean Points Per Turn: 37.5507  Stdev: 6.2719
FastMlBot went first: 154815.0 (50.000%)
Player who went first wins: 172804.5 (55.810%)
```

Didn't do well after training.

### Try bogowin again with 5 plies

Maybe I screwed something up earlier.

```
Games played: 304811
FastMlBot wins: 159551.5 (52.344%)
FastMlBot Mean Score: 427.6470  Stdev: 59.1465
HastyBot Mean Score: 434.8909  Stdev: 69.4700
FastMlBot Mean Bingos: 2.0228  Stdev: 0.9632
HastyBot Mean Bingos: 2.0081  Stdev: 1.0705
FastMlBot Mean Points Per Turn: 36.7400  Stdev: 6.8699
HastyBot Mean Points Per Turn: 37.1322  Stdev: 6.4409
FastMlBot went first: 152406.0 (50.000%)
Player who went first wins: 170784.5 (56.030%)
```

52.34%, best model yet! Try now with 3 plies?

```
Games played: 164109
FastMlBot wins: 83405.0 (50.823%)
FastMlBot Mean Score: 423.5841  Stdev: 58.6700
HastyBot Mean Score: 432.6651  Stdev: 68.7775
FastMlBot Mean Bingos: 1.8706  Stdev: 0.9768
HastyBot Mean Bingos: 1.9896  Stdev: 1.0736
FastMlBot Mean Points Per Turn: 36.0825  Stdev: 6.8554
HastyBot Mean Points Per Turn: 36.6394  Stdev: 6.4978
FastMlBot went first: 82055.0 (50.000%)
Player who went first wins: 91547.0 (55.784%)
```

No. Try with 7 plies:

```
Games played: 172623
FastMlBot wins: 90010.0 (52.143%)
FastMlBot Mean Score: 425.7617  Stdev: 62.9998
HastyBot Mean Score: 437.3403  Stdev: 72.3261
FastMlBot Mean Bingos: 2.0238  Stdev: 0.9613
HastyBot Mean Bingos: 2.0202  Stdev: 1.0764
FastMlBot Mean Points Per Turn: 36.4023  Stdev: 7.2569
HastyBot Mean Points Per Turn: 37.0964  Stdev: 6.3658
FastMlBot went first: 86312.0 (50.000%)
Player who went first wins: 96756.0 (56.050%)
```

6 plies?

```
Games played: 289613
FastMlBot wins: 150492.5 (51.963%)
FastMlBot Mean Score: 425.7795  Stdev: 62.9682
HastyBot Mean Score: 437.6534  Stdev: 72.4049
FastMlBot Mean Bingos: 2.0252  Stdev: 0.9642
HastyBot Mean Bingos: 2.0240  Stdev: 1.0789
FastMlBot Mean Points Per Turn: 36.4122  Stdev: 7.2737
HastyBot Mean Points Per Turn: 37.1189  Stdev: 6.3677
FastMlBot went first: 144807.0 (50.000%)
Player who went first wins: 162029.5 (55.947%)
```

4 plies:

```
Games played: 95615
FastMlBot wins: 48955.5 (51.201%)
FastMlBot Mean Score: 426.1207  Stdev: 57.1186
HastyBot Mean Score: 433.5960  Stdev: 67.8978
FastMlBot Mean Bingos: 1.9389  Stdev: 0.9673
HastyBot Mean Bingos: 2.0086  Stdev: 1.0708
FastMlBot Mean Points Per Turn: 36.7277  Stdev: 6.6532
HastyBot Mean Points Per Turn: 37.1738  Stdev: 6.3974
FastMlBot went first: 47807.0 (49.999%)
Player who went first wins: 53488.5 (55.942%)
```

### Shuffle training vectors

Use a buffer to shuffle the training vectors more. This should remove correlations since games usually get built up turn by turn and are close together in time. Use 100K size buffer.

```
Games played: 118997
FastMlBot wins: 61816.0 (51.948% +/- 0.284)
FastMlBot Mean Score: 425.3099  Stdev: 61.1140
HastyBot Mean Score: 435.3854  Stdev: 71.4321
FastMlBot Mean Bingos: 1.9932  Stdev: 0.9631
HastyBot Mean Bingos: 2.0069  Stdev: 1.0792
FastMlBot Mean Points Per Turn: 36.3852  Stdev: 7.1418
HastyBot Mean Points Per Turn: 36.9522  Stdev: 6.3799
FastMlBot went first: 59499.0 (50.000%)
Player who went first wins: 67075.0 (56.367%)
```

Unexpected poor performance. Try again with 150K instead of 100K for validation vector size (150K was what we used before).

```
Games played: 811139
FastMlBot wins: 424911.5 (52.385% +/- 0.109)
FastMlBot Mean Score: 425.8282  Stdev: 60.5918
HastyBot Mean Score: 434.5577  Stdev: 69.9602
FastMlBot Mean Bingos: 1.9968  Stdev: 0.9655
HastyBot Mean Bingos: 2.0006  Stdev: 1.0708
FastMlBot Mean Points Per Turn: 36.4742  Stdev: 7.0188
HastyBot Mean Points Per Turn: 37.0059  Stdev: 6.4583
FastMlBot went first: 405570.0 (50.000%)
Player who went first wins: 454474.5 (56.029%)
```

weird, but best model so far.


### Train with bigger model (more params)

Used 96 channels, 10 blocks for CNN (vs 64, 6 used for all other prior models).
Also used an adaptive learning rate.

This results in a significant improvement!

```
Games played: 188339
FastMlBot wins: 99632.0 (52.900% +/- 0.225)
FastMlBot Mean Score: 426.8498  Stdev: 60.4142
HastyBot Mean Score: 434.1403  Stdev: 68.6183
FastMlBot Mean Bingos: 2.0211  Stdev: 0.9755
HastyBot Mean Bingos: 2.0002  Stdev: 1.0684
FastMlBot Mean Points Per Turn: 36.4535  Stdev: 6.9247
HastyBot Mean Points Per Turn: 36.9272  Stdev: 6.4077
FastMlBot went first: 94167.0 (49.999%)
Player who went first wins: 105525.0 (56.029%)
```

**52.9% win rate vs HastyBot** !!


Note: The bigger model required shrinking the shuffle buffer from the previous improvement,
otherwise my 64GB RAM machine OOMed. Shrank buffer from 100K vectors to 20K vectors.

### Try smaller model with learning rate improvement

Try (64, 6) CNN size with the adaptive learning rate to isolate that particular improvement.

```
Games played: 237136
FastMlBot wins: 124018.0 (52.298% +/- 0.201)
```

Kinda the same as before

### Longer run for bigger model

```
Games played: 842245
FastMlBot wins: 442500.5 (52.538% +/- 0.107)
```

Sadly the 52.9% might have been illusory / good luck / noise. I'm not sure exactly what happened there.

### Convert model to TensorRT architecture + FP16

```
Games played: 876342
FastMlBot wins: 460561.0 (52.555% +/- 0.105)
FastMlBot Mean Score: 425.7805  Stdev: 59.6600
HastyBot Mean Score: 433.6229  Stdev: 66.9563
FastMlBot Mean Bingos: 2.0136  Stdev: 0.9726
HastyBot Mean Bingos: 1.9991  Stdev: 1.0599
FastMlBot Mean Points Per Turn: 36.4628  Stdev: 6.7907
HastyBot Mean Points Per Turn: 36.9658  Stdev: 6.3853
FastMlBot went first: 438171.0 (50.000%)
Player who went first wins: 491546.0 (56.091%)
```

Not bad. Maybe we try an even bigger model? (128 channels instead of 96?)

```
Games played: 1041365
HastyBot wins: 495144.0 (47.548% +/- 0.096)
HastyBot Mean Score: 431.9879  Stdev: 65.3669
FastMlBot Mean Score: 426.2294  Stdev: 58.0871
HastyBot Mean Bingos: 1.9900  Stdev: 1.0541
FastMlBot Mean Bingos: 1.9795  Stdev: 0.9752
HastyBot Mean Points Per Turn: 36.9933  Stdev: 6.3967
FastMlBot Mean Points Per Turn: 36.6583  Stdev: 6.6265
HastyBot went first: 520683.0 (50.000%)
Player who went first wins: 583944.0 (56.075%)
```

Win rate for FastMLBot is 52.45%. Could be within margin of error, but with no
real improvement, we can go back to the 96-channel network.

Try 8-bit quantization for older model (96-channel):

(Can't get it to work. Returns NaN for all inputs)

Try an 80-channel model; maybe the best performance is between 64 and 96 channels?

```
Games played: 1368951
HastyBot wins: 650248.0 (47.500% +/- 0.084)
HastyBot Mean Score: 434.4154  Stdev: 66.6972
FastMlBot Mean Score: 427.2275  Stdev: 59.8745
HastyBot Mean Bingos: 2.0033  Stdev: 1.0559
FastMlBot Mean Bingos: 2.0332  Stdev: 0.9729
HastyBot Mean Points Per Turn: 37.1745  Stdev: 6.3836
FastMlBot Mean Points Per Turn: 36.7321  Stdev: 6.8234
HastyBot went first: 684476.0 (50.000%)
Player who went first wins: 766876.0 (56.019%)
```

Models so far:

```
128-ch: 52.45%
96-ch: 52.55%
80-ch: 52.50%
64-ch: 52.39%
```

Keep the 96-channel model for now. Don't know if we can ever reproduce the 52.9% result again :/ must have been a fluke / not enough samples.

------

### Better data

Let's collect BestBot / simming data. If we collect multiple plays per position with bogowin pct data up to 5 plies we should have a better model. Hopefully it will win significantly more than this model. But it will take a bunch of time to collect this data.

Data collection experiments:

```
 autoplay -botcode1 CUSTOM_BOT -botcode2 CUSTOM_BOT -threads 1 -botspec1 '{"algo":"none","params":{"min_sim_plies":5,"has_simming":true}}' -botspec2 '{"algo":"none","params":{"min_sim_plies":5,"has_simming":true,"sim_use_magpie":true}}'
```

Bot 1 uses built-in simmer. Bot2 uses magpie. Our first purpose is to collect enough data where we can be pretty sure that the magpie bot is a suitable replacement for BestBot.

Note that we've turned off endgame/preendgame for both. They will both use the same algorithm for this. We only care about the sims for this experiment (for speed).