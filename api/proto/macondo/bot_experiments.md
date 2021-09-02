### Experiments between different level bots

These were run with the macondo shell `autoplay` command and the CSW19 lexicon.

e.g. `autoplay -lexicon CSW19 -botcode1 LEVEL4_PROBABILISTIC -botcode2 LEVEL3_CEL_BOT`

The numbers in parentheses after the Bot descriptions are the `BotFindabilities` and `BotParallelFindabilities` values for this experiment, respectively. See `/ai/runner/filters.go` for more information.

```
HastyBot vs Level 1 CEL Bot (0.2, 0.25)

Games played: 26360
exhaustiveleave-1 wins: 26356.5 (99.987%)
exhaustiveleave-1 went first: 13222.0 (50.159%)
Player who went first wins: 13223.5 (50.165%)
exhaustiveleave-1 Mean Score: 548.402162 Stdev: 63.548076
exhaustiveleave-2 Mean Score: 232.590061 Stdev: 43.576847
exhaustiveleave-1 Mean Bingos: 2.596055 Stdev: 1.099455
exhaustiveleave-2 Mean Bingos: 0.548900 Stdev: 0.634353
```

```
Level 2 CEL Bot (0.5, 0.5) vs Level 1 CEL Bot (0.2, 0.25)

Games played: 29557
exhaustiveleave-1 wins: 25861.0 (87.495%)
exhaustiveleave-1 went first: 14592.0 (49.369%)
Player who went first wins: 15649.0 (52.945%)
exhaustiveleave-1 Mean Score: 363.797239 Stdev: 50.888498
exhaustiveleave-2 Mean Score: 269.741990 Stdev: 45.144363
exhaustiveleave-1 Mean Bingos: 1.010793 Stdev: 0.793795
exhaustiveleave-2 Mean Bingos: 0.579152 Stdev: 0.652822

```

```
Level 2 CEL Bot (0.5, 0.5) vs Level 3 CEL Bot (1, 1)

Games played: 33451
exhaustiveleave-1 wins: 6290.5 (18.805%)
exhaustiveleave-1 went first: 16919.0 (50.578%)
Player who went first wins: 18032.5 (53.907%)
exhaustiveleave-1 Mean Score: 320.711249  Stdev: 49.643561
exhaustiveleave-2 Mean Score: 398.550836  Stdev: 54.618290
exhaustiveleave-1 Mean Bingos: 0.980927  Stdev: 0.787151
exhaustiveleave-2 Mean Bingos: 1.406983  Stdev: 0.894215

```

```
Level 3 CEL Bot (1, 1) vs HastyBot

Games played: 25351
exhaustiveleave-1 wins: 1419.0 (5.597%)
exhaustiveleave-1 went first: 12764.0 (50.349%)
Player who went first wins: 13021.0 (51.363%)
exhaustiveleave-1 Mean Score: 344.426176  Stdev: 52.413843
exhaustiveleave-2 Mean Score: 497.455564  Stdev: 61.826039
exhaustiveleave-1 Mean Bingos: 1.268155  Stdev: 0.859592
exhaustiveleave-2 Mean Bingos: 2.425111  Stdev: 1.093924

```

**Level 4 CEL Bot is a true highest equity CEL bot, with no probability filters. We should use
it instead of the Level 3 one.**

```
Level 4 CEL Bot vs HastyBot

Games played: 50642
exhaustiveleave-1 wins: 2699.5 (5.331%)
exhaustiveleave-1 went first: 25469.0 (50.292%)
Player who went first wins: 25880.5 (51.105%)
exhaustiveleave-1 Mean Score: 379.921073  Stdev: 59.551011
exhaustiveleave-2 Mean Score: 555.629616  Stdev: 68.929911
exhaustiveleave-1 Mean Bingos: 1.432566  Stdev: 0.913210
exhaustiveleave-2 Mean Bingos: 2.731251  Stdev: 1.132050

```

```
Level 3 Probabilistic (0.4, 0.5) vs Level 3 CEL Bot (1, 1)

Games played: 20311
exhaustiveleave-1 wins: 13596.0 (66.939%)
exhaustiveleave-1 went first: 10193.0 (50.185%)
Player who went first wins: 11310.0 (55.684%)
exhaustiveleave-1 Mean Score: 406.924819  Stdev: 56.240675
exhaustiveleave-2 Mean Score: 366.752991  Stdev: 53.332857
exhaustiveleave-1 Mean Bingos: 1.659199  Stdev: 0.973377
exhaustiveleave-2 Mean Bingos: 1.336025  Stdev: 0.878924
```

```
Level 4 Probabilistic (0.7, 0.7) vs Level 3 CEL Bot (1, 1)

Games played: 26204
exhaustiveleave-1 wins: 22586.5 (86.195%)
exhaustiveleave-1 went first: 13102.0 (50.000%)
Player who went first wins: 13945.5 (53.219%)
exhaustiveleave-1 Mean Score: 456.834224  Stdev: 59.441599
exhaustiveleave-2 Mean Score: 354.039994  Stdev: 52.716225
exhaustiveleave-1 Mean Bingos: 2.076019  Stdev: 1.037675
exhaustiveleave-2 Mean Bingos: 1.312967  Stdev: 0.869019
```

```
Level 4 Probabilistic (0.6, 0.7) vs Level 3 CEL Bot (1, 1)

Games played: 21706
exhaustiveleave-1 wins: 18063.5 (83.219%)
exhaustiveleave-1 went first: 10843.0 (49.954%)
Player who went first wins: 11678.5 (53.803%)
exhaustiveleave-1 Mean Score: 447.584355  Stdev: 59.194729
exhaustiveleave-2 Mean Score: 356.628167  Stdev: 53.187985
exhaustiveleave-1 Mean Bingos: 1.984014  Stdev: 1.031623
exhaustiveleave-2 Mean Bingos: 1.314475  Stdev: 0.875829

```

```
Level 4 Probabilistic (0.6, 0.7) vs Level 4 CEL Bot

Games played: 21561
exhaustiveleave-1 wins: 18330.0 (85.015%)
exhaustiveleave-1 went first: 10806.0 (50.118%)
Player who went first wins: 11482.0 (53.254%)
exhaustiveleave-1 Mean Score: 502.192431  Stdev: 66.183802
exhaustiveleave-2 Mean Score: 392.933352  Stdev: 59.897908
exhaustiveleave-1 Mean Bingos: 2.262140  Stdev: 1.082548
exhaustiveleave-2 Mean Bingos: 1.470525  Stdev: 0.930705

```

```
Level 3 Probabilistic (0.4, 0.5) vs HastyBot

Games played: 20535
exhaustiveleave-1 wins: 2562.5 (12.479%)
exhaustiveleave-1 went first: 10082.0 (49.097%)
Player who went first wins: 11135.5 (54.227%)
exhaustiveleave-1 Mean Score: 370.471390  Stdev: 54.663373
exhaustiveleave-2 Mean Score: 482.395374  Stdev: 61.378951
exhaustiveleave-1 Mean Bingos: 1.540346  Stdev: 0.933410
exhaustiveleave-2 Mean Bingos: 2.364694  Stdev: 1.085553

```

```
Level 4 Probabilistic (0.7, 0.7) vs HastyBot

Games played: 25199
exhaustiveleave-1 wins: 7722.5 (30.646%)
exhaustiveleave-1 went first: 12597.0 (49.990%)
Player who went first wins: 13959.5 (55.397%)
exhaustiveleave-1 Mean Score: 416.483392  Stdev: 57.950642
exhaustiveleave-2 Mean Score: 467.672487  Stdev: 60.911954
exhaustiveleave-1 Mean Bingos: 1.925751  Stdev: 1.012151
exhaustiveleave-2 Mean Bingos: 2.308623  Stdev: 1.077128
```

```
Level 4 Probabilistic (0.6, 0.7) vs HastyBot

Games played: 28596
exhaustiveleave-1 wins: 7485.0 (26.175%)
exhaustiveleave-1 went first: 14450.0 (50.532%)
Player who went first wins: 15564.0 (54.427%)
exhaustiveleave-1 Mean Score: 407.765981  Stdev: 57.263336
exhaustiveleave-2 Mean Score: 470.522521  Stdev: 60.867192
exhaustiveleave-1 Mean Bingos: 1.827423  Stdev: 0.999096
exhaustiveleave-2 Mean Bingos: 2.324906  Stdev: 1.074662

```

For the purposes of a good stratification then:

cel2 beats cel1 87.5%
cel3 beats cel2 81.2%
prob4 beats cel3 83.2% (the 0.6, 0.7 version)
hasty beats prob4 73.83%

The hasty win % vs prob4 is not as high as the others, but this is ok. It means this bot (prob4) will be pretty good, almost as good as Hasty.

### Non-English languages

For non-English languages, need to likely use the Level 1 through Level 3 probabilistic bots as well, since there is no CEL equivalent yet.

English baseline (CSW19):

```
Level 1 probabilistic (0.1, 0.1) vs HastyBot

Games played: 26372
exhaustiveleave-1 wins: 58.0 (0.220%)
exhaustiveleave-1 went first: 13267.0 (50.307%)
Player who went first wins: 13131.0 (49.791%)
exhaustiveleave-1 Mean Score: 254.913241  Stdev: 47.337143
exhaustiveleave-2 Mean Score: 528.591688  Stdev: 63.164315
exhaustiveleave-1 Mean Bingos: 0.751213  Stdev: 0.726710
exhaustiveleave-2 Mean Bingos: 2.509214  Stdev: 1.098549

```

L1 bot bingoes more often than the CEL-1, but its average score is still pretty low. This is probably a good level. It's not advisable to go too much lower than 0.1 for the findability factors because it'll make the bot slower.

Let's try it with the French lexicon.

`autoplay -lexicon FRA20 -letterdistribution french -botcode1 LEVEL1_PROBABILISTIC -botcode2 HASTY_BOT`

```
Level 1 probabilistic (0.1, 0.1) vs HastyBot

Games played: 20488
exhaustiveleave-1 wins: 69.5 (0.339%)
exhaustiveleave-1 went first: 10199.0 (49.780%)
Player who went first wins: 10324.5 (50.393%)
exhaustiveleave-1 Mean Score: 277.839809  Stdev: 57.564837
exhaustiveleave-2 Mean Score: 572.418586  Stdev: 70.894352
exhaustiveleave-1 Mean Bingos: 1.095666  Stdev: 0.885126
exhaustiveleave-2 Mean Bingos: 3.106941  Stdev: 1.182037
```

```
Level 1 probabilistic (0.1, 0.1) vs Level 2 probabilistic (0.2, 0.3)

Games played: 19278
exhaustiveleave-1 wins: 2920.5 (15.149%)
exhaustiveleave-1 went first: 9653.0 (50.073%)
Player who went first wins: 10289.5 (53.374%)
exhaustiveleave-1 Mean Score: 306.728291  Stdev: 57.473210
exhaustiveleave-2 Mean Score: 410.007677  Stdev: 62.192600
exhaustiveleave-1 Mean Bingos: 1.108414  Stdev: 0.883040
exhaustiveleave-2 Mean Bingos: 1.714908  Stdev: 1.019525

```

```
Level 2 probabilistic (0.2, 0.3) vs Level 3 probabilistic (0.4, 0.5)

Games played: 18578
exhaustiveleave-1 wins: 3820.5 (20.565%)
exhaustiveleave-1 went first: 9314.0 (50.135%)
Player who went first wins: 10071.5 (54.212%)
exhaustiveleave-1 Mean Score: 364.158306  Stdev: 60.743482
exhaustiveleave-2 Mean Score: 450.962052  Stdev: 64.519130
exhaustiveleave-1 Mean Bingos: 1.634675  Stdev: 1.001521
exhaustiveleave-2 Mean Bingos: 2.247174  Stdev: 1.090090

```

```
Level 3 probabilistic (0.4, 0.5) vs Level 4 probabilistic (0.6, 0.7)
Games played: 21600
exhaustiveleave-1 wins: 6829.0 (31.616%)
exhaustiveleave-1 went first: 10835.0 (50.162%)
Player who went first wins: 11929.0 (55.227%)
exhaustiveleave-1 Mean Score: 417.529167  Stdev: 62.971609
exhaustiveleave-2 Mean Score: 469.108333  Stdev: 65.349698
exhaustiveleave-1 Mean Bingos: 2.145880  Stdev: 1.077645
exhaustiveleave-2 Mean Bingos: 2.513380  Stdev: 1.129270

```

```
Level 1 probabilistic (0.07, 0.1) vs Level 2 probabilistic (0.15, 0.2)

Games played: 31446
exhaustiveleave-1 wins: 5131.0 (16.317%)
exhaustiveleave-1 went first: 15719.0 (49.987%)
Player who went first wins: 16950.0 (53.902%)
exhaustiveleave-1 Mean Score: 287.867455  Stdev: 55.904794
exhaustiveleave-2 Mean Score: 384.373656  Stdev: 61.517495
exhaustiveleave-1 Mean Bingos: 0.880398  Stdev: 0.816649
exhaustiveleave-2 Mean Bingos: 1.461458  Stdev: 0.972655

```

```
Level 2 probabilistic (0.15, 0.2) vs Level 3 probabilistic (0.35, 0.45)

Games played: 25644
exhaustiveleave-1 wins: 3567.0 (13.910%)
exhaustiveleave-1 went first: 12854.0 (50.125%)
Player who went first wins: 13670.0 (53.307%)
exhaustiveleave-1 Mean Score: 336.705350  Stdev: 59.403437
exhaustiveleave-2 Mean Score: 449.348932  Stdev: 64.200254
exhaustiveleave-1 Mean Bingos: 1.424076  Stdev: 0.953858
exhaustiveleave-2 Mean Bingos: 2.160505  Stdev: 1.079085

```

```
Level 3 probabilistic (0.35, 0.45) vs Level 4 probabilistic (0.6, 0.7)

Games played: 29734
exhaustiveleave-1 wins: 7949.0 (26.734%)
exhaustiveleave-1 went first: 14953.0 (50.289%)
Player who went first wins: 16262.0 (54.692%)
exhaustiveleave-1 Mean Score: 405.465427  Stdev: 62.861266
exhaustiveleave-2 Mean Score: 472.920966  Stdev: 65.146921
exhaustiveleave-1 Mean Bingos: 2.047858  Stdev: 1.064321
exhaustiveleave-2 Mean Bingos: 2.530706  Stdev: 1.125398

```

```
Level 4 probabilistic (0.6, 0.7) vs HastyBot

Games played: 31661
exhaustiveleave-1 wins: 9209.0 (29.086%)
exhaustiveleave-1 went first: 15784.0 (49.853%)
Player who went first wins: 17444.0 (55.096%)
exhaustiveleave-1 Mean Score: 443.892360  Stdev: 64.548043
exhaustiveleave-2 Mean Score: 504.733837  Stdev: 67.013928
exhaustiveleave-1 Mean Bingos: 2.425824  Stdev: 1.109470
exhaustiveleave-2 Mean Bingos: 2.876030  Stdev: 1.152525


```

### After adding `longWordFindability` rerun all experiments 😩

Use command like this:

`./bin/shell autoplay -botcode1 HASTY_BOT -botcode2 HASTY_BOT -numgames 20000 -block true && ./bin/shell autoanalyze /tmp/games-autoplay.txt`

Bot configs:

```go
	pb.BotRequest_LEVEL1_CEL_BOT: {baseFindability: 0.45, longWordFindability: 0.1, parallelFindability: 0.3, isCel: true},
	pb.BotRequest_LEVEL2_CEL_BOT: {baseFindability: 0.7, longWordFindability: 0.4, parallelFindability: 0.5, isCel: true},
	pb.BotRequest_LEVEL3_CEL_BOT: {baseFindability: 0.8, longWordFindability: 0.5, parallelFindability: 0.75, isCel: true},
	pb.BotRequest_LEVEL4_CEL_BOT: {isCel: true},

	pb.BotRequest_LEVEL1_PROBABILISTIC: {baseFindability: 0.3, longWordFindability: 0.07, parallelFindability: 0.15, isCel: false},
	pb.BotRequest_LEVEL2_PROBABILISTIC: {baseFindability: 0.4, longWordFindability: 0.2, parallelFindability: 0.3, isCel: false},
	pb.BotRequest_LEVEL3_PROBABILISTIC: {baseFindability: 0.55, longWordFindability: 0.35, parallelFindability: 0.45, isCel: false},
	pb.BotRequest_LEVEL4_PROBABILISTIC: {baseFindability: 0.85, longWordFindability: 0.45, parallelFindability: 0.85, isCel: false},
	pb.BotRequest_LEVEL5_PROBABILISTIC: {baseFindability: 0.9, longWordFindability: 0.8, parallelFindability: 0.85, isCel: false},

```

```
CEL 1 vs CEL 1

Games played: 20000
exhaustiveleave-1 wins: 9969.5 (49.847%)
exhaustiveleave-1 Mean Score: 300.7273  Stdev: 41.4898
exhaustiveleave-2 Mean Score: 301.4013  Stdev: 41.3824
exhaustiveleave-1 Mean Bingos: 0.3349  Stdev: 0.5138
exhaustiveleave-2 Mean Bingos: 0.3449  Stdev: 0.5214
exhaustiveleave-1 Mean Points Per Turn: 17.9225  Stdev: 3.2871
exhaustiveleave-1 Mean Points Per Turn: 17.9575  Stdev: 3.2804
exhaustiveleave-1 went first: 10013.0 (50.065%)
Player who went first wins: 11257.5 (56.288%)
```

```
HASTY vs CEL 1

Games played: 20000
exhaustiveleave-1 wins: 19987.0 (99.935%)
exhaustiveleave-1 Mean Score: 510.4562  Stdev: 62.4128
exhaustiveleave-2 Mean Score: 247.3614  Stdev: 39.0748
exhaustiveleave-1 Mean Bingos: 2.3444  Stdev: 1.0625
exhaustiveleave-2 Mean Bingos: 0.3051  Stdev: 0.4971
exhaustiveleave-1 Mean Points Per Turn: 37.7261  Stdev: 5.8351
exhaustiveleave-1 Mean Points Per Turn: 18.5711  Stdev: 3.7053
exhaustiveleave-1 went first: 9958.0 (49.790%)
Player who went first wins: 9963.0 (49.815%)

```

```
CEL 1 vs CEL 2

Games played: 20000
exhaustiveleave-1 wins: 3297.0 (16.485%)
exhaustiveleave-1 Mean Score: 284.0946  Stdev: 40.5773
exhaustiveleave-2 Mean Score: 358.4997  Stdev: 48.8146
exhaustiveleave-1 Mean Bingos: 0.3345  Stdev: 0.5110
exhaustiveleave-2 Mean Bingos: 0.9183  Stdev: 0.7495
exhaustiveleave-1 Mean Points Per Turn: 18.1619  Stdev: 3.3805
exhaustiveleave-1 Mean Points Per Turn: 22.8416  Stdev: 4.0532
exhaustiveleave-1 went first: 10100.0 (50.500%)
Player who went first wins: 10694.0 (53.470%)

```

```
Level 2 CEL Bot vs Level 4 CEL Bot


Games played: 20000
exhaustiveleave-1 wins: 3533.5 (17.668%)
exhaustiveleave-1 Mean Score: 319.6785  Stdev: 46.9307
exhaustiveleave-2 Mean Score: 400.4173  Stdev: 55.8149
exhaustiveleave-1 Mean Bingos: 0.8449  Stdev: 0.7252
exhaustiveleave-2 Mean Bingos: 1.4369  Stdev: 0.9149
exhaustiveleave-1 Mean Points Per Turn: 23.2010  Stdev: 4.3025
exhaustiveleave-1 Mean Points Per Turn: 28.9344  Stdev: 5.0025
exhaustiveleave-1 went first: 9987.0 (49.935%)
Player who went first wins: 10781.5 (53.907%)


```

```
Level 4 CEL Bot vs Level 4 Prob bot

Games played: 20000
exhaustiveleave-1 wins: 4857.5 (24.288%)
exhaustiveleave-1 Mean Score: 361.3573  Stdev: 54.0872
exhaustiveleave-2 Mean Score: 424.7307  Stdev: 54.6320
exhaustiveleave-1 Mean Bingos: 1.3255  Stdev: 0.8809
exhaustiveleave-2 Mean Bingos: 1.6058  Stdev: 0.9369
exhaustiveleave-1 Mean Points Per Turn: 28.9520  Stdev: 5.2767
exhaustiveleave-1 Mean Points Per Turn: 33.8321  Stdev: 5.3244
exhaustiveleave-1 went first: 10097.0 (50.485%)
Player who went first wins: 10933.5 (54.667%)




```

```
Level 4 Prob bot vs Hasty Bot

Games played: 20000
exhaustiveleave-1 wins: 5084.5 (25.422%)
exhaustiveleave-1 Mean Score: 405.7336  Stdev: 55.1540
exhaustiveleave-2 Mean Score: 470.9975  Stdev: 60.8158
exhaustiveleave-1 Mean Bingos: 1.6935  Stdev: 0.9724
exhaustiveleave-2 Mean Bingos: 2.3356  Stdev: 1.0824
exhaustiveleave-1 Mean Points Per Turn: 35.9524  Stdev: 5.8806
exhaustiveleave-1 Mean Points Per Turn: 41.5876  Stdev: 6.6473

```

Using FRA20 lexicon:

```
Hasty v Hasty

Games played: 20000
exhaustiveleave-1 wins: 10044.0 (50.220%)
exhaustiveleave-1 Mean Score: 490.8259  Stdev: 67.1289
exhaustiveleave-2 Mean Score: 490.7139  Stdev: 67.6279
exhaustiveleave-1 Mean Bingos: 2.8243  Stdev: 1.1581
exhaustiveleave-2 Mean Bingos: 2.8209  Stdev: 1.1684
exhaustiveleave-1 Mean Points Per Turn: 43.3770  Stdev: 7.3999
exhaustiveleave-1 Mean Points Per Turn: 43.3694  Stdev: 7.4864
exhaustiveleave-1 went first: 9984.0 (49.920%)
Player who went first wins: 11048.0 (55.240%)
```

```
Level 1 Prob bot vs Hasty Bot

Games played: 20000
exhaustiveleave-1 wins: 44.0 (0.220%)
exhaustiveleave-1 Mean Score: 285.8092  Stdev: 51.4911
exhaustiveleave-2 Mean Score: 577.6897  Stdev: 72.6677
exhaustiveleave-1 Mean Bingos: 0.6857  Stdev: 0.6941
exhaustiveleave-2 Mean Bingos: 3.2911  Stdev: 1.1890
exhaustiveleave-1 Mean Points Per Turn: 21.1958  Stdev: 5.0523
exhaustiveleave-1 Mean Points Per Turn: 42.0990  Stdev: 6.7054
exhaustiveleave-1 went first: 10078.0 (50.390%)
Player who went first wins: 9942.0 (49.710%)
```

```
Level 2 Prob vs Level 1 Prob

Games played: 20000
exhaustiveleave-1 wins: 16171.0 (80.855%)
exhaustiveleave-1 Mean Score: 410.0470  Stdev: 60.6334
exhaustiveleave-2 Mean Score: 322.3127  Stdev: 54.0413
exhaustiveleave-1 Mean Bingos: 1.4424  Stdev: 0.9106
exhaustiveleave-2 Mean Bingos: 0.7311  Stdev: 0.7166
exhaustiveleave-1 Mean Points Per Turn: 25.1007  Stdev: 5.0271
exhaustiveleave-1 Mean Points Per Turn: 19.8481  Stdev: 4.6158
exhaustiveleave-1 went first: 10114.0 (50.570%)
Player who went first wins: 10840.0 (54.200%)

```

```
Level 3 Prob vs Level 2 Prob

Games played: 20000
exhaustiveleave-1 wins: 15151.5 (75.757%)
exhaustiveleave-1 Mean Score: 437.9868  Stdev: 61.6071
exhaustiveleave-2 Mean Score: 367.7466  Stdev: 56.6542
exhaustiveleave-1 Mean Bingos: 1.9421  Stdev: 1.0161
exhaustiveleave-2 Mean Bingos: 1.3334  Stdev: 0.8938
exhaustiveleave-1 Mean Points Per Turn: 30.9208  Stdev: 5.6962
exhaustiveleave-1 Mean Points Per Turn: 26.0925  Stdev: 5.3202
exhaustiveleave-1 went first: 10077.0 (50.385%)
Player who went first wins: 11018.5 (55.093%)


```

```
Level 4 Prob vs Level 3 Prob

Games played: 20000
exhaustiveleave-1 wins: 15524.5 (77.623%)
exhaustiveleave-1 Mean Score: 480.8806  Stdev: 63.3233
exhaustiveleave-2 Mean Score: 400.9642  Stdev: 59.2692
exhaustiveleave-1 Mean Bingos: 2.3977  Stdev: 1.0902
exhaustiveleave-2 Mean Bingos: 1.8081  Stdev: 0.9993
exhaustiveleave-1 Mean Points Per Turn: 37.7178  Stdev: 6.2981
exhaustiveleave-1 Mean Points Per Turn: 31.6303  Stdev: 6.0130
exhaustiveleave-1 went first: 10051.0 (50.255%)
Player who went first wins: 10946.5 (54.733%)

```

```
HastyBot vs Level 4 Prob

exhaustiveleave-1 wins: 14037.0 (70.185%)
exhaustiveleave-1 Mean Score: 505.2295  Stdev: 67.1159
exhaustiveleave-2 Mean Score: 447.5797  Stdev: 61.8292
exhaustiveleave-1 Mean Bingos: 2.9086  Stdev: 1.1555
exhaustiveleave-2 Mean Bingos: 2.2509  Stdev: 1.0689
exhaustiveleave-1 Mean Points Per Turn: 43.0800  Stdev: 7.2696
exhaustiveleave-1 Mean Points Per Turn: 38.2625  Stdev: 6.6482
exhaustiveleave-1 went first: 9973.0 (49.865%)
Player who went first wins: 11041.0 (55.205%)

```

One more experiment, lowered level1 cel bot to
`{baseFindability: 0.3, longWordFindability: 0.1, parallelFindability: 0.3, isCel: true}`

```
Level 1 CEL vs HastyBot

Games played: 20000
exhaustiveleave-1 wins: 1.0 (0.005%)
exhaustiveleave-1 Mean Score: 225.9808  Stdev: 36.8490
exhaustiveleave-2 Mean Score: 553.3431  Stdev: 63.8713
exhaustiveleave-1 Mean Bingos: 0.2276  Stdev: 0.4396
exhaustiveleave-2 Mean Bingos: 2.6511  Stdev: 1.1092
exhaustiveleave-1 Mean Points Per Turn: 17.1924  Stdev: 3.4998
exhaustiveleave-1 Mean Points Per Turn: 41.1971  Stdev: 6.1402
exhaustiveleave-1 went first: 10124.0 (50.620%)
Player who went first wins: 9877.0 (49.385%)

```

and vs Level 2:

```
Level 1 CEL vs Level 2 CEL

Games played: 20000
exhaustiveleave-1 wins: 1763.0 (8.815%)
exhaustiveleave-1 Mean Score: 265.6067  Stdev: 38.5800
exhaustiveleave-2 Mean Score: 367.2212  Stdev: 49.1819
exhaustiveleave-1 Mean Bingos: 0.2497  Stdev: 0.4581
exhaustiveleave-2 Mean Bingos: 0.9165  Stdev: 0.7537
exhaustiveleave-1 Mean Points Per Turn: 16.6848  Stdev: 3.1712
exhaustiveleave-1 Mean Points Per Turn: 22.8887  Stdev: 4.0306
exhaustiveleave-1 went first: 9898.0 (49.490%)
Player who went first wins: 10696.0 (53.480%)
```
