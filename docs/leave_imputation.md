# Marginal-Lift Imputation: the Complete Leave Posterior

`rangefinder/imputation.go`, `rangefinder/enumerate.go`, `rangefinder/stats.go`

## 1. The problem

Tile-placement inference asks: *given the play our opponent just made, what
tiles are they likely holding?* Bayes' rule gives the posterior over leaves
(the tiles kept after the play):

```math
P(L \mid \text{play}) \;\propto\; P(L) \cdot P(\text{play} \mid L)
```

- **Prior** $P(L)$: the multivariate hypergeometric probability of drawing
  exactly the multiset $L$ from the unseen pool
  (`combinatorialPrior`, rangefinder/inference.go).
- **Likelihood** $P(\text{play} \mid L)$: for a candidate leave, reconstruct
  the opponent's full rack, generate their moves, and softmax the resulting
  equities with temperature $\tau$; the likelihood is the softmax mass on the
  play actually observed. Lower $\tau$ models a more optimal opponent.

The likelihood is expensive — each evaluation is a movegen + equity pass (or a
mini-sim). Within the time budget we evaluate ("measure") only a subset of the
feasible leaves: typically a few hundred out of thousands. Historically the
simmer bridged the gap with an ad-hoc sigmoid gate blending inferred racks
with uniform random draws. Imputation replaces that: we build **one complete
posterior over every feasible leave**, where measured leaves keep their
evaluated likelihood and every unmeasured leave gets a likelihood *imputed*
from low-order containment statistics of the measured set.

## 2. Containment lifts

Every evaluated sample $j$ contributes its leave $L_j$, its measured
likelihood $w_j \ge 0$ (zero is real evidence — "this leave never produces the
observed play" — and is recorded, not skipped), and an **importance weight**
$u_j = P(L_j)/q(L_j)$, where $q$ is the proposal that produced the draw.

For a small sub-multiset $S$ (a single tile, a pair, or a triple — possibly
with repeats, like $\{A,A\}$), define its **containment lift**:

```math
\mathrm{lift}(S) \;=\; \frac{\mathbb{E}[\,w \mid S \subseteq L\,]}{\mathbb{E}[w]}
\;=\; \frac{\sum_{j:\,S \subseteq L_j} u_j w_j \big/ \sum_{j:\,S \subseteq L_j} u_j}
           {\sum_j u_j w_j \big/ \sum_j u_j}
```

summing over evaluated samples whose leave contains $S$ (each *distinct*
sub-multiset counted once per sample; `subleaveAccumulator.record`). Numerator
and denominator share the same sample set, so selection noise partially
cancels.

**Why the weights.** Today every draw comes from the prior
(`SetRandomRack`), so sampling multiplicity *is* prior weight and every
$u_j = 1$: the sums degenerate to a plain count and likelihood-sum, and this
is the estimator it has always been. That identity is exactly what breaks if
leaves are ever drawn from anything else — counting draws equally would then
estimate a lift under the *proposal* rather than the prior. Weighting each
draw by $P/q$ makes the estimator target the same population quantity under
any proposal, including MCMC-style samplers, since the lift is a ratio and $q$
need only be known up to a constant. Shrinkage correspondingly keys off
effective sample size $\mathrm{ESS}(S) = (\sum u)^2/\sum u^2$, which is again
exactly the observation count when all $u_j = 1$
(`TestUnitWeightsMatchCounts`, `TestWeightedLiftUnbiasedUnderSkewedProposal`).

Intuition: $\mathrm{lift}(Z) = 28$ means "evaluated leaves containing Z
produced the observed play 28× more often than average" — Z is strongly
implicated by the play.

The maximum order of tracked sub-multisets is $m = \lceil k/2 \rceil$ capped
at 3, where $k$ is the leave size (`marginalOrder`). For a 3-tile leave that
means singles and pairs; for 5+ tiles, singles, pairs, and triples.

## 3. Combining lifts: Möbius inversion on the divisor lattice

The naive estimate $\log \hat\ell(L) = \sum_{t \in L} \log\mathrm{lift}(t)$
double-counts: if I and T are *individually* fine but *jointly* poison
(because the observed play would have used one of them), single-tile lifts
can't see it. We want a truncated log-linear expansion where each order adds
only the **new** information not already explained by its sub-parts.

Sub-multisets of a multiset ordered by inclusion form a **divisor lattice**
(map each tile to a distinct prime; a sub-multiset is a divisor). Interaction
terms $\varphi$ are defined bottom-up by Möbius inversion on that lattice:

```math
\varphi(S) \;=\; \log \mathrm{lift}(S) \;-\; \sum_{\varnothing \ne T \subsetneq S} \varphi(T)
```

For all-distinct tiles this reduces to familiar inclusion–exclusion:

$$
\varphi(\{a\}) = \log\mathrm{lift}(a), \qquad
\varphi(\{a,b\}) = \log\mathrm{lift}(ab) - \varphi(a) - \varphi(b)
$$

but the divisor lattice handles repeats correctly, which subset-lattice
inclusion–exclusion gets wrong. The proper sub-multisets of $\{A,A\}$ are just
$\{A\}$ — the Möbius function of the divisor lattice is zero whenever any
multiplicity gap is ≥ 2, so the empty set gets coefficient 0 here, not +1:

$$
\varphi(\{A,A\}) = \log\mathrm{lift}(AA) - \varphi(A)
$$

The imputed log-likelihood of a full leave $L$ (up to the calibration
constant of §5) is the sum of $\varphi$ over all its distinct sub-multisets up
to order $m$:

```math
\log \hat\ell(L) \;=\; C + \sum_{\substack{S \subseteq L \\ 1 \le |S| \le m}} \varphi(S)
```

**Telescoping identity.** When $m = |L|$ and no shrinkage or clamping is
applied, the sum telescopes exactly:
$\sum_{S \subseteq L} \varphi(S) = \log\mathrm{lift}(L)$ — the expansion
reproduces the full-leave lift whenever it is directly estimable
(`TestMobiusTelescoping`). Truncating at order $m < |L|$ is precisely the
approximation "interactions of order $> m$ are negligible."

## 4. Regularization

Raw lifts from thin support are noisy, so three guards apply
(`buildImputationModel`):

| Guard | Value | Effect |
|---|---|---|
| Shrinkage $\frac{c}{c+\lambda}$, $\lambda = 10$ | applied to every $\varphi$ | a term seen $c$ times decays toward 0 (lift 1 / no interaction) instead of adding variance; $c=10$ keeps half the signal |
| Log-lift clamp $\pm\log 10^6 \approx \pm 13.8$ | order-1 terms and raw lifts | a sub-multiset whose every containing sample had $w=0$ would otherwise be $-\infty$; it clamps low (strong but finite negative evidence) |
| Interaction clamp $\pm\log 10$ | order-2/3 terms, pre-shrinkage | one pair/triple can adjust the estimate by at most 10× beyond what its parts explain |

A sub-multiset never observed at all ($c = 0$) keeps $\varphi = 0$: no
evidence, no adjustment.

## 5. Calibration: moment matching

$\sum \varphi$ lives on the *lift* scale (relative to the average evaluated
leave); measured likelihoods live on the raw softmax scale. A constant $C$
aligns them. We choose $C$ by **moment matching over the measured leaves**
(count-weighted, $w > 0$ only):

```math
\sum_i c_i \, e^{\,C + \sum\varphi(L_i)} \;=\; \sum_i c_i w_i
\qquad\Longrightarrow\qquad
C \;=\; \log\frac{\sum_i c_i w_i}{\sum_i c_i e^{\sum \varphi(L_i)}}
```

where $c_i$ is the leave's total importance weight $U_L = \sum_{j: L_j = L}
u_j$ — its draw count while everything is prior-sampled. Computed with
log-sum-exp for stability. Under this calibration the model reproduces the
**arithmetic** mean of the measured likelihoods exactly
(`TestCalibrationMomentMatched`).

### Why not mean-of-logs (historical bug)

The first implementation used
$C = \mathrm{mean}_i \left(\log w_i - \sum\varphi(L_i)\right)$ —
an anchor at the **geometric** mean. Bayes' rule consumes likelihoods
linearly, so the arithmetic scale is the right one, and by Jensen's
inequality $\mathbb{E}[\log w] \le \log \mathbb{E}[w]$, with the gap growing
as $\sigma^2/2$ for log-scale variance $\sigma^2$. Softmax likelihoods at low
$\tau$ are extremely right-skewed (most leaves are near-impossible, a few are
very likely), so the gap is enormous in practice: in one real position the
mean-of-logs constant was $-12.51$ against $\log \bar w = -5.42$ — a factor
of ~1200. Since $C$ is shared by every imputed leaf, this didn't disturb
rankings *among* imputed leaves, but it silently starved **all** of them of
posterior mass relative to measured leaves (which enter with their raw
arithmetic $w$), badly inflating the "measured mass" diagnostic and burying
strong unmeasured leaves. Moment matching removes the bias by construction.

## 6. Assembling the posterior

`imputeFullPosterior` enumerates every feasible leave $L$ of size $k$
drawable from the unseen pool (parallel recursive enumeration, prior computed
incrementally in log space) and assigns:

```math
W(L) \;\propto\; P(L) \times
\begin{cases}
\overline{w}(L) & L \text{ measured, } \overline{w}(L) > 0 \\[2pt]
\text{(excluded)} & L \text{ measured, } \overline{w}(L) = 0 \\[2pt]
e^{\,C + \sum \varphi(S)} & L \text{ unmeasured}
\end{cases}
```

where $\overline{w}(L)$ is the mean over repeat evaluations of the same
leave. Weights are normalized so the max is 1; leaves below $10^{-15}$ of the
max are dropped (bounded total discarded mass ~$10^{-9}$). The simmer
(`montecarlo.InferenceCompletePosterior`) then samples opponent leaves
directly from this posterior via CDF binary search — no fallback-to-random
gate.

If there is no usable signal at all ($n = 0$ or $\sum w = 0$), every
$\varphi$ is 0 and the posterior degrades gracefully to the prior
(`TestImputeNoSignalIsPrior`).

## 7. Worked example

Real output for a 3-tile leave (order $m = 2$: singles + pairs), via
`infer leave ITZ`:

```
  sub   n       E[lik|sub]   lift       shrink   φ          e^φ
  I     570     0.003499     0.7902     0.983    -0.2314    0.7935
  T     366     0.001496     0.3378     0.973    -1.056     0.3477
  Z     72      0.1243       28.08      0.878    +2.928     18.69
  IT    72      4.206e-05    0.009499   0.878    -2.022     0.1324
  IZ    14      0.1419       32.06      0.583    +0.4496    1.568
  TZ    9       0.06007      13.57      0.474    +0.3485    1.417
  Σφ = 0.4168
```

Reading it:

- **Z** is the play's smoking gun: 28× lift, well supported (72 samples), so
  $\varphi(Z) = 0.878 \times \log 28.08 = +2.93$.
- **IZ looks huge (32×) but adds little.** Its interaction subtracts what the
  singles already explain: $\log 32.06 - \varphi(I) - \varphi(Z) = +0.77$,
  shrunk by thin support (14 samples) to $+0.45$. The 32× is mostly "Z is
  great," and Z is only credited once.
- **IT is the hidden killer.** Leaves holding both I and T almost never
  produce the observed play (lift 0.0095). Raw interaction
  $\log 0.0095 - \varphi(I) - \varphi(T) = -3.37$ hits the interaction clamp
  at $-2.30$, shrinks to $-2.02$ — a ~7.5× penalty beyond what I and T
  individually explain.

Net $e^{\Sigma\varphi} = 1.52$: ITZ is estimated ~1.5× as likely as the
average evaluated leave — Z's pull mostly cancelled by the I, T, and IT
evidence. Final weight is
$P(L) \cdot e^{C + \Sigma\varphi}$ normalized by the max.

## 8. Diagnostics

- `infer analyze` (detailed): posterior header — measured/imputed counts,
  measured mass, marginal order, $\tau$, effective sample size — plus the top
  racks with their source (`measured ×n` / `imputed`).
- `infer leave <tiles>`: the full walkthrough above (`AnalyzeLeave` /
  `writeImputationWalkthrough`, rangefinder/stats.go), ending with the exact
  chain `weight = prior × ℓ̂ ÷ max` that reproduces the stored weight — a
  built-in self-check that the display matches the engine
  (`TestImputationResultReconstructsWeights`).

The model, calibration constant, and normalization max are persisted on
`imputationResult` specifically so this replay is exact.

## 9. Parameters and code map

| Parameter | Value | Where |
|---|---|---|
| Marginal order $m$ | $\lceil k/2 \rceil$, cap 3 | `marginalOrder` |
| Shrinkage $\lambda$ | 10 | `imputationLambda` |
| Log-lift clamp | $\pm 13.8$ ($\approx \log 10^6$) | `maxAbsLogLift` |
| Interaction clamp | $\pm 2.303$ ($\log 10$) | `maxAbsInteraction` |
| Negligible-mass cutoff | $10^{-15}$ of max weight | `negligibleWeightFraction` |
| Softmax temperature $\tau$ | configurable, default 0.05 | `SoftmaxTemperature`, `RangeFinder.SetTau` |

| Concern | Code | Tests |
|---|---|---|
| Sub-multiset accumulation | `subleaveAccumulator.record` | `TestAccumulatorDistinctSubMultisets`, `TestUnitWeightsMatchCounts` |
| Importance weighting | `subleaveAccumulator.record` (`u`) | `TestWeightedLiftUnbiasedUnderSkewedProposal` |
| Model uncertainty | `imputationModel.uncertainty` | `TestUncertaintyFavorsThinSupport` |
| Möbius terms | `buildImputationModel` | `TestMobiusTelescoping` |
| Term enumeration for display | `subleaveTerms` | `TestSubleaveTermsSumMatchesLogImputed` |
| Calibration | `imputeFullPosterior` (calibration block) | `TestCalibrationMomentMatched` |
| Posterior assembly | `imputeFullPosterior` | `TestImputeFullPosteriorMeasuredExact`, `TestImputeUnmeasuredLift`, `TestImputeNoSignalIsPrior`, `TestMeasuredZeroExcluded` |
| Weight replay | `AnalyzeLeave` | `TestImputationResultReconstructsWeights` |

Known gap: exchange inference still uses the legacy Monte Carlo sampling path
without a complete posterior; only tile-placement inference is imputed.
