# Iterative measure–impute–recalibrate inference

Supersedes `two_stage_inference_plan.md`, which predates cross-fit calibration
and patched the selection problem with prior reweighting; this design replaces
that with proper importance weights.

## Context

Tile-placement inference measures $P(\text{play} \mid L)$ for a few hundred
leaves out of thousands and imputes the rest from marginal lifts
(`docs/leave_imputation.md`). Two known weaknesses:

1. **Shape error, never checked.** Low-order lifts miss synergies. ZIT is a
   strong leave, but if it never appears in sampling, `lift(I)` is diluted
   across mediocre I-leaves and imputed ZIT lands far too low. Nothing in the
   pipeline ever compares an imputed value against a real evaluation.
2. **Effort ignores mass.** Every draw comes from the prior, so evaluation is
   spread over the whole space rather than concentrated where the posterior
   says it matters.

Fix: alternate measurement and imputation. Evaluate a batch, fit the model,
impute, draw the next batch using the imputed posterior — so leaves the model
thinks are good get formally evaluated instead of taken on faith — and repeat
until the model stops being wrong or the budget runs out.

## 1. The estimator, generalized

Once the proposal stops being the prior, the current lift estimator breaks.
`lift(S) = (wsum(S)/cnt(S)) / (wTotal/n)` is valid only because stage-1 draws
are i.i.d. from the prior: sampling multiplicity *is* the prior weight, so
counting each draw once gives the Hansen–Hurwitz estimator of the
prior-weighted containment marginal. Draw from anything else and the counts
stop tracking $P$.

The estimand is a property of the leave space, not of the sampler:

```math
\mathrm{lift}(S) \;=\;
\frac{\sum_{L \supseteq S} P(L) w(L) \big/ \sum_{L \supseteq S} P(L)}
     {\sum_{L} P(L) w(L) \big/ \sum_{L} P(L)}
```

Estimate it with per-draw importance weights. Draw $j$ comes from proposal
$q_{r(j)}$; give it $u_j = P(L_j)/q_{r(j)}(L_j)$ and accumulate weighted sums:

```math
\mathrm{lift}(S) = \frac{\sum_{j:\,S\subseteq L_j} u_j w_j \big/ \sum_{j:\,S\subseteq L_j} u_j}
                        {\sum_j u_j w_j \big/ \sum_j u_j}
```

Unbiased under **any** proposal sequence — posterior-guided, uncertainty-guided,
or MCMC. Because it is a ratio, $q$ need only be known up to a constant, which
is what admits samplers whose stationary density is unnormalized.

Shrinkage keys off effective sample size, the right notion of support once
draws carry unequal weight:

```math
\mathrm{ESS}(S) = \frac{\left(\sum_{j:\,S\subseteq L_j} u_j\right)^2}
                       {\sum_{j:\,S\subseteq L_j} u_j^2}
```

**Exactly backward compatible.** With $q = P$ every $u_j = 1$, so $\sum u =
\mathrm{cnt}$, $\sum u^2 = \mathrm{cnt}$, $\mathrm{ESS} = \mathrm{cnt}$, and
every formula collapses to today's. Round 0 behaviour is unchanged.

Calibration gets the same treatment: the moment condition becomes $\sum_L U_L
e^{C + \sum\varphi^{(-f(L))}(L)} = \sum_L U_L \bar w(L)$ with $U_L = \sum_{j:
L_j = L} u_j$ replacing the draw count $c_L$. Cross-fitting
(`docs/leave_imputation.md` §5) is orthogonal and carries over untouched.

## 2. The proposal: exploit by weight, explore by uncertainty

Round $r$ draws from the unmeasured leaves $U_r$ with

```math
q_r(L) \;\propto\; W_r(L)\,\bigl(1 + \lambda_{\text{ex}}\,\mathrm{unc}(L)\bigr),
\qquad L \in U_r, \quad \lambda_{\text{ex}} = 1
```

where $W_r(L) = P(L)\hat\ell_r(L)$ is the current posterior weight and
$\mathrm{unc}(L)$ is the model's own uncertainty about it. Weight alone would
be pure exploitation; the bonus is the UCB term.

**Uncertainty comes free from the shrinkage.** A term estimated from thin
support is multiplied by $\mathrm{ESS}/(\mathrm{ESS}+\lambda)$; the part
discarded is precisely what we do not know:

```math
\mathrm{unc}(S) =
\begin{cases}
\left(1 - \frac{\mathrm{ESS}(S)}{\mathrm{ESS}(S)+\lambda}\right)\bigl|\log \mathrm{lift}(S)\bigr| & \mathrm{ESS}(S) > 0\\[4pt]
\sigma_0(|S|) & \mathrm{ESS}(S) = 0
\end{cases}
\qquad
\mathrm{unc}(L) = \sqrt{\sum_{S \subseteq L} \mathrm{unc}(S)^2}
```

$\sigma_0(o)$ is the RMS of $|\log\mathrm{lift}|$ over *observed* cells of
order $o$: a never-seen sub-multiset is the most uncertain thing in the model,
not the least, so it inherits the typical magnitude at its order rather than
$\varphi = 0$. Computed once per model build into `unc1/unc2/unc3`, walked by
the same recursion as `logImputed`.

Why not a defensive prior mixture (the textbook choice):

- **Round 0 is already the prior.** Mixing $P$ back in re-samples the region
  stage 0 covered hardest — common-tile leaves, where support is densest and
  the model is already confident.
- **It misses the motivating bug.** ZIT is mispriced because the (Z,I) pair
  has thin support. The uncertainty bonus targets that directly; prior mixing
  would need luck, and Z's low prior makes that luck rare.
- **It anneals for free.** As ESS grows the bonus decays and the loop slides
  from exploration to exploitation with no schedule.
- **Positivity does not need it.** $W_r(L) > 0$ for every feasible leave, so
  every leave keeps a positive draw probability regardless.

The prior mixture's real job was bounding $u_j$; §2.2 handles that directly.

### 2.1 Systematic PPS sampling

Turn $q_r$ into $m_r$ draws by building its CDF and taking $m_r$ equally
spaced positions from one uniform offset. Each leave is drawn $\lfloor m_r q_r
\rfloor$ or $\lceil m_r q_r \rceil$ times, so $\mathbb{E}[n_L] = m_r q_r(L)$
exactly — unbiased with the same $u_j$, far lower variance than i.i.d. draws,
no accidental duplicates, and everything with $q_r \ge 1/m_r$ is measured with
certainty.

A leave drawn $n_L > 1$ times is evaluated once with its multiplicity folded
into the weight ($n_L u_j$) rather than re-simmed. That is algebraically the
same estimator with a slightly higher variance on $w$, and it buys coverage
instead of precision. The alternative — actually running $n_L$ independent
mini-sims and averaging — is worth revisiting given how noisy a single
likelihood is (at $\tau = 0.05$ a ±0.03 win-probability error is ±0.12 in
log-odds, which the softmax turns into a factor of $e^{0.12/0.05} \approx 11$),
but duplicates require a leave holding $\ge 1/m_r$ of the proposal mass and are
rare over thousands of candidates.

### 2.2 Weight stability

$u_j = P(L_j)/q_r(L_j)$ is unbounded only as $\hat\ell \to 0$, and such leaves
are drawn with correspondingly tiny probability — the classic heavy-tail risk.
Note the uncertainty bonus sits in the denominator, so exploratory draws
already carry *smaller* weights. On top of that, truncate per round at
$u_{\max} = \sqrt{m_r}\,\bar u$ (Ionides). Report the Pareto tail index
$\hat k$ of the round's weights as a model-adequacy diagnostic: $\hat k > 0.7$
means the proposal is a poor match for the target and the round's evidence
should be treated as weak.

## 3. The loop

```
round 0 (explore):  m_0 i.i.d. prior draws         — existing MC path, q_0 = P, u_j = 1
repeat r = 1 .. R_max:
    fit lifts from all draws (importance weighted)
    cross-fit calibration constant C_r
    assemble posterior: measured → prior × w̄, unmeasured → prior × ℓ̂
    q_r ← W_r · (1 + λ_ex · unc) over unmeasured leaves
    draw m_r leaves by systematic PPS; evaluate in parallel
    record draws with u_j = P/q_r (truncated)
    stop if any criterion below fires
final: refit, recalibrate, assemble
```

Every round ends with a full refit, so a discovery in round 2 (Z+I leaves
measure strong) raises the (Z,I) pair lift and pulls ZIT-adjacent leaves
forward in round 3. That feedback is the reason to iterate rather than run one
large posterior-guided batch.

### Stopping criteria

Whichever fires first:

| Criterion | Test | Default |
|---|---|---|
| Max rounds | $r = R_{\max}$ | 6 |
| Calibration converged | miscalibration below noise (see below) | — |
| Space exhausted | $U_r = \varnothing$ | — |
| Mass covered | unmeasured posterior mass $< \varepsilon_{\text{mass}}$ | 0.02 |
| Budget | context deadline | `-time` |

**The calibration criterion.** Before round $r$ is evaluated, the model
predicts $\hat\ell_{r-1}(L_j)$ for each drawn leave — all previously
unmeasured, so these are genuinely out-of-sample. After evaluation, the
realized scale error is

```math
\hat R_r = \frac{\sum_j u_j w_j}{\sum_j u_j \hat\ell_{r-1}(L_j)}
```

the self-normalized importance-sampling estimate of the correction the imputed
set still needs. $\hat R_r = 1$ means the model was right about the mass it
was pointing at. Stop when the miscalibration is no longer statistically
detectable:

```math
\left|\log \hat R_r\right| \;<\; \max\!\left(\varepsilon_{\text{floor}},\; 2\,\mathrm{SE}(\log \hat R_r)\right),
\qquad \varepsilon_{\text{floor}} = 0.05
```

using the linearized ratio-estimator variance

```math
\widehat{\mathrm{Var}}(\hat R_r) = \frac{\sum_j u_j^2 \bigl(w_j - \hat R_r \hat\ell_{r-1}(L_j)\bigr)^2}
                                        {\bigl(\sum_j u_j \hat\ell_{r-1}(L_j)\bigr)^2},
\qquad \mathrm{SE}(\log \hat R_r) = \frac{\mathrm{SE}(\hat R_r)}{\hat R_r}
```

The $2\,\mathrm{SE}$ term self-calibrates against the mini-sim noise floor:
with noisy $w$ and a small batch the SE is large and the loop stops rather
than chasing measurement noise; with a large batch it keeps going until the
residual bias is genuinely small. $\varepsilon_{\text{floor}}$ stops a very
large batch from chasing a 1% effect. $\log \hat R_r$, its SE, and $\hat k$
are logged per round, so convergence is visible.

### Budget

Round 0 takes `stage0Frac` (default 0.4) of the time budget; each later round
gets an equal share of the remainder divided by the rounds left, so early
termination hands its time back. Evaluations cost the same ~100 ms mini-sim in
every round — the total `-time` is unchanged, only its allocation moves.

## 4. Changes

### `rangefinder/imputation.go` — weighted accumulator
- `subleaveAccumulator` stores `sumU`, `sumU2`, `sumUW` per sub-multiset
  (orders 1–3) plus totals, replacing `cnt`/`wsum`. `record(sorted, w, u)`
  takes the draw's importance weight.
- `buildImputationModel`: `rawLogLift = log((sumUW/sumU)/(totalUW/totalU))`,
  shrinkage on `ESS = sumU²/sumU2`; Möbius construction, clamps and
  `logImputed` unchanged. Also fills `unc1/unc2/unc3` and $\sigma_0$ per order.
- `imputationModel.uncertainty(runs)` mirrors `logImputed`'s walk.
- `measuredLeave` gains `sumU`; `calibrateLogConstant` weights by it instead
  of `count`. `mean()` stays an unweighted mean over replicates — repeated
  measurements of one leave are equally noisy replicates however they were
  selected.
- `minus`, `foldForKey`, cross-fitting: unchanged.
- `accStats` returns ESS alongside the weighted sums for the walkthrough.

### `rangefinder/refine.go` (new)
- `evaluateLeaves(ctx, leaves, onResult)` — parallel mini-sim evaluator
  extracted from `inferEnumerated` (`enumerate.go:183-240`), fanning out over
  `gameCopies`/`aiplayers` with an atomic index; `inferEnumerated` becomes a
  caller.
- `buildProposal(res, bagMap, lambdaEx)` — weight × uncertainty bonus over
  unmeasured leaves; returns leaves, $q$, and $u = P/q$ truncated.
- `systematicSample(q, m, rng)` — PPS draws.
- `refineRounds(ctx, maxRounds)` — the loop above, logging batch size,
  $\log\hat R$, SE, $\hat k$, and remaining unmeasured mass per round.

### `rangefinder/inference.go`
- `Infer` MC branch: round 0 under a `stage0Frac` sub-deadline, then
  `refineRounds`, then a final `finalizePlacementPosterior`.
- `recordPlacementSample(leave, w, u)`; round 0 passes $u = 1$.
- New fields and setters: `maxRounds` (`SetMaxRounds`, default 6),
  `stage0Frac`, `refinedCount`, `roundLog`.
- `inferEnumerated`: when the enumeration was truncated or timed out, fall
  through to `refineRounds` for the remainder.

### `rangefinder/stats.go`
- Complete-posterior header gains rounds run and refined-leaf count.
- `AnalyzeLeave` shows `measured ×n` for refined leaves automatically; the
  walkthrough's `n` column becomes ESS.

### Shell
- `infer` spec: `-rounds N` (0 disables refinement, restoring single-stage
  behaviour). `shell/api.go` wires it; `shell/helptext/infer.txt` documents
  the loop.

### Unchanged
- `montecarlo/` (posterior shape identical), `ai/bot/elite.go`,
  `cmd/inferdiag` — all inherit the loop through `Infer`.
- Exchange inference (no imputation there).

## 5. Verification

| Claim | Test |
|---|---|
| Weighted estimator reduces to today's at $u \equiv 1$ | `TestUnitWeightsMatchCounts` |
| Lift unbiased under a non-prior proposal | `TestWeightedLiftUnbiasedUnderSkewedProposal` |
| Systematic PPS hits $\mathbb{E}[n_L] = m\,q(L)$ | `TestSystematicSampleExpectation` |
| Uncertainty is highest for never-observed sub-multisets | `TestUncertaintyFavorsThinSupport` |
| Truncation bounds the round's weights | `TestProposalWeightsTruncated` |
| Loop converges and stops on the ratio criterion | `TestRefineRoundsConverge` |
| Existing imputation/calibration invariants | unchanged suite |

## 6. Deliberately not doing

- **Shrinking measured values toward the model.** Given the noise floor, a
  leave measured *once* may be estimated worse by its own measurement than by
  $\hat\ell$; the principled fix is a precision-weighted blend of $\log \bar w$
  and $\log \hat\ell$, which needs a variance model for the mini-sim
  likelihood. PPS replicates on high-mass leaves mitigate the worst of it.
- **Balance-heuristic (MIS) weights across rounds**, $u_j = P/\bar q$ with
  $\bar q = \sum_r m_r q_r / \sum_r m_r$. Lower variance than per-round
  weights, but needs a per-leaf running mixture density across rounds;
  truncation already bounds the main risk.
- **Horvitz–Thompson inclusion-probability weighting.** Equivalent target;
  per-draw weights are simpler and compose across rounds without tracking
  inclusion probabilities.
