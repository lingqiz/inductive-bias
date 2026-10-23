# Estimating the action-policy KL (`kl.py`)

## OPEN PROBLEM (2026-10-22): build a proper generative model of p(action | stimulus)

**Next step:** think through how to build a real generative model of the movement, starting
with one that describes the **side bias through the movement itself**. The choice term
correlates with φ far more strongly than anything else (log-Pearson r = 0.82). So the first
job of a movement model is to produce p(choice | stimulus) from the trajectory, not take it
as given.

### What is wrong with the current estimator

`measure_trial` folds every trace by the mouse's own choice (`trace * response_side`), and
`PolicyKL` then fits one distribution per stimulus side to the folded pool. Correct and error
turns get mirrored into one pile. That mirroring is the mistake:

- From the mouse's side there is one policy p(x | s) over the raw trajectory x, and it is
  **bimodal** (left-crossing vs right-crossing turns). There is one KL,
  KL[ p(x | s) ‖ q(x | s) ], with p = trained and q = naive.
- Mirroring assumes a left turn mirrored looks like a right turn. Turning is probably not
  symmetric (posture, paw preference), so the pool mixes two different shapes. When the
  mix of turns changes over training (a naive side bias that goes away), the pooled shape
  changes even if neither turn did. The choice shift then shows up a second time inside the
  trajectory term.
- `total = choice + joint` was only an approximation, which can be off in either direction.
  Section 4 of the notebook plots `joint` alone.

### What we tried (scratch tests, 60-mouse cohort, R² ≥ 0.25 → n = 58)

**1. Exact chain rule, no folding.** The choice c is a function of x (the side of threshold
the trajectory crosses), so the full-trajectory KL splits exactly:

    KL[p(x|s) ‖ q(x|s)] = KL[p(c|s) ‖ q(c|s)] + Σ_c p(c|s) · KL[p(x|c,s) ‖ q(x|c,s)]

The terms are computed on signed traces, each (s, c) cell with its own PCA (K = 64, diag,
unbiased). This is the KL of a two-mode mixture whose modes are the turn directions, so no
information is removed and no experimenter label is used. Results:

- Noise floor (reshuffled trials): median 1.5 bits, 1.8% of the signal.
- Ranking of mice is almost unchanged from the old total (Spearman 0.94), but single mice
  moved by up to 25–65%, in both directions.
- Trained-end error cells with fewer than 20 trials were skipped. That happened in about
  half the mice, and each skipped cell carried weight ≤ 0.02.

**2. Unlabeled generative model.** A two-component mixture of probabilistic PCA per (mouse,
s, end) on [log onset, log duration, signed trace at 25 pts], with latent dimension by
held-out likelihood and KL from held-out log-likelihood ratios. It **failed**:

- The components do not find the turn direction (they match the choice 62% of the time,
  median); they split along some larger axis, probably post-response movement.
- Held-out likelihood picks full rank (26/26) in 77% of fits, so the noise term never sets a
  resolution.
- Noisy per mouse: null SD 2.7 bits against a median KL of 14 bits, and 5 mice came out
  negative.

| term | log-Pearson r with φ | Spearman ρ |
|---|---|---|
| choice | 0.82 | 0.80 |
| old: choice + folded joint | 0.41 | 0.35 |
| chain rule, full trajectory KL | 0.39 | 0.33 |
| chain rule, trajectory part only | 0.38 | 0.32 |
| unlabeled MPPCA KL | 0.21 (n = 53) | 0.22 |

Partial Spearman of the movement part with φ, controlling for choice: −0.05 (chain rule)
and −0.05 (MPPCA). The movement adds nothing beyond choice under any estimator so far.

Scripts and per-mouse CSVs: the session scratchpad (`kl_traj/analyze.py`, `kl_traj/mfa.py`,
`kl_traj_vs_phi.csv`, `kl_mfa_vs_phi.csv`); not in the repo. `kl.py` / `wheel.py` still fold.

### Why the magnitude problem points to a model

The full-trajectory KL is about 80 bits, almost all from many small motor dimensions where
the trained mouse's turns become stereotyped. Choice is one binary dimension, about 1 bit, and
it gets swamped. KL can only grow as the description gets finer (choice ≤ choice + onset/
duration ≤ … ≤ all PCs), and it kept growing linearly with K. So "the" KL depends on the
resolution we pick. Under PAC-Bayes the loss (error rate) depends only on which side the wheel
crosses, so motor stereotypy costs KL without lowering error. It is likely motor practice, a
separate process from the learning φ measures. A generative model with an explicit noise /
latent structure is the principled way to set that resolution, but the quick test above did
not get there.

### Design notes for the model

- **Choice must emerge from the movement.** The model generates trajectories; the choice is
  read off as the side of threshold crossed. Check: posterior-predictive p(c | s) matches
  the data at both ends, and the model's KL at the choice level reproduces the choice KL.
- **The side bias should live in a movement variable.** For example, a low-dimensional latent
  for initial movement direction / early velocity (the first ~100–200 ms after onset), whose
  distribution given s carries the bias as an offset and the discrimination as an
  s-dependent shift. Changes of mind (start one way, cross the other) should come out of it,
  not be cut away.
- **Bias vs discrimination.** The earlier decision decomposition (`cohort_decision_decomp.csv`)
  found that the marginal side bias is only ~7% (median) of the choice KL, and discrimination
  I(stim; choice) is ~90%. The model has to capture both, and we should check which of the two
  carries the φ correlation before building around "bias".
- **Starting-accuracy caveat.** The choice KL is close to a re-encoding of starting accuracy
  (trained accuracy is ~constant across mice), so its correlation with φ is near-tautological.
  A movement model is worth it only if it gives something beyond that, e.g. a bias/
  discrimination signal visible in the movement before the choice is committed.
- **Representation.** Use real time aligned to movement onset rather than the time-normalized
  trace (the normalization uses response time, which is itself an outcome). Separate the
  post-response tail; it is large and was probably what the unlabeled mixture split on.
- **Lessons from the MPPCA test.** An unlabeled mixture only helps if it recovers the turn
  modes; give it the structure (direction as a latent) instead of hoping EM finds it. Cross-fit
  symmetrically (both models fit on the same number of trials), or the null goes negative.
  Run fits single-threaded (`OMP_NUM_THREADS=1`) when the machine is shared, since thread
  contention made each fit about 7× slower.

---

This note documents *why* estimating the naive→trained action-policy KL is hard, the methods
we considered, what `PolicyKL` actually does, and the empirical calibration (reliability +
null floor) we measured. It's the reasoning behind the code in `kl.py`.

## The quantity

For each animal we want `KL(late ‖ early)` of the action policy `p(action | stimulus)` —
how much the behavioural output distribution changes from the naive to the trained end of
learning, conditioned on the stimulus side. We factor the action and estimate a KL per factor:

| factor | representation | estimator |
|---|---|---|
| `choice` | response side ±1 | discrete Bernoulli KL |
| `onset` | log movement-onset time | 1-D Gaussian KL |
| `duration` | log movement duration | 1-D Gaussian KL |
| `trace` | top-K folded-trace PC scores | K-dim Gaussian KL |
| `joint` | `[log onset, log duration, PC scores]` | (2+K)-dim Gaussian KL |

Everything is in **bits**, `p = late` (posterior), `q = early` (prior), reduced over the two
stimulus sides by the posterior-weighted mean (`E_stim`) or the sup (`max`).

## Why this is hard

The `trace`/`joint` terms are the crux: we estimate a **KL between two distributions in
~50–100 dimensions from ~1000 samples per stimulus condition.** High-dimensional KL
estimation is genuinely difficult, for several compounding reasons:

- **Plug-in bias.** Fitting a Gaussian and plugging sample mean/covariance into the KL formula
  over-estimates, biased **up by ~`k²/(2n)`** (dimension² over sample size). It's dominated by
  the inflated inverse covariance, `E[Ŝ_q⁻¹] = n/(n−k−2)·Σ_q⁻¹`. What governs it is the ratio
  `k/n`: negligible at `k≪n`, severe as `k→n`, and **undefined for `k ≥ n−2`** (singular
  covariance). At our operating point (`k=64, n≈1000/side`, `k/n≈0.06`) the raw bias is a
  meaningful fraction of the signal.
- **Non-Gaussianity.** The trace PC scores aren't Gaussian even after winsorizing, so any
  Gaussian estimator is misspecified — an analytic (Gaussian) bias correction can't remove
  the non-Gaussian part.
- **PCA sorts by the wrong axis.** KL is *scale-invariant per dimension* — it depends on the
  variance **ratio** and standardized mean shift, not absolute variance. The dominant learning
  signal is *stereotypy* (late variance collapses in fine dimensions), which lives in the
  **low-variance** PCs. So PC1 (74% of variance) contributes little KL, while PCs 30–60 (~0%
  variance) contribute the most. PCA orders by variance, i.e. the *least* discriminative
  directions first — which is why the KL keeps growing as you add PCs.

## Methods we considered

- **Discrete histogram KL (old / exploratory).** Bin onset/duration/scores into log-bins.
  Fatal flaw: a bin with `q=0, p>0` gives `KL=∞`; ~20% of animals hit it. Replaced.
- **Gaussian (MVN) plug-in KL — CHOSEN base.** Closed-form, low-variance, and its parametric
  covariance is exactly the right structure for a *second-moment* (variance-collapse) signal.
  Downside: the `k²/n` bias. This is `gaussian_kl(..., unbiased=False)`.
- **Analytic bias correction (`unbiased=True`, default).** Remove the Gaussian plug-in bias in
  closed form: (1) unbiased inverse `A_q = (n_q−k−2)/(n_q−1)·Ŝ_q⁻¹`; (2) subtract the
  sample-mean-variance inflation `tr(A_q Ŝ_p)/n_p + k/n_q`; (3) debias each log-det via the
  digamma term `_logdet_bias` (cancels when `n_p=n_q`). **Exact on Gaussian data** (validated:
  same-distribution KL → 0 at any `k<n−2`). On real data it removes the Gaussian part but
  leaves a **non-Gaussian residual** (~7% of the full-cov trace/joint).
- **Diagonal / independent-Gaussian (`diag=True`, default).** Treat the PCs as independent:
  `KL = Σ_i KL_i(PC_i)` — sum of per-coordinate 1-D KLs, dropping the off-diagonal covariance.
  This cuts the parameter count from `O(k²)` to `O(k)`, so the bias drops from `k²/n` to `k/n`,
  and each 1-D term is auto-debiased with `k=1`. **The residual floor nearly vanishes.** Cost:
  it is a **lower bound** — it drops the change in cross-PC *correlation* structure (~15–20% of
  the full KL). Note the joint PCA diagonalizes the *pooled* covariance, not each group's, so
  the PCs stay correlated *within* each group; the exact basis for a diagonal KL is the
  simultaneous diagonalization of `(Σ_p, Σ_q)`, but computing that is `Σ_q⁻¹Σ_p` — the same
  `O(k²)` biased operation. So the correlation signal and the bias are the same object.
- **Classifier / density-ratio KL — rejected for this regime.** `KL = E_p[log p/q]`, and a
  cross-fit classifier's logit estimates `log p/q` directly (discriminative, cross-fitting
  debiases cleanly). Beautiful in the large-sample limit, but here a *linear* classifier is
  blind to the variance-collapse signal (needs a quadratic boundary), and a *quadratic* one
  needs ~`k²/2 ≈ 2000+` features vs ~2000 samples → under-determined. The Gaussian's parametric
  covariance is what makes the second-moment signal estimable at this `n`.
- **k-NN nonparametric KL — rejected.** Model-free, but in 50–100 D with ~1000 samples the
  curse of dimensionality makes it noisier/more biased than the Gaussian.
- **Ledoit–Wolf shrinkage — the un-built middle ground.** Shrink the off-diagonals toward 0:
  `diagonal (full shrink) … shrinkage … full-cov (no shrink)`. Would recover most of the
  correlation signal at controlled bias. Not implemented.

## What `PolicyKL` uses (defaults)

```
k_pc=64, diag=True, unbiased=True, winsor=5.0, pseudo=0.5,
n_trials=2000, t_tail=0.5, v_thr=0.1
```
- `choice`: Jeffreys pseudocount (`pseudo=0.5`) so a one-sided naive animal can't give `∞`.
- `onset`/`duration`: winsorized 1-D Gaussian KL on log-time.
- `trace`/`joint`: **diagonal, unbiased** Gaussian KL over K=64 PCs — a calibrated lower bound.
- Reduce over stimulus: `mean` (E_stim, less biased) or `max` (sup; upward-biased — read as a
  conservative envelope, not a point estimate).

Rationale for the config: at `k=64, n≈1000/side` the diagonal+unbiased estimator is
well-conditioned, its null floor is ~1% (see below), and pushing K higher keeps the floor at
~1% while the magnitude grows — but the correlation with φ has already saturated by ~K=64.

## Calibration (what we measured)

Two orthogonal checks — **reliability ≠ validity**, so we need both.

**Split-half reliability** (split each animal's trials, estimate twice, correlate across the
cohort; Spearman-Brown to full length; 10 resamples): **0.98–0.99 for every term**, sd ~0.005.
The per-animal KL is reproduced precisely. This is *not* the weak link.

**Permutation null** (pool naive+trained, randomly re-split → true KL = 0; the residual is
pure bias), median across the cohort in bits:

| term | full-cov (K=50) | **diagonal (K=64)** |
|---|---|---|
| choice / onset / duration | ~0.00 | ~0.00 |
| trace / joint | **~6.7 (≈7% of signal)** | **~0.8 (≈1% of signal)** |

The scalar terms are perfectly calibrated. The full-covariance trace/joint carry a ~7%
non-Gaussian residual floor even after `unbiased`; **the diagonal estimator drops that to ~1%.**
Sweeping K with `diag=True` (K=50→80): the null floor grows *linearly* (0.67→1.13 bits,
~0.015/PC) and stays ~1% throughout, reliability stays ~0.99, the signal magnitude grows
~linearly (real signal in the higher PCs) — but the **φ-correlation plateaus at ρ≈0.45**
(r²≥0.8) by ~K=64 and doesn't rise further.

## How to read the outputs

- **Rankings / correlations: trustworthy for all terms, both estimators.** The bias is a
  smooth, near-constant offset; it barely moves ranks (Spearman). Use these freely.
- **Magnitudes:** clean for `choice`/`onset`/`duration` and for the **diagonal** trace/joint
  (~1% floor). The **full-cov** trace/joint magnitude is inflated ~7% (subtract the permutation
  null for a clean number). The diagonal is a **lower bound** — it omits the ~15–20% cross-PC
  correlation term.
- **`total = choice + joint`** is dominated by the large `joint` magnitude (the ~1-bit choice
  is swamped) — don't use the bit-sum as "the" action KL; keep the terms separate.