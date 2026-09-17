# Reproducing Wu et al. (2014) under the Effective-Dimensionality Theory

## Model: the KL term from measured variability

**Prior and target.** The prior over policies is $q_0(\theta) = \mathcal{N}(0, \Sigma)$
with $\theta \in \mathbb{R}^k$. A task has target $a^* = d\,u$, where $u$ is a unit
vector (the task direction) and $d$ is the target distance.

**Measurements.** For a given task we measure the variance along its task direction and
the total variance:

$$v = u^\top \Sigma\, u, \qquad T = \mathrm{tr}\,\Sigma.$$

**Point-mass posterior.** Take the learned posterior to be a point mass at the target,
$q^* = \delta_{a^*}$. Against a continuous prior, $\mathrm{KL}(q^*\Vert q_0)$ is
infinite; as in the discrete case (Eq. 9 of the theory), we approximate it by the
negative log prior density at the target:

$$\mathrm{KL}(q^*\Vert q_0) \approx -\log q_0(d\,u)
= \frac{d^2}{2}\, u^\top \Sigma^{-1} u + \frac{1}{2}\ln\det\Sigma + \frac{k}{2}\ln 2\pi.$$

**Assumptions.**

- **(A)** The task direction is uncorrelated with the orthogonal directions:
  $u^\top \Sigma\, w = 0$ for every $w \perp u$.
- **(B)** The remaining $k-1$ dimensions are isotropic, with variance
  $r = (T - v)/(k-1)$. Given only the trace of that block, this is the maximum-entropy
  choice; it also gives the largest possible determinant for the block (AM–GM), so
  otherwise the volume term below is an upper bound.

Under (A) and (B), in the basis $(u, \text{remaining } k-1 \text{ directions})$,

$$\Sigma = \mathrm{diag}\big(v,\; r\,I_{k-1}\big),$$

so

$$u^\top \Sigma^{-1} u = \frac{1}{v}, \qquad \det\Sigma = v\, r^{\,k-1}.$$

**KL term.**

$$\mathrm{KL}(q^*\Vert q_0) \approx \frac{d^2}{2 v} + \frac{1}{2}\ln v
+ \frac{k-1}{2}\ln\frac{T - v}{k-1} + \frac{k}{2}\ln 2\pi.$$

**Predicted ratio.** For two tasks $j = 1, 2$ with measurements $(v_j, T_j)$, the ratio
of trials needed to reach the same performance (Eq. 10 of the theory) is

$$\frac{m_2}{m_1} =
\frac{\dfrac{d^2}{2 v_2} + \dfrac{1}{2}\ln v_2 + \dfrac{k-1}{2}\ln\dfrac{T_2 - v_2}{k-1} + \dfrac{k}{2}\ln 2\pi}
{\dfrac{d^2}{2 v_1} + \dfrac{1}{2}\ln v_1 + \dfrac{k-1}{2}\ln\dfrac{T_1 - v_1}{k-1} + \dfrac{k}{2}\ln 2\pi}.$$

The inputs are the measured $v_j$ and $T_j$, the target distance $d$, and the prior
dimension $k$.

## Learning speed: the reciprocal $1/\mathrm{KL}$

Since the effective dimensionality acts multiplicatively on the trials needed, learning
speed goes as $1/\mathrm{KL}$. Under the maximum-entropy (isotropic) assumption above,
collecting the three logarithmic terms into the prior's log-normalizer

$$L(v) \;=\; \tfrac{1}{2}\ln\det(2\pi\Sigma)
\;=\; \frac{1}{2}\ln v + \frac{k-1}{2}\ln\frac{T-v}{k-1} + \frac{k}{2}\ln 2\pi,
\qquad \mathrm{KL} = \frac{d^2}{2v} + L(v),$$

gives the exact expression

$$\frac{1}{\mathrm{KL}} = \frac{2v}{d^2 + 2\,v\,L(v)}.$$

**Michaelis–Menten form.** $L$ depends on $v$ only logarithmically, so holding it at a
typical value turns the above into a saturating hyperbola in the measured variance:

$$\frac{1}{\mathrm{KL}} \approx \frac{1}{L}\cdot\frac{v}{v + v_{1/2}},
\qquad v_{1/2} = \frac{d^2}{2L}.$$

Learning speed is half-maximal at $v_{1/2}$ and approaches a ceiling of $1/L$.

**Small-$v$ limit.** For $v \ll v_{1/2}$,

$$\frac{1}{\mathrm{KL}} \approx \frac{2v}{d^2},$$

i.e. learning speed is proportional to task-relevant variance — the regime in which
Wu et al.'s correlations are measured. The saturation at larger $v$ is the additional
prediction of the volume term.

**Sign of $L$.** The hyperbolic form requires $L > 0$. If $L < 0$ the reciprocal grows
faster than linearly and diverges where $\mathrm{KL} = 0$, which signals that the
density approximation has broken down; writing
$L_\varepsilon = \tfrac{1}{2}\ln\det(2\pi\Sigma/\varepsilon^2)$ for a behavioral
resolution $\varepsilon$ restores $L > 0$.

**Turning point.** $1/\mathrm{KL}$ need not be monotone in $v$. Setting
$\mathrm{d}\mathrm{KL}/\mathrm{d}v = 0$ gives a quadratic,

$$k\,v^2 - (d^2 + T)\,v + d^2 T = 0,
\qquad v^* = \frac{(d^2+T) \pm \sqrt{(d^2+T)^2 - 4k\,d^2 T}}{2k}.$$

Real roots require $(d^2+T)^2 \ge 4k\,d^2T$, i.e. $d^2$ and $T$ separated in scale.
Otherwise $\mathrm{KL}$ decreases monotonically in $v$ and $1/\mathrm{KL}$ rises
monotonically toward its ceiling.

## Alternative structure: a power-law spectrum

Assumption (B) — isotropy of the unmeasured block — can be replaced by a spectral shape
for the whole covariance. Under (A) the measured direction is an eigenvector of $\Sigma$,
i.e. a principal component, but we do **not** know its rank.

**(B′) Power-law spectrum.** All $k$ eigenvalues of $\Sigma$ decay as a power law with
exponent $\beta$,

$$\lambda_i = \lambda_1\, i^{-\beta}, \qquad i = 1,\dots,k,$$

and the measured direction is the PC of some unknown rank $p$, so that $v = \lambda_p$.
The total variance fixes the leading eigenvalue,

$$T = \sum_{i=1}^{k} \lambda_i = \lambda_1 H_k(\beta), \qquad
H_k(\beta) = \sum_{i=1}^{k} i^{-\beta}, \qquad
\lambda_1 = \frac{T}{H_k(\beta)},$$

so the determinant depends only on $T$, $k$ and $\beta$:

$$\ln\det\Sigma = k \ln\frac{T}{H_k(\beta)} - \beta\,\ln(k!).$$

The measured $v$ enters only through the alignment term:

$$\mathrm{KL}(q^*\Vert q_0) \approx \frac{d^2}{2 v}
+ \frac{k}{2}\ln\frac{T}{H_k(\beta)} - \frac{\beta}{2}\ln(k!) + \frac{k}{2}\ln 2\pi.$$

The rank of the measured direction is implied but not needed,
$p = \big(T / (v H_k(\beta))\big)^{1/\beta}$; consistency requires $1 \le p \le k$, i.e.
$v$ must lie between $\lambda_k$ and $\lambda_1$.

**Predicted ratio.** For two tasks $j = 1, 2$ measured in the same prior (same $T$),

$$\frac{m_2}{m_1} = \frac{\dfrac{d^2}{2 v_2} + V}{\dfrac{d^2}{2 v_1} + V},
\qquad V = \frac{k}{2}\ln\frac{T}{H_k(\beta)} - \frac{\beta}{2}\ln(k!) + \frac{k}{2}\ln 2\pi.$$

The inputs are the measured $v_1$, $v_2$ and $T$, the target distance $d$, the prior
dimension $k$, and the spectral exponent $\beta$.

**Effect of $\beta$.** The volume term $V$ is shared by the two tasks, so it enters both
the numerator and the denominator and pulls the ratio toward 1. A steeper spectrum
(larger $\beta$) makes $V$ smaller, pushing the predicted ratio back toward the
alignment-only ratio $v_1 / v_2$. At $\beta = 0$, $H_k(0) = k$ and
$V = \tfrac{k}{2}\ln(2\pi T/k)$, the isotropic case.
