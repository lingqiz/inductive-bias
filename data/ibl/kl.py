"""Action-policy KL divergence between the naive and trained mouse.

Given an :class:`~data.ibl.utils.Animal`, estimate how much the animal's action
policy p(action | stimulus) changes from the naive (early) to the trained (late)
end of learning -- the KL(late || early) of each action factor, conditioned on
the stimulus side. The factors are:

    choice     response side +1/-1                 (discrete, binary KL)
    onset      log movement-onset time             (winsorized 1-D Gaussian KL)
    duration   log movement duration               (winsorized 1-D Gaussian KL)
    trace      top-K folded-trace PC scores        (winsorized MVN Gaussian KL)
    joint      [log onset, log duration, PC scores] (winsorized MVN Gaussian KL)

Conventions (uniform across every term):
  - reported in BITS (log base 2);
  - p = late (posterior), q = early (prior);  KL(p || q);
  - computed per stimulus side s in {+1, -1}, then combined over the stimulus
    either as the p(stim)-weighted mean E_stim = sum_s w[s] KL[s] ('mean') or the
    sup over sides max_s KL[s] ('max'); w[s] is the LATE stimulus frequency;
  - K = 10 trace PCs (the reliably-estimated dimensionality; split-half loading
    similarity), one PCA per stimulus side (common basis for early + late);
  - winsor = 5% (each PC / log-time coordinate clipped to its [5, 95] percentiles,
    where the heavy-tailed PC scores become approximately Gaussian).

Note: winsorize() clips each sample to ITS OWN percentiles, so the map is not
strictly shared across the two groups -- the data-processing "lower bound"
framing is therefore approximate. The Gaussian-KL terms are analytically
bias-corrected by default (``unbiased=True``): the plug-in KL is biased up by
~k^2 / (2n), governed by the ratio k/n, so the correction matters once K is a
non-trivial fraction of the per-side sample count (see :func:`gaussian_kl`).

Generated with Claude Code.
"""
import numpy as np
from scipy.special import digamma
from sklearn.decomposition import PCA

from .wheel import collect_trials, DT, T_TAIL, V_THR


# ---------------------------------------------------------------------------
# Estimators (single source of truth for the KL terms)
# ---------------------------------------------------------------------------

def winsorize(Z, pct):
    """Clip each column of Z to its [pct, 100-pct] percentiles (computed within Z)."""
    if pct <= 0:
        return Z
    lo = np.percentile(Z, pct, axis=0)
    hi = np.percentile(Z, 100 - pct, axis=0)
    return np.clip(Z, lo, hi)


def _logdet_bias(k, nu):
    """E[ln det S] - ln det Sigma for an unbiased sample covariance S with nu = n-1 dof.

    Equals sum_{i=1..k} [psi((nu-i+1)/2) - ln(nu/2)] (< 0): the log-det of a sample
    covariance is biased low. Subtract this from ln det(S) to recover ln det(Sigma).
    """
    i = np.arange(1, k + 1)
    return float(np.sum(digamma((nu - i + 1) / 2.0) - np.log(nu / 2.0)))


def gaussian_kl(Zp, Zq, winsor, unbiased=True):
    """KL(N(p) || N(q)) in bits, fitting an MVN (mean + covariance) to winsorized samples.

    p = late (posterior), q = early (prior); Zp, Zq are (n, k) score matrices. The
    covariances (np.cov, ddof=1) are wrapped in ``np.atleast_2d`` so 1-D marginals
    (k = 1) also work.

    With ``unbiased=True`` (default) the plug-in bias is removed analytically under the
    Gaussian model -- the plug-in KL is biased UP by ~k^2 / (2 n) (dimension^2 over sample
    size), dominated by the inflated inverse covariance E[S_q^-1] = n_q/(n_q-k-2) * Sigma_q^-1.
    Three corrections (see derivation notes): (1) unbiased inverse A_q = (n_q-k-2)/(n_q-1)
    * S_q^-1; (2) subtract the sample-mean-variance inflation tr(A_q S_p)/n_p + k/n_q from the
    Mahalanobis term; (3) debias each log-det by _logdet_bias (cancels when n_p == n_q).
    Requires k <= n-2 per group; returns nan otherwise (covariance under-determined).
    """
    Zp, Zq = winsorize(Zp, winsor), winsorize(Zq, winsor)
    np_p, np_q = Zp.shape[0], Zq.shape[0]
    mp, mq = Zp.mean(0), Zq.mean(0)
    Sp = np.atleast_2d(np.cov(Zp, rowvar=False))
    Sq = np.atleast_2d(np.cov(Zq, rowvar=False))
    k = Zp.shape[1]
    Sq_inv = np.linalg.inv(Sq)
    d = mq - mp

    # bias-corrected KL under the Gaussian model
    if unbiased:
        nu_p, nu_q = np_p - 1, np_q - 1
        if nu_q - k - 1 <= 0 or nu_p <= 0:        # inverse under-determined -> can't debias
            return np.nan
        A = ((nu_q - k - 1) / nu_q) * Sq_inv       # unbiased estimate of Sigma_q^-1
        tr_ASp = np.trace(A @ Sp)                  # unbiased tr(Sigma_q^-1 Sigma_p)
        maha = d @ A @ d - tr_ASp / np_p - k / np_q          # debiased Mahalanobis
        logdet = ((np.linalg.slogdet(Sq)[1] - _logdet_bias(k, nu_q))
                  - (np.linalg.slogdet(Sp)[1] - _logdet_bias(k, nu_p)))
        kl_nats = 0.5 * (tr_ASp + maha - k + logdet)
        return float(kl_nats / np.log(2))

    # no correction: plain plug-in KL
    kl_nats = 0.5 * (np.trace(Sq_inv @ Sp) + d @ Sq_inv @ d - k
                     + np.linalg.slogdet(Sq)[1] - np.linalg.slogdet(Sp)[1])
    return float(kl_nats / np.log(2))


def choice_kl(resp_p, resp_q, pseudo=0.5):
    """KL(late || early) in bits of the binary response-side distribution (+1 / -1).

    ``resp_p`` / ``resp_q`` are arrays of +1 / -1 choices (late / early). Each side's
    probability uses a Jeffreys (add-``pseudo``) pseudocount, so a one-sided early
    group (q_plus = 0 or 1) gives a finite KL instead of dividing by zero. This is
    the discrete analog of the winsorization on the Gaussian terms: sample-size aware
    (floor ~ pseudo / n) and negligible at ~1000 trials/side.
    """
    def dist(r):
        p_plus = (np.count_nonzero(r == 1) + pseudo) / (len(r) + 2 * pseudo)
        return np.array([p_plus, 1.0 - p_plus])

    p, q = dist(resp_p), dist(resp_q)
    return float(np.sum(p * np.log2(p / q)))


def log_col(v, eps):
    """log(seconds) as an (n, 1) column, floored at eps (onset can be exactly 0)."""
    return np.log(np.clip(v, eps, None))[:, None]


# ---------------------------------------------------------------------------
# Estimator + result
# ---------------------------------------------------------------------------

class KLResult:
    """The naive->trained action-policy KL for one animal, in bits.

    ``terms`` maps each term name to its per-stimulus-side KL(late || early)::

        terms = {
            'choice':   {+1: kl, -1: kl},   # response side               (discrete)
            'onset':    {+1: kl, -1: kl},   # log movement onset          (1-D Gaussian)
            'duration': {+1: kl, -1: kl},   # log movement duration       (1-D Gaussian)
            'trace':    {+1: kl, -1: kl},   # folded-trace PC scores       (MVN)
            'joint':    {+1: kl, -1: kl},   # [onset, duration, trace]    (joint MVN)
        }

    ``weights`` is the late (posterior) stimulus frequency per side,
    ``{+1: w, -1: w}``, used to average the two sides. Read a term either raw
    per-side or combined over the stimulus::

        res.terms['joint']            # {+1: .., -1: ..}   (same as res.per_stim('joint'))
        res.reduce('joint', 'max')    # sup over sides            (sup_s)
        res.reduce('joint', 'mean')   # p(stim)-weighted average  (E_stim)
        res.summary('mean')           # {name: value} for all five terms
    """

    TERMS = ('choice', 'onset', 'duration', 'trace', 'joint')
    _MODE = {'mean': 'mean', 'e': 'mean', 'max': 'max', 'sup': 'max'}

    def __init__(self, terms, weights):
        self.terms = terms
        self.weights = weights

    def per_stim(self, name):
        """Raw per-side values {+1: kl, -1: kl} for a term."""
        return self.terms[name]

    def reduce(self, name, mode='mean'):
        """Combine a term over the stimulus.

        mode 'mean' (alias 'E')  -> E_stim = sum_s w[s] KL[s]  (p(stim)-weighted)
        mode 'max'  (alias 'sup') -> sup_s  = max_s KL[s]
        """
        try:
            mode = self._MODE[mode.lower()]
        except KeyError:
            raise ValueError("mode must be 'mean'/'E' or 'max'/'sup', got %r" % mode)
        per_side = self.terms[name]
        if mode == 'max':
            return float(max(per_side.values()))
        return float(sum(self.weights[s] * per_side[s] for s in per_side))

    def summary(self, mode='mean'):
        """All five terms combined over the stimulus, as {name: value} (bits)."""
        return {name: self.reduce(name, mode) for name in self.TERMS}


class PolicyKL:
    """Estimate the naive->trained action-policy KL for an animal.

    Configure the estimator once, then apply it to any Animal (so a single config
    runs across the cohort):

        est = PolicyKL()                      # defaults below
        res = est.estimate(cohort['CSHL045'])
        res.summary('mean')                   # E_stim per term, in bits
        res.summary('max')                    # sup_s per term
        res.per_stim('joint')                 # {+1: .., -1: ..}

    Parameters
    ----------
    k_pc      trace PCs kept (reliably-estimated dimensionality)
    n_trials  usable easy trials collected at each end of learning
    t_tail    trace window kept past the response (s) -- forwarded to measure_trial
    v_thr     movement-onset speed threshold (rad/s) -- forwarded to measure_trial
    winsor    per-coordinate winsorization percent for the Gaussian KL terms
    pseudo    Jeffreys pseudocount for the discrete choice KL (add-pseudo smoothing)
    unbiased  analytically remove the plug-in Gaussian-KL bias (~k^2/2n); on by default
    eps       floor applied before log() of onset / duration (s)
    """

    def __init__(self, k_pc=50, n_trials=2000, t_tail=T_TAIL, v_thr=V_THR,
                 winsor=5.0, pseudo=0.5, unbiased=True, eps=DT):
        self.k_pc = k_pc
        self.n_trials = n_trials
        self.t_tail = t_tail
        self.v_thr = v_thr
        self.winsor = winsor
        self.pseudo = pseudo
        self.unbiased = unbiased
        self.eps = eps

    def estimate(self, animal):
        """Collect naive/trained trials for ``animal`` and compute all KL terms."""
        early = collect_trials(animal, self.n_trials, from_end=False,
                               t_tail=self.t_tail, v_thr=self.v_thr)
        late = collect_trials(animal, self.n_trials, from_end=True,
                              t_tail=self.t_tail, v_thr=self.v_thr)

        sides = (+1, -1)
        weights = {s: float(np.mean(late.stim_side == s)) for s in sides}
        terms = {name: {} for name in KLResult.TERMS}

        for s in sides:
            q_sel, p_sel = early.stim_side == s, late.stim_side == s

            # choice: discrete binary KL of the response side
            terms['choice'][s] = choice_kl(late.response_side[p_sel],
                                           early.response_side[q_sel], self.pseudo)

            # per-side PCA on the folded traces (common basis for early + late)
            pca = PCA(n_components=self.k_pc).fit(np.vstack([early.trace[q_sel], late.trace[p_sel]]))
            trace_q = pca.transform(early.trace[q_sel])
            trace_p = pca.transform(late.trace[p_sel])

            # onset / duration: winsorized 1-D Gaussian KL on log-time
            onset_q, onset_p = log_col(early.onset[q_sel], self.eps), log_col(late.onset[p_sel], self.eps)
            dur_q, dur_p = log_col(early.duration[q_sel], self.eps), log_col(late.duration[p_sel], self.eps)
            terms['onset'][s] = gaussian_kl(onset_p, onset_q, self.winsor, self.unbiased)
            terms['duration'][s] = gaussian_kl(dur_p, dur_q, self.winsor, self.unbiased)

            # trace: winsorized MVN Gaussian KL on the PC scores
            terms['trace'][s] = gaussian_kl(trace_p, trace_q, self.winsor, self.unbiased)

            # joint: winsorized MVN over [log onset, log duration, PC scores]
            joint_q = np.column_stack([onset_q, dur_q, trace_q])
            joint_p = np.column_stack([onset_p, dur_p, trace_p])
            terms['joint'][s] = gaussian_kl(joint_p, joint_q, self.winsor, self.unbiased)

        return KLResult(terms, weights)
