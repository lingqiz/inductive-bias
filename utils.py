import types
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.optimize import curve_fit, minimize_scalar


def _loglog_seed(m, y):
    """Rough (alpha, d) seed for a power-law fit from a log-log OLS.

    Assumes y = (d/m)**alpha, so log y = alpha*log d - alpha*log m -- a line in
    log-log with slope = -alpha and intercept = alpha*log d. Points with
    non-positive/non-finite m or y are dropped. The cap (if any) is ignored here;
    this only provides an initial guess for the nonlinear fit.
    """
    m = np.asarray(m, float).ravel()
    y = np.asarray(y, float).ravel()
    ok = (m > 0) & (y > 0) & np.isfinite(m) & np.isfinite(y)
    slope, intercept = np.polyfit(np.log(m[ok]), np.log(y[ok]), 1)
    alpha = max(-slope, 1e-3)
    d = float(np.exp(intercept / alpha))
    return alpha, d


def _gof(pred, y):
    """Goodness-of-fit stats for a fitted curve: ``rss`` and ``r2``.

    ``r2`` is the fraction of variance explained, 1 - SS_res/SS_tot -- comparable
    across animals (unlike raw rss, which grows with the number of points).
    """
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'rss': ss_res, 'r2': float(r2)}


def _safe_pow(base, alpha):
    """base**alpha, saturating at 1e300 instead of overflowing to inf.

    A large exponent turns (phi/m)**alpha into a step that overflows for base > 1;
    the saturated value is still far above any cap, so capped curves are unchanged,
    and non-overflowing values are exactly base**alpha.
    """
    with np.errstate(over='ignore'):
        return np.minimum(np.asarray(base, float) ** alpha, 1e300)


_START_SCALES = (0.3, 3.0, 10.0)


def _multistart_seeds(bases, n_shape):
    """Each base seed, plus copies with one shape parameter rescaled by each of _START_SCALES."""
    seeds = []
    for b in bases:
        seeds.append(list(b))
        for i in range(n_shape):
            for sc in _START_SCALES:
                seeds.append([p * sc if j == i else p for j, p in enumerate(b)])
    return seeds


def _fit_multistart(curve, m, y, starts, lo, hi):
    """Bounded least-squares fit of curve(m, *p) to y from each start; lowest-RSS popt.

    Starts are clipped into the bounds (curve_fit rejects infeasible seeds). Raises
    RuntimeError if no start converges.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    best, best_rss, err = None, np.inf, None
    for p0 in starts:
        p0 = np.clip(np.asarray(p0, float), lo + 1e-9, hi - 1e-9)
        try:
            popt, _ = curve_fit(curve, m, y, p0=p0, bounds=(lo, hi), maxfev=20000)
        except (RuntimeError, ValueError) as e:
            err = e
            continue
        rss = float(np.sum((curve(m, *popt) - y) ** 2))
        if rss < best_rss:
            best, best_rss = popt, rss
    if best is None:
        raise RuntimeError(f'curve fit failed from all {len(starts)} starts: {err}')
    return best


class _LearningCurve:
    """Shared fitting machinery for learning-curve models g(m) of error vs trials m.

    Subclasses define ``PARAMS`` (shape-parameter names), ``BOUNDS`` (default
    (lo, hi) per shape parameter), ``_g(m, f, *p)`` (the curve with asymptotic
    error f, before the cap) and ``_seed(m, y)`` (initial shape parameters from the
    floor-subtracted error).

    ``floor`` is the asymptotic error as m -> inf, held FIXED at the value passed
    (default 0.0) unless ``fit_floor=True``, in which case it is fitted as an extra
    parameter (seeded at the late-session error and near zero). Optionally caps the
    error at ``cap`` (e.g. 0.5 = chance): the returned error is min(g(m), cap),
    applied both when fitting and when predicting. ``bounds`` overrides the (lo, hi)
    of any parameter by name, including 'floor' (default (0, cap or 1)).
    ``multistart=True`` also fits from seeds with each shape parameter rescaled by
    0.3x / 3x / 10x and keeps the lowest-RSS fit.

    The fit minimizes the linear least-squares residual on the raw error rate,
    Sum (g(m_i) - y_i)**2 (``loss='linear'``, default). With ``loss='log'`` it
    minimizes the residual on the LOG error, Sum (log g(m_i) - log y_i)**2, i.e.
    relative residuals: for curves spanning decades this weights all m evenly, where
    the linear fit is dominated by the largest (small-m) errors and ignores the tail
    that sets the floor. ``params`` holds the fitted parameters, ``floor``, ``cap``,
    ``rss`` and ``r2`` (fraction of variance explained), and for ``loss='log'`` the
    minimized ``log_rss``.
    """

    PARAMS = ()
    BOUNDS = {}

    def __init__(self, cap=None, floor=0.0, loss='linear', fit_floor=False,
                 bounds=None, multistart=False):
        if loss not in ('linear', 'log'):
            raise ValueError(f"loss must be 'linear' or 'log', got {loss!r}")
        self.cap = cap
        self.floor = float(floor)
        self.loss = loss
        self.fit_floor = fit_floor
        self.multistart = multistart
        self.bounds = {**self.BOUNDS, 'floor': (0.0, cap if cap is not None else 1.0),
                       **(bounds or {})}
        self.params = None
        self.fitted = False

    def _g(self, m, f, *p):
        raise NotImplementedError

    def _seed(self, m, y):
        raise NotImplementedError

    def _names(self):
        return list(self.PARAMS) + (['floor'] if self.fit_floor else [])

    def _curve(self, m, *p):
        m = np.asarray(m, float)
        f = p[-1] if self.fit_floor else self.floor
        g = self._g(m, f, *p[:len(self.PARAMS)])
        return np.minimum(g, self.cap) if self.cap is not None else g

    def fit(self, m, y, p0=None):
        """Fit to data (m = cumulative trials, y = error rate); p0 seeds all fitted parameters."""
        m = np.asarray(m, float).ravel()
        y = np.asarray(y, float).ravel()
        names = self._names()
        lo = [self.bounds[k][0] for k in names]
        hi = [self.bounds[k][1] for k in names]

        if p0 is not None:
            bases = [list(p0)]
        else:
            y_seed = y - self.floor
            shapes = [list(self._seed(m, np.clip(y_seed, 1e-6, None)))]
            if self.multistart and np.any(y_seed <= 0):
                # zero-error points (clipped to 1e-6) can dominate the log-log seed; also seed without them
                pos = y_seed > 0
                shapes.append(list(self._seed(m[pos], y_seed[pos])))
            floor0 = [min(y[-5:].mean(), 0.9 * hi[-1]), 0.01] if self.fit_floor else []
            bases = [s + [f] for s in shapes for f in floor0] if self.fit_floor else shapes
        starts = _multistart_seeds(bases, len(self.PARAMS)) if self.multistart else bases

        if self.loss == 'log':
            curve, target = (lambda m, *p: np.log(self._curve(m, *p))), np.log(y)
        else:
            curve, target = self._curve, y
        popt = _fit_multistart(curve, m, target, starts, lo, hi)

        fitted = dict(zip(names, map(float, popt)))
        self.params = {**{k: fitted[k] for k in self.PARAMS},
                       'floor': fitted.get('floor', self.floor), 'cap': self.cap,
                       **_gof(self._curve(m, *popt), y)}
        if self.loss == 'log':
            self.params['log_rss'] = float(np.sum((curve(m, *popt) - target) ** 2))
        self.fitted = True
        return self

    def predict(self, m):
        """Predicted error rate at trial counts ``m`` using the fitted model."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._curve(m, *[self.params[k] for k in self._names()])

    def get_params(self):
        """Return fitted parameters."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.params.copy()


class PACBayesCurve(_LearningCurve):
    """Theory-derived learning curve g(m) = floor + (2/3)*phi/m + (2*phi/m)**alpha.

    Fitted parameters are phi > 0 (complexity scale) and alpha > 0 (exponent); see
    :class:`_LearningCurve` for floor / cap / loss / bounds / multistart.
    """

    PARAMS = ('phi', 'alpha')
    BOUNDS = {'phi': (0.0, np.inf), 'alpha': (0.0, np.inf)}

    def _g(self, m, f, phi, alpha):
        return f + (2.0 / 3.0) * phi / m + _safe_pow(2.0 * phi / m, alpha)

    def _seed(self, m, y):
        # phi ~ d/2 since at large m the term (2*phi/m)**alpha matches (d/m)**alpha when d = 2*phi
        a0, d0 = _loglog_seed(m, y)
        return [max(d0 / 2.0, 1e-6), a0]


class PowerLawCurve(_LearningCurve):
    """Phenomenological power-law learning curve g(m) = floor + (d/m)**alpha.

    Fitted parameters are d > 0 (trials-to-scale) and alpha > 0 (exponent); see
    :class:`_LearningCurve` for floor / cap / loss / bounds / multistart.
    """

    PARAMS = ('d', 'alpha')
    BOUNDS = {'d': (0.0, np.inf), 'alpha': (0.0, np.inf)}

    def _g(self, m, f, d, alpha):
        return f + _safe_pow(d / m, alpha)

    def _seed(self, m, y):
        a0, d0 = _loglog_seed(m, y)
        return [max(d0, 1e-6), a0]


class ExponentialCurve(_LearningCurve):
    """Exponential learning curve g(m) = floor + a * exp(-m / tau).

    Fitted parameters are a > 0 (amplitude at m = 0) and tau >= 1 (decay constant,
    in trials); see :class:`_LearningCurve` for floor / cap / loss / bounds / multistart.
    """

    PARAMS = ('a', 'tau')
    BOUNDS = {'a': (0.0, np.inf), 'tau': (1.0, np.inf)}

    def _g(self, m, f, a, tau):
        return f + a * np.exp(-m / tau)

    def _seed(self, m, y):
        a0 = max(y[:3].mean(), 1e-3)
        below = np.flatnonzero(y < a0 / np.e)
        return [a0, m[below[0]] if len(below) else np.median(m)]


class SigmoidCurve(_LearningCurve):
    """Logistic learning curve g(m) = floor + (top - floor) / (1 + exp((m - m0) / s)).

    Starts at ``top`` (= cap, or 0.5 if no cap) and decays to the floor around trial
    m0 with width s >= 1 (in trials); see :class:`_LearningCurve` for floor / cap /
    loss / bounds / multistart.
    """

    PARAMS = ('m0', 's')
    BOUNDS = {'m0': (-np.inf, np.inf), 's': (1.0, np.inf)}

    @property
    def top(self):
        return self.cap if self.cap is not None else 0.5

    def _g(self, m, f, m0, s):
        return f + (self.top - f) / (1 + np.exp(np.clip((m - m0) / s, -50, 50)))

    def _seed(self, m, y):
        below = np.flatnonzero(y < (self.top - self.floor) / 2)
        m0 = m[below[0]] if len(below) else np.median(m)
        return [m0, max(m0 / 4, 1.0)]


def curve_smoothness(curves, err_min=1e-3):
    """How smooth each learning curve is, measured on log10 error (higher = smoother).

    ``curves`` is a list of (m, err) arrays binned evenly in log m, so a power law is a
    straight line. Zero-error bins are clipped to ``err_min`` before taking the log. The
    score is minus the RMS second difference across consecutive bins,
    log_err[k+1] - 2 log_err[k] + log_err[k-1], i.e. how far each bin sits from the line
    through its two neighbours: near 0 for a smooth curve (even a curved one), strongly
    negative for one that zig-zags. It uses no model fit.
    """
    scores = []
    for _, err in curves:
        zigzag = np.diff(np.log10(np.clip(err, err_min, None)), 2)
        scores.append(-np.sqrt(np.mean(zigzag ** 2)) if len(zigzag) else 0.0)
    return np.array(scores)


class PiecewiseLinear:
    """.. deprecated::
        Use :class:`PACBayesCurve` or :class:`PowerLaw` for learning-curve
        fitting. Retained only for backward compatibility; no longer used for
        empirical learning-curve analysis.
    """

    def __init__(self):
        warnings.warn(
            "PiecewiseLinear is deprecated; use PACBayesCurve or PowerLawCurve "
            "for learning-curve fitting.",
            DeprecationWarning, stacklevel=2,
        )
        self.params = None
        self.fitted = False

    def fit(self, x, y, x0_candidates=None):
        """
        Fit y ~ piecewise linear with a single breakpoint x0 and continuity.

        Model:
          y = c + m1*(x - x0)  if x <= x0
              c + m2*(x - x0)  if x  > x0
        Unknowns: c, m1, m2, and x0. We search over x0 candidates and solve
        linear least squares for (c, m1, m2) for each, picking the lowest RSS.

        Args:
            x, y: 1D arrays of equal length.
            x0_candidates: optional iterable of breakpoint candidates. If None,
                           uses midpoints between sorted unique x (excluding extremes).
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # Build default candidate breakpoints between consecutive unique x values
        if x0_candidates is None:
            ux = np.unique(x)
            # midpoints between consecutive unique x's, dropping ends
            mids = (ux[:-1] + ux[1:]) / 2.0
            x0_candidates = mids[1:-1] if len(mids) > 2 else mids  # avoid picking extremes

        best = None
        for x0 in x0_candidates:
            left_mask = x <= x0
            right_mask = x > x0
            if left_mask.sum() < 2 or right_mask.sum() < 2:
                # Need at least 2 points on each side for a robust line fit
                continue

            # Design matrix for [c, m1, m2]: y = c + m1*(x-x0)*I_left + m2*(x-x0)*I_right
            X = np.column_stack([
                np.ones_like(x),
                (x - x0) * left_mask.astype(float),
                (x - x0) * right_mask.astype(float),
            ])
            # Solve least squares
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            c, m1, m2 = beta

            yhat = X @ beta
            rss = np.sum((y - yhat) ** 2)

            if (best is None) or (rss < best['rss']):
                # Convert to intercept form per side: y = m*x + b
                b1 = c - m1 * x0
                b2 = c - m2 * x0
                best = {'x0': float(x0), 'm1': float(m1), 'b1': float(b1),
                        'm2': float(m2), 'b2': float(b2), 'rss': float(rss)}

        if best is None:
            raise ValueError("No valid breakpoint found. Provide more diverse x or explicit x0_candidates.")

        self.params = best
        self.fitted = True
        return self

    def predict(self, x):
        """Predict using fitted piecewise linear model."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        x = np.asarray(x)
        x0 = self.params['x0']
        yhat = np.where(
            x <= x0,
            self.params['m1'] * x + self.params['b1'],
            self.params['m2'] * x + self.params['b2'],
        )
        return yhat

    def plot(self, x, y, show=True):
        """Scatter data and overlay the fitted two-segment line."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        order = np.argsort(x)
        x_sorted = x[order]
        x0 = self.params['x0']

        # Dense x for drawing each segment
        xl = np.linspace(x_sorted.min(), x0, 200)
        xr = np.linspace(x0, x_sorted.max(), 200)
        yl = self.params['m1'] * xl + self.params['b1']
        yr = self.params['m2'] * xr + self.params['b2']

        plt.figure(figsize=(6,4))
        plt.scatter(x, y, s=25)
        plt.plot(xl, yl, linestyle='--')
        plt.plot(xr, yr, linestyle='--')
        plt.axvline(x0, linestyle=':')
        plt.xlabel("x")
        plt.ylabel("y")
        if show:
            plt.show()

    def get_params(self):
        """Return fitted parameters."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.params.copy()
