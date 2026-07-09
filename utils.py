import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.optimize import curve_fit


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


class PACBayesCurve:
    """Theory-derived learning curve g(m) = (2/3)*phi/m + (2*phi/m)**alpha.

    Fitted parameters are phi > 0 (complexity scale) and alpha > 0 (exponent).
    Optionally caps the error rate at ``cap`` (e.g. 0.5 = chance): the returned
    error is min(g(m), cap), applied both when fitting and when predicting.

    The fit minimizes the linear least-squares residual on the raw error rate,
    Sum (g(m_i) - y_i)**2, where m is cumulative trials and y is the error rate.
    """

    def __init__(self, cap=None):
        self.cap = cap
        self.params = None
        self.fitted = False

    def _curve(self, m, phi, alpha):
        m = np.asarray(m, float)
        g = (2.0 / 3.0) * phi / m + (2.0 * phi / m) ** alpha
        return np.minimum(g, self.cap) if self.cap is not None else g

    def fit(self, m, y, p0=None):
        """Fit phi, alpha to data (m = cumulative trials, y = error rate)."""
        m = np.asarray(m, float).ravel()
        y = np.asarray(y, float).ravel()

        if p0 is None:
            # seed alpha from a log-log slope; phi ~ d/2 since at large m the
            # power-law term (2*phi/m)**alpha matches (d/m)**alpha when d = 2*phi
            a0, d0 = _loglog_seed(m, y)
            p0 = [max(d0 / 2.0, 1e-6), a0]

        popt, _ = curve_fit(self._curve, m, y, p0=p0,
                            bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=20000)
        phi, alpha = popt
        rss = float(np.sum((self._curve(m, *popt) - y) ** 2))
        self.params = {'phi': float(phi), 'alpha': float(alpha),
                       'cap': self.cap, 'rss': rss}
        self.fitted = True
        return self

    def predict(self, m):
        """Predicted error rate at trial counts ``m`` using the fitted model."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._curve(m, self.params['phi'], self.params['alpha'])

    def get_params(self):
        """Return fitted parameters."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.params.copy()


class PowerLawCurve:
    """Phenomenological power-law learning curve g(m) = (d/m)**alpha.

    Fitted parameters are d > 0 (trials-to-scale) and alpha > 0 (exponent).
    Optionally caps the error rate at ``cap`` (e.g. 0.5 = chance): the returned
    error is min(g(m), cap), applied both when fitting and when predicting.

    The fit minimizes the linear least-squares residual on the raw error rate,
    Sum (g(m_i) - y_i)**2, where m is cumulative trials and y is the error rate.
    """

    def __init__(self, cap=None):
        self.cap = cap
        self.params = None
        self.fitted = False

    def _curve(self, m, d, alpha):
        m = np.asarray(m, float)
        g = (d / m) ** alpha
        return np.minimum(g, self.cap) if self.cap is not None else g

    def fit(self, m, y, p0=None):
        """Fit d, alpha to data (m = cumulative trials, y = error rate)."""
        m = np.asarray(m, float).ravel()
        y = np.asarray(y, float).ravel()

        if p0 is None:
            a0, d0 = _loglog_seed(m, y)
            p0 = [max(d0, 1e-6), a0]

        popt, _ = curve_fit(self._curve, m, y, p0=p0,
                            bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=20000)
        d, alpha = popt
        rss = float(np.sum((self._curve(m, *popt) - y) ** 2))
        self.params = {'d': float(d), 'alpha': float(alpha),
                       'cap': self.cap, 'rss': rss}
        self.fitted = True
        return self

    def predict(self, m):
        """Predicted error rate at trial counts ``m`` using the fitted model."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._curve(m, self.params['d'], self.params['alpha'])

    def get_params(self):
        """Return fitted parameters."""
        if not self.fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.params.copy()


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


class ConvergenceCurve():
    def __init__(self, distribution, model, asmp_error=None, n_test=4096,
                 train_size = 2 ** np.arange(1, 15, 1), n_repeat=128):
        """
        Args:
            distribution: callable that generates data.
            model: callable that returns model instance.
            asmp_error: expected asymptotic error rate.
            n_test: number of test samples.
            n_repeat: number of repeats for empirical error estimation.
        """
        self.distribution = distribution
        self.model = model
        self.asmp_error = asmp_error
        self.n_test = n_test
        self.train_size = train_size
        self.n_repeat = n_repeat

        self.X_D, self.y_D = distribution(n_test)

    def run(self):
        error = np.zeros(len(self.train_size))
        error_sd = np.zeros(len(self.train_size))
        for i in range(len(self.train_size)):
            m = self.train_size[i]
            test_error = []
            for _ in range(self.n_repeat):
                # sample data
                X_S, y_S = self.distribution(m)

                # fit a linear model
                lc = self.model()
                lc.fit(X_S, y_S)

                # predict on the test set
                pred = lc.predict(self.X_D)
                test_error.append(1 - np.mean(pred == self.y_D))                

            error[i] = np.mean(test_error)
            error_sd[i] = np.std(test_error)

        self.error = error
        self.error_sd = error_sd
        return self.train_size, self.error, self.error_sd

    def plot_error(self, ax, log_scale=True):
        # plot generalization error as a function of training set size
        ax.errorbar(self.train_size, self.error, yerr=self.error_sd,
                    marker='o', markersize=6, capsize=4)

        if log_scale:
            ax.set_xscale('log', base=2)
            ax.set_yscale('log', base=2)
            ax.minorticks_off()
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(style='plain', axis='y')

        ax.set_xlabel('training set size (m)')
        ax.set_ylabel(r'$ \epsilon = L_D(h)$')

    def plot_gap(self, ax, piecewise=False):
        # plot generalization gap as a function of log training set size
        log_m = np.log2(self.train_size)
        if self.asmp_error is None:
            print('estimate asymptotic error rate from data')
            asmp_error = self.error[-1]
            error_gap = self.error - asmp_error

            log_m = log_m[:-1]  # exclude last point
            log_gap = np.log2(error_gap[:-1])
        else:
            error_gap = self.error - self.asmp_error            
            log_gap = np.log2(error_gap)

        l = ax.plot(log_m, log_gap, marker='o', markersize=6)
        ax.set_xlabel('$\log_2(m)$')
        ax.set_ylabel(r'$\log |\epsilon - \epsilon^*|$')

        if not piecewise:
            # fit a linear function to the error gap
            coeffs = np.polyfit(log_m, log_gap, 1)
            a, b = coeffs
            ax.plot(log_m, a * log_m + b, color=l[0].get_color(),
                    linestyle='--', label=f'$%.2f * \log_2(m) + %.2f$' % (a, b))
        else:
            # fit a piecewise linear function to the error gap
            pwl = PiecewiseLinear()
            pwl.fit(log_m, log_gap)
            x0, m2, b2 = pwl.params['x0'], pwl.params['m2'], pwl.params['b2']
            ax.plot(log_m, m2 * log_m + b2, linestyle='--', color=l[0].get_color(),
                    label=f'$%.2f * \log_2(m) + %.2f$' % (m2, b2))

        ax.legend()