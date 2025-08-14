import numpy as np
import matplotlib.pyplot as plt


class PiecewiseLinear:
    def __init__(self):
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