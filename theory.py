import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from utils import PiecewiseLinear

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

        # no error bars: this is the expected error, which can be estimated
        # to arbitrary precision with more simulation repeats
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
