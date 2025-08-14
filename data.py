import numpy as np


class Gaussian2D:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'mean_0': np.array([0.0, 0.0]),
                'cov_0': np.array([[1.0, 0.2],
                                   [0.2, 0.8]]),
                'mean_1': np.array([2.0, 1.5]),
                'cov_1': np.array([[1.0, -0.2],
                                   [-0.2, 1.0]])
            }
        else:
            self.params = params

    def sample(self, n, seed=None):
        if seed is not None:
            np.random.seed(seed)

        X0 = np.random.multivariate_normal(self.params['mean_0'], self.params['cov_0'], n)
        X1 = np.random.multivariate_normal(self.params['mean_1'], self.params['cov_1'], n)
        X = np.vstack([X0, X1])
        y = np.hstack([np.zeros(n), np.ones(n)])

        shuffle_indices = np.arange(len(X))
        np.random.shuffle(shuffle_indices)
        X = X[shuffle_indices]
        y = y[shuffle_indices]

        return X, y

    def __call__(self, n, seed=None):
        return self.sample(n, seed)


class LinearSeparable:
    def __init__(self, n, k=1.0, seed=None):
        self.n = n
        self.k = k
        self.a = None
        self.b = None

        if seed is not None:
            np.random.seed(seed)

        # Generate decision boundary with balanced classes
        while True:
            # Sample test points to check balance
            X_test = np.random.uniform(-k, k, size=(1024, n))

            # Generate random decision boundary: a^T x + b >= 0
            a = np.random.randn(n)
            a = a / np.linalg.norm(a)  # normalize
            b = np.random.uniform(-k, k)

            # Check class balance on test points
            y_test = (X_test @ a + b >= 0).astype(int)
            class_ratio = np.mean(y_test)
            if 0.4 <= class_ratio <= 0.6:
                self.a = a
                self.b = b
                break

    def sample(self, m, noise_rate=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Sample M points in N-dimensional space bounded by [-k, k]
        X = np.random.uniform(-self.k, self.k, size=(m, self.n))

        # Label points based on saved decision boundary
        y = (X @ self.a + self.b >= 0).astype(int)

        # Add label noise
        n_flip = int(noise_rate * m)
        flip_indices = np.random.choice(m, n_flip, replace=False)
        y[flip_indices] = 1 - y[flip_indices]

        # Shuffle
        indices = np.arange(m)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        return X, y

    def __call__(self, m, noise_rate=0.05, seed=None):
        return self.sample(m, noise_rate, seed)

    def get_boundary(self):
        return self.a.copy(), self.b