import numpy as np
from itertools import combinations_with_replacement


def polynomial_expend(X, degree):
    """
    Expand features to polynomial space.

    Args:
        X: Input data of shape (m, n) where m is number of samples, n is number of features
        degree: Degree of polynomial expansion (k)

    Returns:
        Expanded feature matrix of shape (m, num_expanded_features)
    """
    m, n = X.shape

    # Generate all combinations of features up to the specified degree
    features = []

    # Start with degree 1 (original features)
    for d in range(1, degree + 1):
        # Get all combinations with replacement for degree d
        for combo in combinations_with_replacement(range(n), d):
            # Compute the product of features for this combination
            feature_col = np.ones(m)
            for feature_idx in combo:
                feature_col *= X[:, feature_idx]
            features.append(feature_col)

    return np.column_stack(features)


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

    def sample(self, m, noise_rate=0.05, seed=None, both_classes=False):
        if seed is not None:
            np.random.seed(seed)

        while True:
            # Sample M points in N-dimensional space bounded by [-k, k]
            X = np.random.uniform(-self.k, self.k, size=(m, self.n))

            # Label points based on saved decision boundary
            y = (X @ self.a + self.b >= 0).astype(int)

            # Add label noise
            n_flip = int(noise_rate * m)
            flip_indices = np.random.choice(m, n_flip, replace=False)
            y[flip_indices] = 1 - y[flip_indices]

            # Check if both classes are present
            if not both_classes or (len(np.unique(y)) == 2):
                break

        # Shuffle
        indices = np.arange(m)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        return X, y

    def __call__(self, m, noise_rate=0.05, seed=None, both_classes=False):
        return self.sample(m, noise_rate, seed, both_classes)

    def get_boundary(self):
        return self.a.copy(), self.b