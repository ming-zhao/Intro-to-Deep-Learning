import numpy as np
from .Regressor import Regressor

class Empirical(Regressor):
    def __init__(self, alpha=1., beta=1.):
        self.alpha = alpha
        self.beta = beta

    def _fit(self, X, t, max_iter=100):
        M = X.T @ X
        eigenvalues = np.linalg.eigvalsh(M)
        eye = np.eye(np.size(X, 1))
        N = len(t)
        for _ in range(max_iter):
            params = [self.alpha, self.beta]
            w_precision = self.alpha * eye + self.beta * X.T @ X
            w_mean = self.beta * np.linalg.solve(w_precision, X.T @ t)
            gamma = np.sum(eigenvalues / (self.alpha + eigenvalues))
            self.alpha = float(gamma / np.sum(w_mean ** 2).clip(min=1e-10))
            self.beta = float(
                (N - gamma) / np.sum(np.square(t - X @ w_mean))
            )
            if np.allclose(params, [self.alpha, self.beta]):
                break
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(w_precision)
        return self

    def log_evidence(self, X, t):
        M = X.T @ X
        return 0.5 * (
            len(M) * np.log(self.alpha)
            + len(t) * np.log(self.beta)
            - (self.beta * np.square(t - X @ self.w_mean).sum() + self.alpha * np.sum(self.w_mean ** 2))
            - np.linalg.slogdet(self.w_precision)[1]
            - len(t) * np.log(2 * np.pi)
        )

    def _predict(self, X, sample_size=None):
        if isinstance(sample_size, int):
            w_sample = np.random.multivariate_normal(
                self.w_mean, self.w_cov, size=sample_size
            )
            return X @ w_sample.T
        return X @ self.w_mean

    
    def _predict_dist(self, X):
        return X @ self.w_mean, np.sqrt(1 / self.beta + np.sum(X @ self.w_cov * X, axis=1))