import numpy as np
from .Regressor import Regressor

"""
Bayesian regression model
w ~ N(w|0, alpha^(-1)I)
y = X @ w
t ~ N(t|X @ w, beta^(-1))
"""
class Bayesian(Regressor):
    def __init__(self, alpha=1., beta=1.):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None

    def _fit(self, X, t):
        if self.w_mean is not None:
            prior_mean = self.w_mean
        else:
            prior_mean = np.zeros(np.size(X, 1))
            
        if self.w_precision is not None:
            prior_precision = self.w_precision
        else:
            prior_precision = self.alpha * np.eye(np.size(X, 1))
            
        w_precision = prior_precision + self.beta * X.T @ X  # calculate S_N^{-1} (3.54)
        
        w_mean = np.linalg.solve(
            w_precision,
            prior_precision @ prior_mean + self.beta * X.T @ t
        )   # calculate m_N (3.53)
        
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)
        return self
    
    def _predict(self, X, sample_size=None):
        if isinstance(sample_size, int):
            w_sample = np.random.multivariate_normal(
                self.w_mean, self.w_cov, size=sample_size
            )
            return X @ w_sample.T
        return X @ self.w_mean
    
    def _predict_dist(self, X):
        return X @ self.w_mean, np.sqrt(1 / self.beta + np.sum(X @ self.w_cov * X, axis=1))  #(3.59)