import numpy as np
from .Regressor import Regressor

class LeastSquares(Regressor):
    
    def __init__(self, alpha=0):
        self.alpha = alpha
    
    def _fit(self, X, t):
        eye = np.eye(np.size(X, 1))
        self.w = np.linalg.solve(self.alpha * eye + X.T @ X, X.T @ t)
        self.var = np.mean(np.square(X @ self.w - t))  # beta of maximal likelihood (3.21)
        return self

    def _predict(self, X):
        return X @ self.w
    
    def _predict_dist(self, X):
        t = X @ self.w
        t_std = np.sqrt(self.var) + np.zeros_like(t)
        return t, t_std