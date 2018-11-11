import numpy as np
from .Classifier import Classifier
from DesignMat.Polynomial import Polynomial
"""
Least squares classifier model
y = argmax_k X @ W
"""
class LeastSquares(Classifier):
    def __init__(self, W=None):
        self.W = W

    def _fit(self, x, t):
        self._check_input(x)
        X = Polynomial(1).dm(x)
        self._check_target(t)
        T = np.eye(int(np.max(t)) + 1)[t]
        self.W = np.linalg.pinv(X) @ T #(4.16)
        return self

    def _predict(self, x):
        X = Polynomial(1).dm(x)
        return np.argmax(X @ self.W, axis=-1)