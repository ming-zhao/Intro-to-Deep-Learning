import numpy as np
from .Regressor import Regressor

"""
A simple implementation by assuming known beta
"""

class Variational(Regressor):
    """
    variational bayesian estimation the parameters
    p(w,alpha|X,t)
    ~ q(w)q(alpha)
    = N(w|w_mean, w_var)Gamma(alpha|a,b)

    Attributes
    ----------
    a : float
        a parameter of variational posterior gamma distribution
    b : float
        another parameter of variational posterior gamma distribution
    w_mean : (n_features,) ndarray
        mean of variational posterior gaussian distribution
    w_var : (n_features, n_feautures) ndarray
        variance of variational posterior gaussian distribution
    n_iter : int
        number of iterations performed
    """

    def __init__(self, beta=1., a0=1., b0=1.):
        """
        construct variational linear regressor
        Parameters
        ----------
        beta : float
            precision of observation noise
        a0 : float
            a parameter of prior gamma distribution
            Gamma(alpha|a0,b0)
        b0 : float
            another parameter of prior gamma distribution
            Gamma(alpha|a0,b0)
        """
        self.beta = beta
        self.a0 = a0
        self.b0 = b0

    def _fit(self, X, t, iter_max=100):
        assert X.ndim == 2
        assert t.ndim == 1
        self.a = self.a0 + 0.5 * np.size(X, 1) # (10.94)
        self.b = self.b0
        I = np.eye(np.size(X, 1))
        for i in range(iter_max):
            param = self.b
            self.w_var = np.linalg.inv(
                self.a * I / self.b
                + self.beta * X.T @ X) # (10.101)
            self.w_mean = self.beta * self.w_var @ X.T @ t  # (10.100)
            self.b = self.b0 + 0.5 * (
                np.sum(self.w_mean ** 2)
                + np.trace(self.w_var))  # (10.103)
            if np.allclose(self.b, param):
                break
        self.n_iter = i + 1

    def _predict(self, X, return_std=False):
        assert X.ndim == 2
        y = X @ self.w_mean  #(10.105)
        if return_std:
            y_var = 1 / self.beta + np.sum(X @ self.w_var * X, axis=1) # (10.106)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y
