import numpy as np
from .Classifier import Classifier
from DesignMat.Polynomial import Polynomial

"""
Softmax regression model
aka multinomial logistic regression,
multiclass logistic regression, or maximum entropy classifier.

y = softmax(X @ W)
t ~ Categorical(t|y)
"""
class Logistic(Classifier):
    def _fit(self, x, t, max_iter=100, learning_rate=0.01):
        X = Polynomial(1).dm(x)
        self.n_classes = np.max(t) + 1
        T = np.eye(self.n_classes)[t]
        W = np.zeros((np.size(X, 1), self.n_classes))
        for _ in range(max_iter):
            W_prev = np.copy(W)
            y = self.softmax(X @ W)
            grad = X.T @ (y - T)
            W -= learning_rate * grad
            if np.allclose(W, W_prev):
                break
        self.W = W
        return self

    def _prob(self, x):
        X = Polynomial(1).dm(x)
        y = self.softmax(X @ self.W)
        return y

    def _predict(self, x):
        prob = self._prob(x)
        label = np.argmax(prob, axis=-1)
        return label

class BayesianLogistic(Logistic):
    def __init__(self, alpha=1.):
        self.alpha = alpha
        
    def _fit(self, x, t, max_iter=100):
        self._check_binary(t)
        X = Polynomial(1).dm(x)
        w = np.zeros(np.size(X, 1))
        eye = np.eye(np.size(X, 1))
        self.w_mean = np.copy(w)
        self.w_precision = self.alpha * eye
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self.sigmoid(X @ w)
            grad = X.T @ (y - t) + self.w_precision @ (w - self.w_mean)
            hessian = (X.T * y * (1 - y)) @ X + self.w_precision
            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w_mean = w
        self.w_precision = hessian
        return self

    def _prob(self, x):
        X = Polynomial(1).dm(x)
        mu_a = X @ self.w_mean
        var_a = np.sum(np.linalg.solve(self.w_precision, X.T).T * X, axis=1)
        y = self.sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
        print(y.shape)
        return y
    
    def _predict(self, x, threshold=0.5):
        proba = self._prob(x)
        label = (proba > threshold).astype(np.int)
        return label    