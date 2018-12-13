import numpy as np

class Classifier(object):
    def fit(self, X, t, **kwargs):
        self._check_input(X)
        self._check_target(t)
        if hasattr(self, "_fit"):
            self._fit(X, t, **kwargs)
        else:
            raise NotImplementedError
        return self

    def predict(self, X, **kwargs):
        self._check_input(X)
        if hasattr(self, "_predict"):
            return self._predict(X, **kwargs)
        else:
            raise NotImplementedError

    def prob(self, X, **kwargs):
        self._check_input(X)
        if hasattr(self, "_prob"):
            return self._prob(X, **kwargs)
        else:
            raise NotImplementedError

    def softmax(self, a):
        a_max = np.max(a, axis=-1, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)            
    
    def sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5    
            
    def _check_input(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X(input) must be np.ndarray")
        if X.ndim != 2:
            raise ValueError("X(input) must be two dimensional array")
        if hasattr(self, "n_features") and self.n_features != np.size(X, 1):
            raise ValueError(
                "mismatch in dimension 1 of X(input) (size {} is different from {})"
                .format(np.size(X, 1), self.n_features)
            )

    def _check_target(self, t):
        if not isinstance(t, np.ndarray):
            raise ValueError("t(target) must be np.ndarray")
        if t.ndim != 1:
            raise ValueError("t(target) must be one dimensional array")
        if t.dtype != np.int:
            raise ValueError("dtype of t(target) must be np.int")
        if (t < 0).any():
            raise ValueError("t(target) must only has positive values")

    def _check_binary(self, t):
        if np.max(t) > 1 and np.min(t) >= 0:
            raise ValueError("t(target) must only has 0 or 1")

    def _check_binary_negative(self, t):
        n_zeros = np.count_nonzero(t == 0)
        n_ones = np.count_nonzero(t == 1)
        if n_zeros + n_ones != t.size:
            raise ValueError("t(target) must only has -1 or 1")
