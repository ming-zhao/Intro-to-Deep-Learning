import numpy as np

"""
sigmoidal design matrix

1 / (1 + exp((m - x) @ c)
"""

class Sigmoidal(object):
    def __init__(self, mean, coef=1):
        
        if mean.ndim == 1:
            mean = mean[:, None]
        else:
            assert mean.ndim == 2
            
        if isinstance(coef, int) or isinstance(coef, float):
            if np.size(mean, 1) == 1:
                coef = np.array([coef])
            else:
                raise ValueError("mismatch of dimension")
        else:
            assert coef.ndim == 1
            assert np.size(mean, 1) == len(coef)
            
        self.mean = mean
        self.coef = coef

    def _sigmoid(self, x, mean):
        return np.tanh((x - mean) @ self.coef * 0.5) * 0.5 + 0.5

    def dm(self, x):
        if x.ndim == 1:
            x = x[:, None]
        else:
            assert x.ndim == 2
        assert np.size(x, 1) == np.size(self.mean, 1)
        
        basis = [np.ones(len(x))]
        for m in self.mean:
            basis.append(self._sigmoid(x, m))
        return np.asarray(basis).transpose()
