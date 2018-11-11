import numpy as np

"""
Base class for regressors
"""
class Regressor(object):
    
    def fit(self, X, t, **kwargs):
        """
        estimates parameters given training dataset
        Parameters
        ----------
        X : (sample_size, n_features) np.ndarray
            design matrix based on training data input
        t : (sample_size,) np.ndarray
            training data target
        """
        self._check_input(X)
        self._check_target(t)
        if hasattr(self, "_fit"):
            return self._fit(X, t, **kwargs)
        else:
            raise NotImplementedError

    def predict(self, X, **kwargs):
        """
        predict outputs of the model
        Parameters
        ----------
        X : (sample_size, n_features) np.ndarray
            samples to predict their output
        Returns
        -------
        y : (sample_size,) np.ndarray
            prediction of each sample
        """
        self._check_input(X)
        if hasattr(self, "_predict"):
            return self._predict(X, **kwargs)
        else:
            raise NotImplementedError
            
    def predict_dist(self, X, **kwargs):
        """
        predict outputs of the model
        Parameters
        ----------
        X : (sample_size, n_features) np.ndarray
            samples to predict their output
        Returns
        -------
        y : (sample_size,) np.ndarray
            prediction of each sample
        """
        self._check_input(X)
        if hasattr(self, "_predict"):
            return self._predict_dist(X, **kwargs)
        else:
            raise NotImplementedError            

    def _check_input(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X(input) is not np.ndarray")
            
        if X.ndim != 2:
            raise ValueError("X(input) is not two dimensional array")
            
        if hasattr(self, "n_features") and self.n_features != np.size(X, 1):
            raise ValueError(
                "mismatch in dimension 1 of X(input) "
                "(size {} is different from {})"
                .format(np.size(X, 1), self.n_features)
            )

    def _check_target(self, t):
        if not isinstance(t, np.ndarray):
            raise ValueError("t(target) must be np.ndarray")
            
        if t.ndim != 1:
            raise ValueError("t(target) must be one dimenional array")