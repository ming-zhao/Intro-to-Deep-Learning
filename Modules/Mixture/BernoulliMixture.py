import numpy as np
from scipy.misc import logsumexp
from Mixture import Mixture

class BernoulliMixture(Mixture):
    """
    p(x|pi,mu)
    = sum_k pi_k mu_k^x (1 - mu_k)^(1 - x)
    """
    def __init__(self, n_components=3, mu=None, pi=None):
        """
        construct mixture of Bernoulli

        Parameters
        ----------
        n_components : int
            number of bernoulli component
        mu : (n_components, ndim) np.ndarray
            probability of value 1 for each component
        coef : (n_components,) np.ndarray
            mixing coefficients
        """
        super().__init__()
        assert isinstance(n_components, int)
        self.n_components = n_components
        self.mu = mu
        self.pi = pi

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, np.ndarray):
            assert mu.ndim == 2
            assert np.size(mu, 0) == self.n_components
            assert (mu >= 0.).all() and (mu <= 1.).all()
            self.ndim = np.size(mu, 1)
            self.parameter["mu"] = mu
        else:
            assert mu is None
            self.parameter["mu"] = None

    @property
    def pi(self):
        return self.parameter["pi"]

    @pi.setter
    def pi(self, pi):
        if isinstance(pi, np.ndarray):
            assert pi.ndim == 1
            assert np.allclose(pi.sum(), 1)
            self.parameter["pi"] = pi
        else:
            assert pi is None
            self.parameter["pi"] = np.ones(self.n_components) / self.n_components

    def _log_bernoulli(self, X):
        np.clip(self.mu, 1e-10, 1-1e-10, out=self.mu)
        return (
            X[:, None, :] * np.log(self.mu)
            + (1 - X[:, None, :]) * np.log(1 - self.mu)
        ).sum(axis=-1)

    def _fit(self, X):
        self.mu = np.random.uniform(0.25, 0.75, size=(self.n_components, np.size(X, 1)))
        params = np.hstack((self.mu.ravel(), self.pi.ravel()))
        while True:
            resp = self._expectation(X)
            self._maximization(X, resp)
            new_params = np.hstack((self.mu.ravel(), self.pi.ravel()))
            if np.allclose(params, new_params):
                break
            else:
                params = new_params

    def _expectation(self, X):
        log_resps = np.log(self.pi) + self._log_bernoulli(X)
        log_resps -= logsumexp(log_resps, axis=-1)[:, None]
        resps = np.exp(log_resps)      # (9.56)
        return resps

    def _maximization(self, X, resp):
        Nk = np.sum(resp, axis=0)      # (9.57)
        self.mu = (X.T @ resp / Nk).T  # (9.58)
        self.pi = Nk / len(X)          # (9.60)

    def predict(self, X):
        """
        classify input
        max_z p(z|x, theta)

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input

        Returns
        -------
        output : (sample_size,) ndarray
            corresponding cluster index
        """
        return np.argmax(self.predict_prob(X), axis=1)

    def predict_prob(self, X):
        """
        posterior probability of cluster p(z|x,theta)

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input

        Returns
        -------
        output : (sample_size, n_components) ndarray
            posterior probability of cluster
        """
        return self._expectation(X)
