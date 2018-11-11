"""
polynomial design matrix

Example
=======
x = [[a, b],
     [c, d]]
y = polynomial(degree=2).dm(x)
y =
[[1, a, b, a^2, a*b, b^2],
 [1, c, d, c^2, c*d, d^2]]
"""
import itertools
import functools
import numpy as np

class Polynomial(object):
    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree

    def dm(self, x):
        """
        ----------
        x : (sample_size, n) ndarray 
            input array
        Returns
        -------
        output : (sample_size, 1 + nC1 + ... + nCd) ndarray
            design matrix
        """
        if not isinstance(x, np.ndarray):
            raise ValueError("x(input) is not np.ndarray")

        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        
        basis = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                basis.append(functools.reduce(lambda x, y: x * y, items))
                
        return np.asarray(basis).transpose()
    