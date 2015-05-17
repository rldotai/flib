"""
Implementing general "DropOut" style functions, for regularizing or sparsifying
features stochastically.
"""
import numpy as np 
from flib.abstract import Feature 


class DropOut(Feature):
    """
    A simple dropout implementation. Given an array, returns an array of the 
    same size and shape with its entries either unchanged or set to zero with
    probability `p`. 
    """
    def __init__(self, n_input, p, **kwargs):
        if not 0 <= p <= 1:
            raise ValueError("Invalid value for `p`:", p)
        super().__init__(n_input, n_input, **kwargs)
        self.p = p

    def apply(self, x):
        ret = x.copy()
        ret[[np.random.random(size=self.n_input) < self.p]] = 0
        return ret 
