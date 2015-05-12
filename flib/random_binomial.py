"""
Implementation of a feature function which, given a hashable input, maps that
input to a random binary vector with a specified number of nonzero entries.

That is, the number of nonzero entries is fixed, but which entries are zero and
which entries are one is chosen randomly.
"""
import numpy as np 
from flib.abstract import BinaryFeature


class RandomBinomial(BinaryFeature):
    def __init__(self, length: int, num_active: int):
        self.length = length 
        self.num_active = num_active
        self.mapping = {}

    def generate(self):
        ret = np.zeros(self.length)
        ix = np.random.choice(np.arange(self.length), self.num_active, replace=False)
        ret[ix] = 1
        return ret 

    def func(self, x: int) -> np.ndarray:
        if x in self.mapping:
            return self.mapping[x]
        else:
            self.mapping[x] = self.generate()
            return self.mapping[x]
        
    def __call__(self, x):
        if hasattr(x, '__iter__'):
            return np.array([self.func(i) for i in x])
        else:
            return self.func(x)
