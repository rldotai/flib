"""
Binary valued feature vectors.
"""

import numpy as np 


class BinaryFeature:
    def __init__(self, length):
        self.length = length 

    def __len__(self):
        return self.length


class Int2Bin(BinaryFeature):
    def __init__(self, length):
        super().__init__(length)

        # Precompute the array for converting integers to bit vectors
        self._array = (1 << np.arange(self.length))

    def __call__(self, x):
        a = np.array(x)[:, np.newaxis]
        return ((a & self._array) > 0).astype(np.uint8)


class RandomBinomial(BinaryFeature):
    def __init__(self, length, num_active):
        self.length = length 
        self.num_active = num_active
        self.mapping = {}

    def generate(self):
        ret = np.zeros(self.length)
        ix = np.random.choice(np.arange(self.length), self.num_active, replace=False)
        ret[ix] = 1
        return ret 

    def func(self, x):
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


class UnaryFeature:
    def __init__(self, length):
        self.length = length


class Int2Unary(UnaryFeature):
    def __init__(self, length):
        self.length = length
        self._array = np.eye(length)

    def __call__(self, x):
        return self._array[x]


def bitunpack(x):
    """A wrapper function for unpacking an arrays binary representation."""
    return np.array(x).view(np.uint8)