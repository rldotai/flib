"""
Implementation of a feature which maps integers to binary-valued vectors, with
the resulting vector having all entries zero except at the index indicated by
the supplied integer. 

Can be used to represent the tabular case in terms of arrays, for example.
"""
import numpy as np
from flib.abstract import UnaryFeature

class Int2Unary(UnaryFeature):
    def __init__(self, length):
        self.length = length
        self._array = np.eye(length)

    def __call__(self, x):
        return self._array[x]