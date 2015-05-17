"""
Implementation of a feature which maps integers to an array containing
their binary representation, modulo the length of the array.
"""
import numpy as np 
from functools import partial
from flib.abstract import BinaryFeature


class Int2Bin(BinaryFeature):
    """
    Convert integer to its bit vector representation.

    On initialization, it precomputes an array which is used to extract the 
    individual bits of each integer. 

    .. note ::
        The function only extracts the first `length` bits of the integers 
        supplied to it, so it's essentially converting modulo `length`.
        For example, `2**length` will have the same representation as `0`, and
        `-1` will be represented the same way as `2**length -1`.
    """
    def __init__(self, length: int):
        super().__init__(1, length)
        # Precompute the array for converting integers to bit vectors
        self._array = (1 << np.arange(length))

        # Create the function to apply to inputs
        self.apply = partial(self.func, self._array)

    @staticmethod
    def func(array, x):
        return ((x & array) > 0)

    def __call__(self, x):
        # TODO: Add proper vectorization here / as an abstract mixin
        x = np.array(x)
        if x.ndim > 0:
            ret = np.empty(shape=(x.size, self.n_output), dtype=np.uint8)
            for ix, i in enumerate(x.flat):
                ret[ix] = self.apply(i)
            return ret
        else:
            return self.apply(x)