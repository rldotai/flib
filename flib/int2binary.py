"""
Implementation of a feature which maps integers to an array containing
their binary representation, modulo the length of the array.
"""
import numpy as np 
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
        super().__init__(length)

        # Precompute the array for converting integers to bit vectors
        self._array = (1 << np.arange(self.length))

    def __call__(self, x) -> np.ndarray:
        # TODO: vectorize this properly
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 0:
            return ((x & self._array) > 0).astype(np.uint8)
        else:
            return ((x[:,np.newaxis] & self._array) > 0).astype(np.uint8)