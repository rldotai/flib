"""
Kanerva coding, also known as sparse distributed representations.
"""
import numpy as np 
from functools import partial
from flib.abstract import FunctionalFeature
from flib.norm import hamming_distance


class Hamming(FunctionalFeature):
    """
    Hamming distance between arrays and a prototype array.
    """
    def __init__(self, prototype):
        self.prototype = np.array(prototype)
        super().__init__(len(self.prototype), 1)


        self.apply = partial(hamming_distance, self.prototype)