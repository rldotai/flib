"""
Abstract base classes for different kinds of feature.
"""
import numpy as np 


class BinaryFeature:
    """
    Base class for binary valued features.
    """
    def __init__(self, length):
        self.length = length 

    def __len__(self) -> int:
        return self.length


class UnaryFeature:
    def __init__(self, length):
        self.length = length

    def __len__(self) -> int:
        return self.length