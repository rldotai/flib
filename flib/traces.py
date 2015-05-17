"""
Implementation of trace-style features
"""
import numpy as np 
from flib.abstract import Feature


class AccumulatingTrace(Feature):
    """
    Accumulating traces, which retain a memory of past inputs that decays 
    at a specified rate.

    Given a 1-D binary valued array, the trace first decays according to 
    `trace *= decay`, and is then incremented by `1` at every index where the
    input array was nonzero.
    """
    def __init__(self, n_input, decay):
        if 0 > decay or decay > 1:
            raise ValueError("Invalid decay parameter:", decay)
        super().__init__(n_input, n_input, dtype=np.float)
        self.decay = decay
        self._array = np.zeros(n_input, dtype=self.dtype)

        def func(x):
            self._array *= decay
            self._array += (x != 0)
            return self._array

        self.apply = func


class ReplacingTrace(Feature):
    """
    Replacing traces, which retain a memory of past inputs that decays 
    at a specified rate.

    Given a 1-D binary valued array, the trace first decays according to 
    `trace *= decay`, and is then set to `1` at every index where the input 
    array was nonzero.
    """
    def __init__(self, n_input, decay):
        if 0 > decay or decay > 1:
            raise ValueError("Invalid decay parameter:", decay)
        super().__init__(n_input, n_input, dtype=np.float)
        self.decay = decay
        self._array = np.zeros(n_input, dtype=self.dtype)

        def func(x):
            self._array *= decay
            self._array[np.nonzero(x)] = 1
            return self._array

        self.apply = func

