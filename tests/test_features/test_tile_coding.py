import pytest
import numpy as np 
import flib
from flib import TileCoder

def test_init():
    n_input = 4
    n_output = 16
    n_tiles = 1000
    f = TileCoder(n_input, n_output, n_tiles)
    f = TileCoder(n_input, n_output, n_tiles, scale=np.arange(n_input))
    f = TileCoder(n_input, n_output, n_tiles, table_size=512)
    f = TileCoder(n_input, n_output, n_tiles, random_seed=123)


# TODO: test that outputs fall within [0, n_tiles-1]
# TODO: test that it handles multiple simultaneous inputs properly
# TODO: test that outputs are, over the possible inputs, suitably random
# TODO: test that varying a single element of the input causes appropriate change in output