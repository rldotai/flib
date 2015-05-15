"""
Tests for tile_coding.py
"""

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


def test_output_range():
    n_input = 4
    n_output = 16
    n_tiles = 1000
    f = TileCoder(n_input, n_output, n_tiles)

    low, high = 0, 100
    inputs = np.random.uniform(low, high, size=(1000, n_input))

    for i in inputs:
        out = f(i)
        assert(np.all(0 <= out))
        assert(np.all(n_tiles > out))

def test_output_shape():
    cases = 1000
    n_input = 4
    n_output = 16
    n_tiles = 1000
    f = TileCoder(n_input, n_output, n_tiles)

    low, high = 0, 100
    inputs = np.random.uniform(low, high, size=(cases, n_input))

    # Test individual outputs
    for i in inputs:
        out = f(i)
        assert(len(out) == n_output)

    # Test coding for multiple inputs at the same time
    outputs = f(inputs)
    assert(outputs.shape == (cases, n_output))    

def test_randomization():
    cases = 10000
    n_input = 4
    n_output = 16
    n_tiles = 1000
    f = TileCoder(n_input, n_output, n_tiles)

    low, high = 0, 100
    inputs = np.random.uniform(low, high, size=(cases, n_input))
    outputs = f(inputs)

    # Attempt to quantify how random the tile coding was
    tiles = outputs.flatten()
    total_tiles = len(tiles)
    nbins = 50
    hist, bins = np.histogram(tiles, bins=np.linspace(0, n_tiles, nbins))

    # More than three standard deviations away is a cause for worry
    zscore = np.abs(hist - hist.mean())/np.std(hist)
    assert(np.all(zscore < 3))


# TODO: test that varying a single element of the input causes appropriate change in output