import pytest
import flib
from flib import Int2Bin


def test_Int2Bin():
    integers = [i for i in range(256)]
    length = 10

    func = Int2Bin(length)

    # individual outputs
    output = [func(i) for i in integers]

    # over all the integers at once
    out_array = func(integers)