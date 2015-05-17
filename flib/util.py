"""
Utilities and small functions used in the rest of flib.
"""

import numpy as np 
from functools import wraps 


def coerce(f):
    """
    Decorator for coercing arguments to the type indicated by their annotation.
    """
    def coerce_arg(name, arg):
        """
        Convert the argument with the given name to its annotated type, if it 
        isn't already that type.
        """
        _type = f.__annotations__.get(name, None)
        if _type and not isinstance(arg, _type):
            if _type is np.ndarray:
                return np.array(arg)
            else:
                return _type(arg)
        else:
            return arg 

    # TODO: complete this


def ensure_shape(func):
    """
    Decorator which ensures arguments passed to the function are of the same
    shape, in order to avoid unwanted broadcasting.
    """
    # TODO: Specify which arguments to check
    # TODO: Finish this

