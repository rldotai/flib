"""
Abstract base classes for different kinds of feature.
"""
import numpy as np 


class Feature:
    """
    Feature function base class.

    Implements various methods common to feature functions, which are generally
    the same across the various features in this library.
    """
    def __init__(self, n_input, n_output, *args, **kwargs):
        self.n_input = n_input
        self.n_output = n_output

        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']

        # Get the seed for pseudorandom number generator
        # This may not be the best way to initialize, but it's consistent
        random_seed = kwargs.get('random_seed', None)

        if isinstance(random_seed, np.random.RandomState):
            self.random_seed = random_seed.get_state()
        else:
            self.random_seed = random_seed

        # Set up the pseudorandom number generator
        self.random_state = np.random.RandomState(self.random_seed)

    def __call__(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if x.ndim > 1:
            return np.apply_along_axis(self.apply, axis=1, arr=x)
        else:
            return self.apply(x)

    def __len__(self) -> int:
        return self.n_output

    @property 
    def length(self):
        return self.n_output


class FunctionalFeature(Feature):
    """
    Base class for features that are essentially functional in nature, i.e.,
    they could be applied to arbitrary arrays, and have no side effects. 
    In order to ensure our feature pipeline is well-formed, this class provides 
    some of the various methods and properties common to such features.
    """
    def __init__(self, n_input, n_output, func=None, *args, **kwargs):
        """
        Initialize the functional feature, specifying the number of inputs, 
        outputs, and optionally the function to compute the resulting feature.

        Args:
            n_input (int) : The number of inputs the feature expects.
            n_output (int) : The number of outputs the feature will return.
            func (Callable, optional): The function that computes features.
        """
        super().__init__(n_input, n_output, *args, **kwargs)
        self.n_input = n_input
        self.n_output = n_output
        if func is not None:
            self.apply = func 

class OneToMany(Feature):
    """
    Base class for features which return multi-element arrays from inputs 
    consisting of a single element.
    """

class ManyToOne(Feature):
    """
    Base class for features which take arrays containing multiple elements and
    return single element arrays.
    """

class BinaryFeature(Feature):
    """
    Base class for binary valued features.
    """
    def __init__(self, n_input, n_output, *args, **kwargs):
        super().__init__(n_input, n_output)


class UnaryFeature(Feature):
    """
    Base class for unary features (i.e., those with a single nonzero bit).
    """
    def __init__(self, n_input, n_output, *args, **kwargs):
        super().__init__(n_input, n_output)