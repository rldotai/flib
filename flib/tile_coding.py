"""
Hashed tile coding implemented in Python, following the reference 
implementations available as part of the RL Toolkit[0], but diverging somewhat
to make the tile coder easier to use.

0. http://rlai.cs.ualberta.ca/RLAI/RLtoolkit
1. http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/RLtoolkit/tilesUNHdoc.pdf
"""
import numpy as np 
from fractions import gcd 
from itertools import cycle
from toolz import take


class TileCoder:
    """
    A simple hashed tilecoder, following the documentation for the "UNH CMAC".

    .. note::
        This implementation uses the same randomized hash table for every 
        coordinate and tiling.
    """
    def __init__(self, n_input: int, n_output: int, n_tiles: int, scale=None, 
                 table_size=2048, random_seed=None):
        """
        Initialize the tile coder.

        Initialization proceeds by storing the input arguments and setting the
        optional arguments if they are unspecified.
        It then computes the displacement used to offset each separate tiling,
        and initializes the hashing function.

        Args:
            n_input (int): The number of inputs to be tiled, per-call.
            n_output (int): The number of outputs to be returned, per-call.
                This can also be thought of as the number of tilings.
            n_tiles (int): The total number of tiles available, that is, the 
                maximum value of any single entry returned by the coder.
            scale (np.ndarray, optional): The scaling applied to the input 
                prior to tiling.
            table_size (int, optional): The size of the hash table used by the
                hashing function, `hfunc`.
            random_seed (int, seq, or np.random.RandomState, optional): The 
                seed used to initialize random number generation used by the 
                tile coder.
        """

        self.n_input = n_input
        self.n_output = n_output
        self.n_tiles = n_tiles
        self.table_size = table_size

        # Get the seed for pseudorandom number generator
        # This may not be the best way to initialize, but it's consistent
        if isinstance(random_seed, np.random.RandomState):
            self.random_seed = random_seed.get_state()
        else:
            self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)

        if scale is None:
            self.scale = np.ones(n_input)
        else:
            assert(len(scale) == n_input)
            self.scale = np.array(scale)

        # Compute displacement vector, and then the offsets for each tiling
        self.dvec = self.get_displacement(n_input, n_output)
        self.dmat = np.outer(np.arange(self.n_output), self.dvec)
        # Set up the hashing function
        self.hfunc = SimpleHash(self.table_size, self.n_tiles) 


    def apply(self, array):
        """
        Map the input array to its tile coding representation.

        Essentially, this proceeds by first getting the integer coordinates of
        the input array (subject to scaling), then by offsetting the 
        coordinates according to the displacement vector for each tiling.
        Then, the displaced coordinates are hashed using `hfunc`, and the 
        resulting hashed values are summed modulo `n_tiles` to produce the 
        indices of the active tiles to be used as features.

        Args:
            array (np.ndarray): The array to be tiled.
                Must be of length `n_input`, or else an exception is raised.

        Returns:
            ret (np.ndarray): An array of length `n_output`, whose entries 
                correspond to the indices of the active tiles.
        """
        if len(array) != self.n_input:
            raise ValueError("Incompatible array with length", len(array))
        x = np.floor_divide(array, self.scale).astype(np.int)
        v = x - ((x - self.dmat) % self.n_output)
        a = np.apply_along_axis(self.hfunc, axis=0, arr=v)
        ret = np.sum(a, axis=1) % self.n_tiles
        return ret 

    def __call__(self, array):
        """
        Wraps `self.apply`, with slightly different behavior to accomodate
        multidimensional inputs to allow for tile-coding multiple inputs at
        the same time.

        Args:
            array (np.ndarray): The input to be tiled

        Returns:
            (np.ndarray): Array whose entries correspond to the indices of the
                active tiles.
        """
        # Not sure if this is the best way to achieve broadcasting...
        if array.ndim > 1:
            return np.apply_along_axis(self.apply, axis=1, arr=array)
        else:
            return self.apply(array)

    @staticmethod
    def get_displacement(n_input, n_tilings):
        """
        Get the displacement vector to use in offsetting the tilings.

        Essentially, we look for numbers less than `n_tilings//2` that are 
        coprime with `n_tilings`. 
        If we can find at least `n_input` of them, we just take the first 
        `n_input`. If there are fewer such viable numbers, we instead cycle
        through the candidates, ensuring we repeat as seldom as possible.

        ..note::
            It's recommended by the CMAC people to just increase the number of 
            tilings when there aren't enough candidate values for the 
            displacement vector.
        """
        viable = [i for i in range(1, n_tilings//2) if gcd(i, n_tilings) == 1]
        ret = list(take(n_input, cycle(viable)))
        return np.array(ret)

class SimpleHash:
    def __init__(self, n_entries, high, random_seed=None):
        """
        Initialize a hash table with `n_entries` total size, and with each 
        entry in the table an integer drawn uniformly at random from (0, high).
        """
        self.n_entries = n_entries
        self.high = high

        # Get the seed for pseudorandom number generator
        # This may not be the best way to initialize, but it's consistent
        if isinstance(random_seed, np.random.RandomState):
            self.random_seed = random_seed.get_state()
        else:
            self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)
        
        # Generate the hash table
        self.table = self.random_state.random_integers(0, high, size=n_entries)

    def __call__(self, x):
        """
        Return the value(s) of the hash table associated with `x`.

        Args:
            x (int, Seq[int]): the indices of the table entries to look up.

        Returns:
            int or Array[int]: the value(s) of the hash table associated with `x`
        """
        return self.table[x % self.n_entries]
