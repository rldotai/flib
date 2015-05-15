"""
Code that was written but that was replaced or became obsolete, and is on its
way to removal, but is stored here in case it becomes relevant again.
"""
import numpy as np 
from fractions import gcd 

def sieve(n: int):
    """
    Implementation of the Sieve of Eratosthenes algorithm.

    Given an integer `n`, return a list of all primes less than or equal to it.
    """
    candidates = np.ones(n)     # nonzero index ==> possible prime!
    candidates[[0, 1]] = 0      # one and zero aren't prime.
    for i in range(2, n):
        if candidates[i] != 0:
            for j in range(2*i, n, i):
                candidates[j] = 0
    return list(np.nonzero(candidates))

def coprime(n: int):
    """Return a list of all numbers less than `n` that are coprime with `n`."""
    return [i for i in range(1, n) if gcd(i, n) == 1]