"""
Implementing various distance measures, i.e., norms.
"""
import numpy as np 



def hamming_distance(a, b):
    """
    The Hamming distance (a discrete norm between binary-valued arrays).
    """
    return np.sum(a != b)


def l1_norm(x):
    """
    The L1-norm, a continuous valued norm.
    """
    return np.sum(np.abs(x))


def l2_norm(x):
    """
    The L2-norm, a continuous valued norm.
    """
    return np.sqrt(np.sum(np.abs(x)**2))


# TODO: Why is this faster than the built-in `np.linalg.norm`?
def lp_norm(x, p):
    """
    The general LP-norm, a continuous valued norm.
    """
    return np.power(np.sum(np.power(np.abs(x), p)), 1/p)


# TODO: Divergence or distance?
def l1_divergence(a, b):
    """
    The L1 di, a continuous valued norm between two arrays of the same shape.
    """
    return np.sum(np.abs(a - b))


# TODO: Divergence or distance?
def l2_divergence(a, b):
    """
    The L2-norm, a continuous valued norm between two arrays of the same shape.
    """
    return np.sqrt(np.sum(np.abs(a - b)**2))