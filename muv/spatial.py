"""
Spatial statistics.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import numpy as np


def spread(d, t):
    """
    Calculate the spread between two sets of compounds.

    Given a matrix containing distances between two sets of compounds, A
    and B, calculate the fraction of compounds in set A that are closer
    than t to any compound in set B.

    Parameters
    ----------
    d : ndarray
        Distance matrix with compounds from set A on first axis.
    t : float
        Distance threshold.
    """
    p = np.mean(np.any(d < t, axis=1))
    return p
