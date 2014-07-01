"""
Miscellaneous utilities.
"""
import numpy as np


class MUV(object):
    """
    Generate maximum unbiased validation (MUV) datasets for virtual
    screening as described in Rohrer and Baumann, J. Chem. Inf. Model.
    2009, 49, 169-184.
    """
    

def kennard_stone(d, k):
    """
    Use the Kennard-Stone algorithm to select k maximally separated
    examples from a dataset.

    See Kennard and Stone, Technometrics 1969, 11, 137-148.

    Algorithm
    ---------
    1. Choose the two examples separated by the largest distance. In the
        case of a tie, use the first examples returned by np.where.
    2. For the remaining k - 2 selections, choose the example with the
        greatest distance to the closest example among all previously
        chosen points.

    Parameters
    ----------
    d : ndarray
        Pairwise distance matrix between dataset examples.
    k : int
        Number of examples to select.
    """
    assert 1 < k < d.shape[0]
    chosen = []

    # choose initial points
    first = np.where(d == np.amax(d))
    chosen.append(first[0][0])
    chosen.append(first[1][0])
    d = np.ma.array(d, mask=np.ones_like(d, dtype=bool))

    # choose remaining points
    while len(chosen) < k:
        d.mask[:, chosen] = False
        d.mask[chosen] = True
        print d
        p = np.ma.argmax(np.ma.amin(d, axis=1))
        chosen.append(p)

    return chosen
