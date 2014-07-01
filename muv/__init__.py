"""
Miscellaneous utilities.
"""
import numpy as np

from muv.descriptors import MUVDescriptors


class MUV(object):
    """
    Generate maximum unbiased validation (MUV) datasets for virtual
    screening as described in Rohrer and Baumann, J. Chem. Inf. Model.
    2009, 49, 169-184.
    """
    def create_dataset(self, actives, decoys, n_actives=30,
                       n_decoys_per_active=500):
        """
        Create a MUV dataset.

        Parameters
        ----------
        actives : iterable
            Potential actives.
        decoys : iterable
            Potential decoys.
        """

        # calculate descriptors
        ad = self.calculate_descriptors(actives)
        dd = self.calculate_descriptors(decoys)

        # apply filters

        # select actives and decoys


    def calculate_descriptors(self, mols):
        """
        Calculate MUV descriptors for molecules.

        Parameters
        ----------
        mols : iterable
            Molecules.
        """
        describer = MUVDescriptors()
        x = []
        for mol in mols:
            x.append(describer(mol))
        x = np.asarray(x)
        return x


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
