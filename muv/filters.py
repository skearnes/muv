"""
MUV filters.
"""
import numpy as np

from rdkit import Chem


class EmbeddingFilter(object):
    """
    Remove compounds that are not adaquately embedded in another dataset.

    Parameters
    ----------
    threshold : float
        Embedding threshold; the maximum distance allowed between a
        compound and its n_neighbors nearest neighbors in another dataset.
    n_neighbors : int, optional (default 500)
        The number of nearest neighbors used to calculate the embedding.
    """
    def __init__(self, threshold, n_neighbors=500):
        self.threshold = threshold
        self.n_neighbors = n_neighbors

    def filter(self, d):
        """
        Filter the examples in set A that are not adaquately embedded in
        set B.

        Parameters
        ----------
        d : ndarray
            Distance matrix between examples in set A (first axis) and
            examples in set B (second axis).
        """
        cut = np.sort(d, axis=1)[:, self.n_neighbors - 1]
        keep = np.where(cut <= self.threshold)[0]
        return keep


class CompoundFilter(object):
    """
    Base class for removing specific compounds from a dataset.

    Parameters
    ----------
    exclude : array_like, optional
        Identifiers to exclude.
    """
    def __init__(self, exclude=None):
        if exclude is None:
            exclude = []
        self.exclude = exclude

    def add(self, exclude):
        """
        Add additional identifiers to the filter list.

        Parameters
        ----------
        exclude : array_like
            Identifiers to exclude.
        """
        self.exclude = np.concatenate((self.exclude, exclude))

    def filter(self, mols):
        """
        Filter a set of molecules.

        Parameters
        ----------
        mols : iterable
            Molecules.
        """
        keep = []
        for mol in mols:
            identifer = self.get_identifier(mol)
            if identifer not in self.exclude:
                keep.append(mol)
        keep = np.asarray(keep)
        return keep

    def get_identifier(self, mol):
        """
        Extract the relevant identifier from a molecule.

        Parameters
        ----------
        mol : Mol
            Molecule.
        """
        raise NotImplementedError


class NameFilter(CompoundFilter):
    """
    Remove specific compounds (matched by compound name) from a dataset.

    Parameters
    ----------
    exclude : array_like
        SMILES strings to exclude.
    """
    def get_identifier(self, mol):
        """
        Extract the name from a molecule.

        Parameters
        ----------
        mol : Mol
            Molecule.
        """
        name = mol.GetProp('_Name')
        assert name
        return name


class SmilesFilter(CompoundFilter):
    """
    Remove specific compounds (matched by canonical isomeric SMILES string)
    from a dataset.

    Parameters
    ----------
    exclude : iterable
        SMILES strings to exclude.
    """
    def get_identifier(self, mol):
        """
        Extract the name from a molecule.

        Parameters
        ----------
        mol : Mol
            Molecule.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return smiles
