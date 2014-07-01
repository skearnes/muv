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


class PropertyFilter(object):
    """
    Remove compounds from a dataset that have properties outside an
    acceptable range.

    Parameters
    ----------
    min_value : float, optional
        Minimum property value.
    max_value : float, optional
        Maximum property value.
    allow_min : bool, optional (default True)
        Whether to allow a molecular property to equal min_value.
    allow_max : bool, optional (default True)
        Whether to allow a molecular property to equal max_value.
    """
    def __init__(self, min_value=None, max_value=None, allow_min=True,
                 allow_max=True):
        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must " +
                             "provided.")
        self.min_value = min_value
        self.max_value = max_value
        self.allow_min = allow_min
        self.allow_max = allow_max

    def get_prop(self, mol, prop=None):
        """
        Extract the relevant property from the molecule. Subclasses can
        override this method to return properties not accessible through
        Mol.GetProp.

        Parameters
        ----------
        mol : Mol
            Molecule.
        prop : str, optional
            Molecular property to extract using Mol.GetProp.
        """
        if prop is not None:
            return mol.GetProp(prop)
        else:
            raise NotImplementedError

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
            prop = self.get_prop(mol)
            if self.min_value is not None:
                if self.allow_min and prop < self.min_value:
                    continue
                elif not self.allow_min and prop <= self.min_value:
                    continue
            if self.max_value is not None:
                if self.allow_max and prop > self.max_value:
                    continue
                elif not self.allow_max and prop >= self.max_value:
                    continue
            keep.append(mol)
        keep = np.asarray(keep)
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
