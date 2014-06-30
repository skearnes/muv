"""
MUV simple descriptors:
* Atom counts: all, heavy, B, Br, C, Cl, F, I, N, O, P, S
* Number of hydrogen bond acceptors / donors
* cLogP
* Number of chiral centers
* Number of ring systems

Descriptors are calculated with the RDKit.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChiralType


class Descriptors(object):
    def __call__(self, mol):
        return self.get_descriptors(mol)

    def get_descriptors(self, mol):
        d = []

        # prep
        mol = Chem.AddHs(mol)

        # atom counts
        atoms = {'B': 5, 'Br': 35, 'C': 6, 'Cl': 17, 'F': 9, 'I': 53, 'N': 7,
                 'O': 8, 'P': 15, 'S': 16}
        counts = self.atom_counts(mol)
        total = mol.GetNumAtoms()
        d.append(total)
        heavy = mol.GetNumHeavyAtoms()
        d.append(heavy)
        for name in sorted(atoms.keys()):
            if atoms[name] in counts:
                d.append(counts[atoms[name]])
            else:
                d.append(0)

        # hydrogen bond acceptors / donors
        n_acc = AllChem.CalcNumHBA(mol)
        d.append(n_acc)
        n_don = AllChem.CalcNumHBD(mol)
        d.append(n_don)

        # cLogP
        c_log_p, _ = AllChem.CalcCrippenDescriptors(mol)
        d.append(c_log_p)

        # number of chiral centers
        n_chiral = 0
        for atom in mol.GetAtoms():
            if (atom.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CW or
                    atom.GetChiralTag() == ChiralType.CHI_TETRAHEDRAL_CCW):
                n_chiral += 1
        d.append(n_chiral)

        # number of ring systems (not the number of rings)
        n_ring_systems = self.count_ring_systems(mol)
        d.append(n_ring_systems)

    @staticmethod
    def atom_counts(mol):
        """
        Get the number of atoms for each atomic number.

        Parameters
        ----------
        mol : Mol
            Molecule.
        """
        counts = {}
        for atom in mol.GetAtoms():
            z = atom.GetAtomicNum()
            if z not in counts:
                counts[z] = 0
            counts[z] += 1
        return counts

    @staticmethod
    def count_ring_systems(mol):
        """
        Get the number of ring systems (not rings) for a molecule.

        Parameters
        ----------
        mol : Mol
            Molecule.
        """
        n_ring_systems = 0
        ring_atoms = set()
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            this = set(ring)
            if not len(ring_atoms.intersection(this)):
                n_ring_systems += 1
            ring_atoms.update(this)
        return n_ring_systems
