"""
Tests for descriptors.
"""
from rdkit import Chem

from muv.descriptors import Descriptors


def test_count_ring_systems():
    d = Descriptors()
    hexane = Chem.MolFromSmiles("CCCCCC")
    assert d.count_ring_systems(hexane) == 0
    benzene = Chem.MolFromSmiles("c1ccccc1")
    assert d.count_ring_systems(benzene) == 1
    naphthalene = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    assert d.count_ring_systems(naphthalene) == 1
    biphenyl = Chem.MolFromSmiles("c1ccc(cc1)c2ccccc2")
    assert d.count_ring_systems(biphenyl) == 2
