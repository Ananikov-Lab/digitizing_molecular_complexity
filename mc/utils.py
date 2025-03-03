import random
from typing import Generator

import matplotlib.pyplot as plt

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize.rdMolStandardize import Normalizer


def hide_frame(plt):
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)


def disable_rdkit_log():
    rdBase.DisableLog("rdApp.*")


def valid_mol(smiles: str):
    if "." in smiles:
        return False
    elts_to_keep = set(["C", "O", "N", "Cl", "Br", "F", "I", "P", "S", "Si", "B"])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    elts = set([atom.GetSymbol() for atom in mol.GetAtoms()])
    if not elts.issubset(elts_to_keep):
        return False

    normalizer = Normalizer()
    try:
        mol = Chem.AddHs(mol)
        mol = normalizer.normalize(mol)
        tpsa = Chem.Descriptors.TPSA(mol)
        return True
    except:
        return False


def weight_triples(precision=0.1) -> Generator:
    """
    generator that returns the tuples in the form (p1, p2, p3),
    where p1 + p2 + p3 = 1 and p1, p2, p3 are non-negative weights for each phase

    Parameters:
    -----------
    param: precision: precision of the weights (0.1 by default)

    Yields:
    -----------
    tuple, (p1, p2, p3) of three non-negative weights for each phase
    """
    n = int(1 / precision)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            yield (i * precision, j * precision, k * precision)


# Helper dictionary for functional groups
FUNCTIONAL_GROUPS = {
    "phenyl": "c1ccccc1",
    "methyl": "C",
    "amine": "N",
    "hydroxyl": "O",
    "carboxyl": "C(=O)O",
    "aldehyde": "C=O",
    "t-Bu": "C(C)(C)C",
    "isopropyl": "C(C)C",
    "nitril": "C#N",
    "sulfo": "S(=O)(=O)O",
    "fluorine": "F",
    "chlorine": "Cl",
    "bromine": "Br",
    "iodine": "I",
    "methoxy": "OC",
}


def replace_hydrogen_with_group(smiles, group_name):
    if group_name not in FUNCTIONAL_GROUPS:
        raise ValueError("Invalid functional group name")

    mol = Chem.MolFromSmiles(smiles)
    mol_with_h = Chem.AddHs(mol)

    hydrogens = [
        atom.GetIdx() for atom in mol_with_h.GetAtoms() if atom.GetAtomicNum() == 1
    ]
    if not hydrogens:
        raise ValueError("No hydrogen atoms found in the molecule")

    hydrogen_to_replace = random.choice(hydrogens)
    parent_atom_idx = next(
        bond.GetOtherAtomIdx(hydrogen_to_replace)
        for bond in mol_with_h.GetBonds()
        if bond.GetBeginAtomIdx() == hydrogen_to_replace
        or bond.GetEndAtomIdx() == hydrogen_to_replace
    )

    group_mol = Chem.MolFromSmiles(FUNCTIONAL_GROUPS[group_name])

    emol = Chem.EditableMol(mol_with_h)

    emol.RemoveAtom(hydrogen_to_replace)

    atom_mapping = {}
    for atom in group_mol.GetAtoms():
        new_idx = emol.AddAtom(atom)
        atom_mapping[atom.GetIdx()] = new_idx

    for bond in group_mol.GetBonds():
        a1 = atom_mapping[bond.GetBeginAtomIdx()]
        a2 = atom_mapping[bond.GetEndAtomIdx()]
        emol.AddBond(a1, a2, order=bond.GetBondType())

    emol.AddBond(parent_atom_idx, atom_mapping[0], order=Chem.rdchem.BondType.SINGLE)

    final_mol = emol.GetMol()

    Chem.SanitizeMol(final_mol)

    return Chem.MolToSmiles(final_mol, isomericSmiles=True)


def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)
