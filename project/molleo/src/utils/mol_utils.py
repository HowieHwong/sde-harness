"""Molecular utility functions"""

from typing import Optional, List
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')


def mol_to_smiles(mol: Chem.Mol) -> str:
    """Convert RDKit Mol to SMILES string"""
    return Chem.MolToSmiles(mol)


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert SMILES string to RDKit Mol"""
    return Chem.MolFromSmiles(smiles)


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES string is valid"""
    mol = smiles_to_mol(smiles)
    return mol is not None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES string"""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return mol_to_smiles(mol)


def get_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Get Morgan fingerprint for molecule"""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def calculate_properties(mol: Chem.Mol) -> dict:
    """Calculate basic molecular properties"""
    return {
        "molecular_weight": Descriptors.ExactMolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "num_h_acceptors": Descriptors.NumHAcceptors(mol),
        "num_h_donors": Descriptors.NumHDonors(mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
        "qed": Descriptors.qed(mol),
        "sa_score": calculate_sa_score(mol),
    }


def calculate_sa_score(mol: Chem.Mol) -> float:
    """Calculate synthetic accessibility score"""
    # Simplified implementation - in practice would use full SA score
    # from https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score
    return 1.0  # Placeholder