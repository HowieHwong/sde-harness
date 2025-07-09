import numpy as np

import selfies as sf
import yaml
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

def get_selfies_chars(selfies):
    """Obtain a list of all selfie characters in string selfies
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfies_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns
    -------
    chars_selfies (list of strings) : 
        list of selfie characters present in molecule selfie
    """
    chars_selfies = sf.split_selfies(selfies)
    return list(chars_selfies)

def sanitize_smiles(smi):
    """
    Return a canonical smile representation of smi 

    Parameters
    ----------
    smi : str
        smile string to be canonicalized 

    Returns
    -------
    mol (rdkit.Chem.rdchem.Mol) : 
        RdKit mol object (None if invalid smile string smi)
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): 
        True/False to indicate if conversion was  successful 
    """
    if smi == '':
        return None
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return smi_canon
    except:
        return None

def get_fp_scores(smiles_back, target_smi):
    """
    Given a list of SMILES (smiles_back), tanimoto similarities are calculated 
    (using Morgan fingerprints) to SMILES (target_smi). 
    Parameters
    ----------
    smiles_back : (list)
        List of valid SMILE strings. 
    target_smi : (str)
        Valid SMILES string. 
    Returns
    -------
    smiles_back_scores : (list of floats)
        List of fingerprint similarity scores of each smiles in input list. 
    """
    smiles_back_scores = []
    target = smi2mol(target_smi)
    fp_target = AllChem.GetMorganFingerprint(target, 2)
    for item in smiles_back:
        mol = smi2mol(item)
        fp_mol = AllChem.GetMorganFingerprint(mol, 2)
        score = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores

def from_yaml(work_dir, 
        fitness_function, 
        start_population,
        yaml_file, **kwargs):

    # create dictionary with parameters defined by yaml file 
    with open(yaml_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    params.update(kwargs)
    params.update({
        'work_dir': work_dir,
        'fitness_function': fitness_function,
        'start_population': start_population
    })

    return params
if __name__ == "__main__":
    data= [["CN1N=NC2=C1C=C([N+](=O)[C@H](O)C1=C(F)C(C3=CCOC(C(=S)N4N=NC5=CC=C([N+](=O)[O-])C=C54)=C3)C=C1)C=C2", "C=C1C2=C(C(=O)N3NC=C(C)N3)C=CC(=C(C3=CC(NC)NC3[N+](=O)[O-])NN1C)C2(F)F"],
            ["C[NH+]1CCCC[C@@H]1C1=NC(CN)=NN1C=CC=CN1N=NC2=CC=C([N+](=O)[O-])C=C21", "CC1(C)CCCC[C@@H]1C1=Nc2nn3nccc([N+](=O)[O-])cc-3ccn2NC(CN)=N1"],
            ["O=C(NC1=CCCC1)NC1=CC=C(c2ccc(N3CC=CC3)c(N3CC=CC3)c2)C1", "C1=CN(C2=CC=C(C3=CC=NC(NC4=CCCC4)=C3)N=C2)C=CC1"],
            ["N[N+](=O)C1=C(NC2=CC=CC(O[O-])=C2)N=CCC1", "N#Cc1ccnc(Nc2ccc(NC3=CCC(N)=C3N)cc2N)n1"],
            ["FC1=CC=C(NC2=CC=CCN2)C1", "NC1C=C(F)CC(NC2=CC=CCN2)=C1"],
            ["C=Cc1ccc([N+](=O)[O-])cc1", "C=Cc1ccc(CNc2cc([N+](=O)[O-])ccc2C=C)cc1"],
            ["C1=C(Nc2ccc(-c3ccnc(NC4=CCCC4)c3)cn2)CCC1", "C1=CN(c2ccc(-c3ccnc(NC4=CCCC4)c3)nc2)C=CC1"],
            ["Cc1cnc(Nc2ccnc(Nc3cccc(F)c3)c2)cc1C", "Cc1cnc(Nc2cccc(F)c2)cc1C"],
            ["O=C=CC(C(=O)N1N=NC2=CC=C([N+](=O)[O-])C=C21)[N+](=O)C#CC(=O)[O-]", "CN(C)C(=O)C#CC(=O)C(=O)N1C2=CN=CCC2=NN1C(=O)C(C)(C)C#CC(N)=O"],
            ["C=C(O)ONC(=O)N1N=NC2=CC=C([N+](=O)[O-])C=C21", "C=C(C)ONC(=O)N1NC2=CC=CC2=NN1C(=O)NNC(=O)O"]]
    for elem in data:
        print(get_fp_scores([elem[0]], elem[1]))
