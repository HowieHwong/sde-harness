"""Evolutionary operations for MolLEO"""

import random
from typing import List, Tuple, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import os

# Import crossover and mutation operations from ga module
from src.ga import crossover as co
from src.ga import mutations as mu

MINIMUM = 1e-10


def crossover(mol1: Chem.Mol, mol2: Chem.Mol) -> Optional[Chem.Mol]:
    """Perform crossover between two molecules"""
    return co.crossover(mol1, mol2)


def mutate(mol: Chem.Mol, mutation_rate: float, mol_lm=None) -> Optional[Chem.Mol]:
    """Perform mutation on a molecule"""
    # If mol_lm is provided and has a mutate method (MolLEOOptimizer), use it
    if mol_lm is not None and hasattr(mol_lm, 'mutate'):
        return mol_lm.mutate(mol)
    else:
        # Use original mutation without LLM
        return mu.mutate(mol, mutation_rate, None)


def make_mating_pool(population_mol: List[Chem.Mol], 
                     population_scores: List[float], 
                     offspring_size: int) -> List[Tuple[float, Chem.Mol]]:
    """
    Create mating pool for reproduction
    
    Args:
        population_mol: list of RDKit Mol objects
        population_scores: list of scores for each molecule
        offspring_size: number of molecules to return
        
    Returns:
        List of (score, mol) tuples
    """
    # Combine scores and molecules
    all_tuples = list(zip(population_scores, population_mol))
    
    # Handle negative scores by shifting to make all scores positive
    min_score = min(population_scores)
    if min_score < 0:
        # Shift all scores to make them non-negative
        shifted_scores = [s - min_score + MINIMUM for s in population_scores]
    else:
        # Add minimum to ensure all scores are positive
        shifted_scores = [s + MINIMUM for s in population_scores]
    
    # Normalize scores to probabilities
    sum_scores = sum(shifted_scores)
    if sum_scores <= 0:
        # Fallback: equal probabilities if all scores are zero or negative
        population_probs = [1.0 / len(shifted_scores)] * len(shifted_scores)
    else:
        population_probs = [p / sum_scores for p in shifted_scores]
    
    # Ensure all probabilities are non-negative and sum to 1
    population_probs = [max(0.0, p) for p in population_probs]
    total_prob = sum(population_probs)
    if total_prob > 0:
        population_probs = [p / total_prob for p in population_probs]
    else:
        # Fallback: equal probabilities
        population_probs = [1.0 / len(population_probs)] * len(population_probs)
    
    # Sample with replacement
    mating_indices = np.random.choice(
        len(all_tuples), 
        p=population_probs, 
        size=offspring_size, 
        replace=True
    )
    
    mating_tuples = [all_tuples[idx] for idx in mating_indices]
    
    return mating_tuples


def reproduce(mating_tuples: List[Tuple[float, Chem.Mol]], 
              mutation_rate: float, 
              mol_lm=None,
              net=None) -> Tuple[Optional[Chem.Mol], Optional[Chem.Mol]]:
    """
    Reproduce new molecules from mating pool
    
    Args:
        mating_tuples: List of (score, mol) tuples
        mutation_rate: Probability of mutation
        mol_lm: Language model for guided mutation
        net: Neural network model (optional)
        
    Returns:
        Tuple of (crossover child, mutated child)
    """
    # Select two parents randomly
    parent1 = random.choice(mating_tuples)
    parent2 = random.choice(mating_tuples)
    
    parent_mols = [parent1[1], parent2[1]]
    
    # Debug: Print parent SMILES
    parent1_smiles = Chem.MolToSmiles(parent1[1])
    parent2_smiles = Chem.MolToSmiles(parent2[1])
    print(f"DEBUG: Parents - {parent1_smiles} x {parent2_smiles}")
    
    # Perform crossover
    new_child = crossover(parent_mols[0], parent_mols[1])
    
    # Debug: Print crossover result
    if new_child is not None:
        child_smiles = Chem.MolToSmiles(new_child)
        print(f"DEBUG: Crossover successful - {child_smiles}")
    else:
        print(f"DEBUG: Crossover failed")
    
    # Perform mutation if crossover succeeded
    new_child_mutation = None
    if new_child is not None:
        new_child_mutation = mutate(new_child, mutation_rate, mol_lm)
        
        # Debug: Print mutation result
        if new_child_mutation is not None:
            mutation_smiles = Chem.MolToSmiles(new_child_mutation)
            print(f"DEBUG: Mutation successful - {mutation_smiles}")
        else:
            print(f"DEBUG: Mutation failed")
        
    return new_child, new_child_mutation


def get_best_mol(population_scores: List[float], 
                 population_mol: List[Chem.Mol]) -> str:
    """Get SMILES of best molecule in population"""
    best_idx = np.argmax(population_scores)
    best_mol = population_mol[best_idx]
    return Chem.MolToSmiles(best_mol)