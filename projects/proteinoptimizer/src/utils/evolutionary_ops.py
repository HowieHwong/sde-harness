"""Evolutionary operations for protein sequences."""

from __future__ import annotations

import random
from typing import List, Tuple
import numpy as np

# --- Re-implementations for sequences ---

def crossover_seq(seq1: str, seq2: str) -> Tuple[str, str]:
    """Uniform crossover for two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must have the same length for crossover.")
    
    child1, child2 = list(seq1), list(seq2)
    for i in range(len(child1)):
        if random.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]
            
    return "".join(child1), "".join(child2)

def mutate_seq(seq: str, mutation_rate: float) -> str:
    """Random point mutations."""
    from ..core.protein_optimizer import AMINO_ACIDS
    
    mutated_seq = list(seq)
    for i in range(len(mutated_seq)):
        if random.random() < mutation_rate:
            mutated_seq[i] = random.choice(AMINO_ACIDS)
            
    return "".join(mutated_seq)

# --- Kept for potential compatibility, but adapted ---

def make_mating_pool(population: List[str], 
                     scores: List[float], 
                     offspring_size: int) -> List[Tuple[float, str]]:
    """Create a mating pool using fitness proportionate selection."""
    
    # Shift scores to be non-negative for probability calculation
    min_score = min(scores)
    shifted_scores = [s - min_score + 1e-9 for s in scores]
    total_fitness = sum(shifted_scores)
    
    if total_fitness == 0:
        # All individuals have the same fitness, equal probability
        probs = [1.0 / len(population)] * len(population)
    else:
        probs = [s / total_fitness for s in shifted_scores]
    
    # Sample individuals with replacement
    mating_indices = np.random.choice(
        len(population),
        size=offspring_size,
        replace=True,
        p=probs
    )
    
    mating_pool = [(scores[i], population[i]) for i in mating_indices]
    return mating_pool

def reproduce(mating_pool: List[Tuple[float, str]], 
              mutation_rate: float) -> Tuple[str, str]:
    """Create two new sequences via crossover and mutation."""
    
    # Select two parents from the mating pool
    parent1_idx, parent2_idx = np.random.choice(len(mating_pool), 2, replace=False)
    parent1 = mating_pool[parent1_idx][1]
    parent2 = mating_pool[parent2_idx][1]
    
    # Perform crossover
    child1, child2 = crossover_seq(parent1, parent2)
    
    # Perform mutation
    mutated_child1 = mutate_seq(child1, mutation_rate)
    mutated_child2 = mutate_seq(child2, mutation_rate)
    
    return mutated_child1, mutated_child2