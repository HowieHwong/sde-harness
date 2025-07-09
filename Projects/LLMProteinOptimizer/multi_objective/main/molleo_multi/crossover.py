import random

import numpy as np


def crossover_seq(seq1, seq2):
    """Perform single-point crossover between two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length for crossover.")
    
    # Randomly choose a crossover point
    point = random.randint(1, len(seq1) - 1)
    
    # Perform crossover
    new_seq1 = seq1[:point] + seq2[point:]
    # new_seq2 = seq2[:point] + seq1[point:]
    # new_seq1 = seq1
    # new_seq1 = seq1[:point] + seq2[point] + seq1[point+1:]
    return new_seq1