import random

import numpy as np

def mutate_seq(sequence, mutation_rate=0.1, valid_chars=None):
    """Perform mutation on a sequence."""
    if valid_chars is None:
        valid_chars = 'ACDEFGHIKLMNPQRSTVWY'  
    # print(mutation_rate)
    if random.random() < mutation_rate:
        return sequence
    sequence_list = list(sequence)
    char = random.choice(valid_chars)
    index = random.choice(range(len(sequence_list)))
    # for i in range(len(sequence_list)):
    #     sequence_list[i] = random.choice(valid_chars)
    sequence_list[index] = char
    return ''.join(sequence_list)