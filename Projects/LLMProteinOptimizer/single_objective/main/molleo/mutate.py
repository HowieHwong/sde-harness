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



# 100/10000 | avg_top1: 2.600 | avg_top10: 0.855 | avg_top100: 0.090 | 
# {'avg_top1': 2.600067722, 'avg_top10': 0.8552329219999999, 'avg_top100': 0.08983971902999999, 'auc_top1': 0.01300033861, 'auc_top10': 0.004276164609999999, 'auc_top100': 0.0004491985951499999, 'n_oracle': 100}

# (500-population/offspring)