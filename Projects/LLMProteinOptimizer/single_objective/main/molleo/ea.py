
import crossover as co
import mutate as mu
import re
import random
MINIMUM = 1e-10


class EA:
    def __init__(self):
        pass
    def edit(self, mating_tuples, mutation_rate, dataset='GB1'):
        parent = []
        parent.append(random.choice(mating_tuples))
        parent.append(random.choice(mating_tuples))
        parent_mol = [t[1] for t in parent]
        parent_scores = [t[0] for t in parent]
        new_child = co.crossover_seq(parent_mol[0], parent_mol[1])
        if new_child is not None:
            new_child = mu.mutate_seq(new_child, mutation_rate)
        return new_child, parent_mol[0], parent_mol[1]
    
if __name__ == "__main__":
    model = EA()
    # model=model.to('cuda')
    print(model.edit([[0.001791394,"KWNA"],[0.007212719,"ARAF"],[2.31199298,"MRFG"],[0.022350843,'LDVA']],0.0))

