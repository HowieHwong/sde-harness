from __future__ import print_function

import random
from typing import List

import joblib
import numpy as np
from joblib import delayed
import progressbar
from tqdm import tqdm

import main.molleo_multi_pareto.crossover as co, main.molleo_multi_pareto.mutate as mu
from main.pareto_optimizer import BaseOptimizer

from main.molleo_multi_pareto.llama3 import Llama3
from main.molleo_multi_pareto.ea import EA
MINIMUM = 1e-10

def make_mating_pool(population_mol: List, population_scores, offspring_size: int):
    """
    Given a population of Protein sequences and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of Protein sequences
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of Protein sequences (probably not unique)
    """
    # scores -> probs

    all_tuples = list(zip(population_scores, population_mol))
    if len(population_mol) == 1:
        return [all_tuples[0] for _ in range(offspring_size)]
    min_s = min(population_scores)
    max_s = max(population_scores)
    population_scores = [(s-min_s) / (max_s-min_s) for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_indices = np.random.choice(len(all_tuples), size=offspring_size, replace=True, p=population_probs) #
    
    mating_tuples = [all_tuples[indice] for indice in mating_indices]
    
    return mating_tuples

def reproduce(mating_tuples, mutation_rate, mol_lm=None, net=None):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent = []
    parent.append(random.choice(mating_tuples))
    parent.append(random.choice(mating_tuples))

    parent_mol = [t[1] for t in parent]
    new_child = co.crossover(parent_mol[0], parent_mol[1])
    new_child_mutation = None
    if new_child is not None:
        new_child_mutation = mu.mutate(new_child, mutation_rate, mol_lm)
    return new_child, new_child_mutation


class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "molleo"
        self.save_score = []
        self.save_tuple = []
        self.mol_lm = None
        if args.mol_lm == "GPT-4":
            self.mol_lm = GPT4()
        elif args.mol_lm == "BioT5":
            self.mol_lm = BioT5()
        elif args.mol_lm == "Llama3":
            self.mol_lm = Llama3()
        elif args.mol_lm == "EA":
            self.mol_lm = EA()
        self.args = args
        lm_name = "baseline"
        if args.mol_lm != None:
            self.lm_name = args.mol_lm

    def _optimize(self, config):
        self.config = config
        self.save_score = []
        self.save_tuple = []
        self.oracle.assign_evaluator(self.args)

        pool = joblib.Parallel(n_jobs=self.n_jobs)
        
        # starting_population = np.random.choice(self.all_protein[:int(0.6*len(self.all_protein))], config["population_size"])
        starting_population = np.random.choice(self.all_protein, config["population_size"])
        # select initial population
        population_smiles = starting_population
        population_mol = [s for s in population_smiles]
        # print(population_mol)
        population_scores = self.oracle([mol for mol in population_mol])

        population_tuples = list(zip(population_scores, population_mol))
        self.save_score.append([a[0] for a in population_tuples])
        self.save_tuple.append(population_tuples)
        self.log_intermediate()
        patience = 0
        count =0   
        while True:
            if config['iteration'] != -1:
                if count >= config['iteration']:
                    self.log_intermediate(finish=True)
                    break
            if len(self.oracle) > 1:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())])
            else:
                old_score = 0

            # new_population
            mating_tuples = make_mating_pool(population_mol, population_scores, config["population_size"])

            
            fp_scores = []
            offspring_mol_temp = []
            if self.args.mol_lm == 'GPT-4':
                offspring_mol = [self.mol_lm.edit(mating_tuples, config["mutation_rate"]) for _ in range(config["offspring_size"])]
            elif self.args.mol_lm == "Llama3":
                offspring_mol = [self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.args.dataset) for _ in tqdm(range(config["population_size"]))]
                # mean = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:config["population_size"]]])
                # std = np.std([item[1][0] for item in list(self.mol_buffer.items())[:config["population_size"]]])
                # bar = progressbar.ProgressBar(maxval=config["population_size"])
                # offspring_mol = set()
                # while len(offspring_mol) < config["population_size"]:
                #     new_seq = self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.args.dataset,mean=mean,std=std)
                #     new_seq = self.filter([new_seq])
                #     if len(new_seq) == 0:
                #         continue
                #     new_seq = new_seq[0]
                #     # if new_seq not in self.mol_buffer.keys():
                #     #     offspring_mol.add(new_seq)
                #     offspring_mol.add(new_seq)
                #     bar.update(len(offspring_mol))
                # offspring_mol = list(offspring_mol)
                # bar.finish()

            elif self.args.mol_lm == "EA":
                offspring_mol = [self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.args.dataset) for _ in tqdm(range(config["population_size"]))]
                # bar = progressbar.ProgressBar(maxval=config["population_size"])
                # offspring_mol = set()
                # while len(offspring_mol) < config["population_size"]:
                #     new_seq = self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.args.dataset)
                #     new_seq = self.filter([new_seq])
                #     if len(new_seq) == 0:
                #         continue
                #     new_seq = new_seq[0]
                #     # if new_seq not in population_mol:
                #     #     offspring_mol.add(new_seq)
                #     offspring_mol.add(new_seq)
                #     bar.update(len(offspring_mol))
                # offspring_mol = list(offspring_mol)
                # bar.finish()
            # add new_population

            population_mol += offspring_mol
            population_mol = list(set(population_mol))

            #Pareto optimal set
            self.oracle.clean_buffer()
            population_mol = self.filter(population_mol)
            population_mol,population_mol_r = self.oracle.select_pareto_front(population_mol)
            # population_mol= np.append(population_mol, self.oracle.select_replaxed_pareto_front(population_mol, population_mol_r, num_closest=1))
            # stats
            # print(population_mol)
            old_scores = population_scores
            population_mol = [m for m in population_mol]
            population_scores = self.oracle(population_mol)
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)
            self.save_score.append([a[0] for a in population_tuples])
            self.save_tuple.append(population_tuples)
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]


            ### early stopping
            if len(self.oracle) > 1:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())])
                # import ipdb; ipdb.set_trace()
                if (new_score - old_score) < 1e-3:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

                old_score = new_score
            count += 1    
            if self.finish:
                break

