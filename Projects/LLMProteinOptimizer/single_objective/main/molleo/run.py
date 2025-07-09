from __future__ import print_function

import random
from typing import List

import progressbar
import json
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from tqdm import tqdm
from joblib import delayed
import progressbar
import main.molleo.crossover as co, main.molleo.mutate as mu
from main.optimizer import BaseOptimizer

from main.molleo.ea import EA
from main.molleo.llama3 import Llama3



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
    if len(population_mol) == 1:
        return list(zip(population_scores, population_mol))
    all_tuples = list(zip(population_scores, population_mol))
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
        mating_pool: list of Protein sequences
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

def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))
class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "molleo"
        self.save_score = []
        self.save_tuple = []
        self.mol_lm = None
        self.sample_worse_data_first =  False
        if args.mol_lm == "Llama3":
            self.mol_lm = Llama3()
            
        elif args.mol_lm == 'EA':
            self.mol_lm = EA()
        self.args = args
        self.lm_name = "baseline"
        if args.mol_lm != None:
            self.lm_name = args.mol_lm
            # self.mol_lm.task = self.args.oracles

    def _optimize(self, oracle, config):
        self.save_score = []
        self.save_tuple = []
        self.oracle.assign_evaluator(oracle)
        if self.args.dataset.__contains__('Syn'):
            starting_population = np.random.choice(self.all_protein, config["population_size"])
        else:
            starting_population = np.random.choice(self.all_protein[:int(0.6*len(self.all_protein))], config["population_size"])
        
        # starting_population = self.all_protein[:config["population_size"]]
        # select initial population
        population_smiles = starting_population
        population_mol = [s for s in population_smiles]
        population_scores = self.oracle([mol for mol in population_mol])
        population_tuples = list(zip(population_scores, population_mol))
        self.save_score.append([a[0] for a in population_tuples])
        self.save_tuple.append(population_tuples)
        self.log_intermediate()
        # print(population_scores)
        patience = 0
        count = 0
        save_score = []
        while True:
            if config['iteration'] != -1:
                if count >= config['iteration']:
                    self.log_intermediate(finish=True)
                    break
            
            if len(self.oracle) > 100:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
            else:
                old_score = 0

            # new_population
            mating_tuples = make_mating_pool(population_mol, population_scores, config["population_size"])
            rep = 0
            fp_scores = []
            offspring_mol_temp = []
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.mol_lm.to(device)
            if self.args.mol_lm == "GPT-4":
                offspring_mol = [self.mol_lm.edit(mating_tuples, config["mutation_rate"]) for _ in range(config["population_size"])]
            # elif self.args.mol_lm == "Llama3":

            #     offspring_mol = [self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.args.dataset) for _ in tqdm(range(config["population_size"]))]

            elif self.args.mol_lm == "Llama3":
                bar = progressbar.ProgressBar(maxval=config["population_size"])
                bar.start()
                offspring_mol = set()
                mean = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:config["population_size"]]])
                std = np.std([item[1][0] for item in list(self.mol_buffer.items())[:config["population_size"]]])
                # print(mean, std)
                while len(offspring_mol) < config["population_size"]:
                    new_seq, p1, p2 = self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.args.dataset, self.constrain_K, self.budget_K, mean, std)
                    if self.budget_K != -1:
                        if hamming_distance(p1, new_seq) > self.budget_K and hamming_distance(p2, new_seq) > self.budget_K:
                            continue
                    new_seq = self.filter([new_seq])

                    if len(new_seq) == 0:
                        continue
                    new_seq = new_seq[0]
                    if new_seq in offspring_mol:
                        rep += 1
                    if new_seq not in self.mol_buffer.keys():
                        
                        # print(' ', new_seq, p1, p2)
                        # print('hamming' , hamming_distance(self.wild_type, new_seq))
                        offspring_mol.add(new_seq)
                    bar.update(len(offspring_mol))
                
                offspring_mol = list(offspring_mol)
                bar.finish()
                # offspring_mol = [self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.args.dataset) for _ in tqdm(range(config["population_size"]))]
            elif self.args.mol_lm == "EA":
                # print('EA')
                dis = []
                bar = progressbar.ProgressBar(maxval=config["population_size"])
                offspring_mol = set()
                while len(offspring_mol) < config["population_size"]:
                    new_seq, p1, p2 = self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.args.dataset)
                    if self.budget_K != -1:
                        if hamming_distance(p1, new_seq) > self.budget_K and hamming_distance(p2, new_seq) > self.budget_K:
                            continue
                    new_seq = self.filter([new_seq])
                    if len(new_seq) == 0:
                        continue
                    new_seq = new_seq[0]
                    if new_seq not in self.mol_buffer.keys():
                        # print('hamming' , hamming_distance(self.wild_type, new_seq))
                        dis.append(hamming_distance(self.wild_type, new_seq))
                        offspring_mol.add(new_seq)
                    bar.update(len(offspring_mol))
                offspring_mol = list(offspring_mol)
                bar.finish()

                # print(max(dis))
                # if max(dis) > 5:
                #     exit()
                # offspring_mol = [self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.args.dataset, self.constrain_K) for _ in tqdm(range(config["population_size"]))]
            # add new_population
            population_mol += offspring_mol
            # print(len(population_mol))
            # population_mol = self.sanitize(population_mol)
            population_mol = list(set(population_mol))
            # print(len(population_mol))
            # exit()
            population_mol = self.filter(population_mol)
            
            # print(population_mol,len(population_mol))
            # exit()
            # stats
            old_scores = population_scores
            population_scores = self.oracle([mol for mol in population_mol])
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            self.save_score.append([a[0] for a in population_tuples])
            self.save_tuple.append(population_tuples)
            # self.log_distribution(population_tuples, count)
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]
            # print(len(self.oracle))
            
            ### early stopping
            if len(self.oracle) > 1:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
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


        


    # def log_distribution(self, population_tuples, count):
    #     fitness_scores = [a[0] for a in population_tuples]
    #     plt.figure(figsize=(8, 6))
    #     plt.hist(fitness_scores, bins=10, alpha=0.7, edgecolor='black')
    #     plt.title(f"Distribution of Fitness Scores (Iteration {count})")
    #     plt.xlabel("Fitness Score")
    #     plt.ylabel("Frequency")
    #     plt.grid(axis='y', linestyle='--', alpha=0.7)
    #     plt.savefig(f'/cluster/tufts/liulab/yiwan01/MOLLEO/single_objective/main/molleo/log/distribution_{self.args.dataset}_{count}.png')
        

