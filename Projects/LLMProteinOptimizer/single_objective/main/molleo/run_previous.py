from __future__ import print_function

import random
from typing import List
import torch
import json
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from tqdm import tqdm
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

import main.molleo.crossover as co, main.molleo.mutate as mu
from main.optimizer import BaseOptimizer

from main.molleo.mol_lm import MolCLIP
from main.molleo.biot5 import BioT5
from main.molleo.ea import EA
from Molleo_Protein.single_objective.main.molleo.llama3_previous import Llama3

from main.molleo.GPT4 import GPT4
from .utils import get_fp_scores
from .network import create_and_train_network, obtain_model_pred


MINIMUM = 1e-10

def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs
    all_tuples = list(zip(population_scores, population_mol))
    min_s = min(population_scores)
    max_s = max(population_scores)
    population_scores = [(s-min_s) / (max_s-min_s)for s in population_scores]
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

def get_best_mol(population_scores, population_mol):
    top_mol = population_mol[np.argmax(population_scores)]
    top_smi = Chem.MolToSmiles(top_mol)
    return top_smi

class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "molleo"
        self.save_score = []
        self.save_tuple = []
        self.mol_lm = None
        self.sample_worse_data_first =  False
        if args.mol_lm == "GPT-4":
            self.mol_lm = GPT4()
        elif args.mol_lm == "BioT5":
            self.mol_lm = BioT5()
        elif args.mol_lm == "Llama3":
            self.mol_lm = Llama3()
            
        elif args.mol_lm == 'EA':
            self.mol_lm = EA()
        self.args = args
        lm_name = "baseline"
        if args.mol_lm != None:
            lm_name = args.mol_lm
            # self.mol_lm.task = self.args.oracles

    def _optimize(self, oracle, config):
        self.save_score = []
        self.save_tuple = []
        self.oracle.assign_evaluator(oracle)

        pool = joblib.Parallel(n_jobs=self.n_jobs)
        
        # if self.smi_file is not None:
            # Exploitation run
            # starting_population = self.all_smiles[:config["population_size"]]
        # else:
        #     # Exploration run
        
        if self.sample_worse_data_first:
            #lower score protein will be sample with lower prob
            weights = np.linspace(1, 0.01, len(self.all_protein))  
            starting_population = np.random.choice(self.all_protein[:], size=config["population_size"], replace=False, p=weights/weights.sum())
            # starting_population = np.random.choice(self.all_protein[:int(0.5*len(self.all_protein))], config["population_size"])
        else:
            starting_population = np.random.choice(self.all_protein[:], config["population_size"])
        
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
            
            fp_scores = []
            offspring_mol_temp = []
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.mol_lm.to(device)
            if self.args.mol_lm == "GPT-4":
                offspring_mol = [self.mol_lm.edit(mating_tuples, config["mutation_rate"]) for _ in range(config["population_size"])]
            elif self.args.mol_lm == "Llama3":

                offspring_mol = [self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.args.dataset) for _ in tqdm(range(config["population_size"]))]
            elif self.args.mol_lm == "EA":
                # print('EA')
                offspring_mol = [self.mol_lm.edit(mating_tuples, config["mutation_rate"], self.args.dataset) for _ in tqdm(range(config["population_size"]))]
            # add new_population

            population_mol += offspring_mol
            # print(population_mol,len(population_mol))
            # population_mol = self.sanitize(population_mol)
            # print(len(population_mol))
            population_mol = list(set(population_mol))
            # print(len(population_mol))
            # exit()
            population_mol = self.filter(population_mol)
            # print(population_mol,len(population_mol))
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
            if len(self.oracle) > 100:
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


        


    def log_distribution(self, population_tuples, count):
        fitness_scores = [a[0] for a in population_tuples]
        plt.figure(figsize=(8, 6))
        plt.hist(fitness_scores, bins=10, alpha=0.7, edgecolor='black')
        plt.title(f"Distribution of Fitness Scores (Iteration {count})")
        plt.xlabel("Fitness Score")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'/home/jhe/Molleo_Protein/single_objective/main/molleo/log/{self.mol_lm}_distribution_{self.args.dataset}_{count}.png')
        

