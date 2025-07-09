import os
import yaml
import random
import torch
import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
ALPHABET = "ARNDCQEGHILKMFPSTWYV-"
A2N = {a: n for n, a in enumerate(ALPHABET)}
A2N["X"] = 20
import potts_model
class Oracle_fitness():
    def __init__(self, dataset):
        self.name=dataset
        df = pd.read_csv(f'../data/{self.name}/fitness.csv')
        self.fitness_dict = df.set_index('Combo').to_dict()['fitness']
        if self.name.__contains__('Syn'):
            self.landscape = potts_model.load_from_mogwai_npz(f'data/{self.name[-4:]}_1_A_model_state_dict.npz', coupling_scale=1.0)
    def __call__(self, *args, **kwargs):
        sequencns_lst = args[0]
        if self.name.__contains__('Syn'):
            sequencns_lst = np.array([A2N[i] for i in sequencns_lst])
            if type(sequencns_lst) != list:
                return self.landscape.evaluate(sequencns_lst)[0]
            return self.landscape.evaluate(sequencns_lst)
        
        if type(sequencns_lst) == list:
            return [self.fitness_dict[s] for s in sequencns_lst]
        return self.fitness_dict[sequencns_lst]
    
class Oracle_hamming():
    def __init__(self, dataset):
        if dataset == 'TrpB':
            self.wild_type='VFVS'
        elif dataset == 'GB1':
            self.wild_type='VDGV'
        else:
            self.wild_type='SKLQICVEPTSQKLMPGSTLVLQCVAVGSPIPHYQWFKNELPLTHETKKLYMVPYVDLEHQGTYWCHVYNDRDSQDSKKVEIIID'
    def hamming_distance(self, chaine1, chaine2):
        return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))
        
    def __call__(self, sequencns_lst):
        if type(sequencns_lst) == list:
            return [self.hamming_distance(self.wild_type, sequence)/len(self.wild_type) for sequence in sequencns_lst]
        return self.hamming_distance(self.wild_type, sequencns_lst)/len(self.wild_type)
       
    
class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


class Oracle:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.max_obj = args.max_obj
        self.min_obj = args.min_obj
        self.max_evaluator = None
        self.min_evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.mol_buffer = mol_buffer
        self.storing_buffer = {}
        self.last_log = 0

    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, args):

        self.max_evaluator = []
        self.min_evaluator = []
        eva = Oracle_fitness(args.dataset)
        self.max_evaluator.append(eva)
        
        eva = Oracle_hamming(args.dataset)
        self.min_evaluator.append(eva)

    def evaluate(self, smi):
        score = 0
        for eva in self.max_evaluator:
            score = score + eva(smi)
        for eva in self.min_evaluator:
            score = score * 0.7 + (1 - eva(smi))*0.3
        return score

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def clean_buffer(self):
        self.storing_buffer = self.storing_buffer | self.mol_buffer
        self.mol_buffer = {}

    def save_result(self, suffix=None):
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.storing_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.storing_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                n_calls = len(self.mol_buffer)
        
        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        
        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f}')

        # try:
        print({
            "avg_top1": avg_top1, 
            "avg_top10": avg_top10, 
            "avg_top100": avg_top100, 
            "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
            "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
            "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
            "n_oracle": n_calls,
        })



    def __len__(self):
        return len(self.mol_buffer) 

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if smi is None:
            return 0
        if len(smi) == 0:
            return 0
        else:
            if smi in self.mol_buffer:
                pass
            elif len(self.mol_buffer) > self.max_oracle_calls:
                return -np.inf
            else:
                self.mol_buffer[smi] = [float(self.evaluate(smi)), len(self.mol_buffer)+1]
            return self.mol_buffer[smi][0]
    
    def __call__(self, smiles_lst):
        """
        Score
        """
        if type(smiles_lst) == list:
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                
            self.sort_buffer()
            self.log_intermediate()
            self.last_log = len(self.mol_buffer)
            self.save_result(self.task_label)
        else:  ### a string of SMILES 
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        return score_list

    def select_pareto_front(self, smiles_lst):
        if type(smiles_lst) == list:
            score_list = []
            for smi in smiles_lst:
                single_score = []
                for eva in self.max_evaluator:
                    single_score.append(1 - eva(smi))
                for eva in self.min_evaluator:
                    single_score.append(eva(smi))
                score_list.append(single_score)
            # print(score_list)
            # exit(0)
            score_array = np.array(score_list)
            nds = NonDominatedSorting().do(score_array, only_non_dominated_front=True)
            pareto_front = np.array(smiles_lst)[nds]
            remaining = [smi for i, smi in enumerate(smiles_lst) if i not in nds]

            return pareto_front,remaining
        else:
            print('Smiles should be in the list format.')
            
    def select_replaxed_pareto_front(self, points, pareto_points, num_closest=1):
        pareto_scores = []
        for smi in pareto_points:
            single_score = []
            for eva in self.max_evaluator:
                single_score.append(1 - eva(smi))
            for eva in self.min_evaluator:
                single_score.append(eva(smi))
            pareto_scores.append(single_score)
        pareto_scores = np.array(pareto_scores)

        # Calculate scores for other points
        other_scores = []
        for smi in points:
            single_score = []
            for eva in self.max_evaluator:
                single_score.append(1 - eva(smi))
            for eva in self.min_evaluator:
                single_score.append(eva(smi))
            other_scores.append(single_score)
        other_scores = np.array(other_scores)
        
        distances = np.linalg.norm(pareto_scores[:, np.newaxis] - other_scores, axis=2)
        closest_indices = np.argsort(distances, axis=1)[:, :num_closest]

        closest_points = []
        for indices in closest_indices:
            closest_points.extend(np.array(points)[indices])

        return np.unique(closest_points)

    @property
    def finish(self):
        return len(self.storing_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None):
        self.model_name = args.mol_lm
        self.args = args
        
        self.n_jobs = args.n_jobs
        # self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        # self.smi_file = args.smi_file
        self.oracle = Oracle(args=self.args)
        
        self.all_protein = pd.read_csv(f'../single_objective/ALDE/data/{args.dataset}/fitness.csv')
        self.all_protein = self.all_protein['Combo'].tolist()
        
        #.sort_values(by='fitness')
        self.seq_len = len(self.all_protein[0])

    def filter(self, proteins):
        # print('here',self.seq_len)
        result = []
        # print(self.all_protein)
        
        for i in proteins:
            
            if len(i) == self.seq_len:
                if not self.args.dataset.__contains__('Syn'):
                    if i not in self.all_protein:
                        continue
                
                result.append(i)
        return result
        
    def sort_buffer(self):
        self.oracle.sort_buffer()
    
    def log_intermediate(self, mols=None, scores=None, finish=False):
        self.oracle.log_intermediate(mols=mols, scores=scores, finish=finish)
    
    def log_result(self):

        print(f"Logging final results...")

        # import ipdb; ipdb.set_trace()

        log_num_oracles = [100, 500, 1000, 3000, 5000, 10000]
        assert len(self.mol_buffer) > 0 

        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
        if len(results) > 10000:
            results = results[:10000]
        
        results_all_level = []
        for n_o in log_num_oracles:
            results_all_level.append(sorted(results[:n_o], key=lambda kv: kv[1][0], reverse=True))
        

        # Log batch metrics at various oracle calls
        data = [[log_num_oracles[i]] + self._analyze_results(r) for i, r in enumerate(results_all_level)]
        columns = ["#Oracle", "avg_top100", "avg_top10", "avg_top1", "%Pass", "Top-1 Pass"]
        
    def save_result(self, suffix=None):

        print(f"Saving molecules...")
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)
    


    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    @property
    def finish(self):
        return self.oracle.finish
        
    def _optimize(self, oracle, config):
        raise NotImplementedError
            
            
    def optimize(self, config, seed=0, project="test"):

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed 
        self.config=config
        self.oracle.task_label = self.args.mol_lm + "_" + str(self.args.max_obj) + '_' + str(self.args.min_obj) + str(seed)
        self._optimize(config)
        if self.args.log_results:
            self.log_result()
        # self.save_result(self.args.mol_lm + "_" + str(self.args.max_obj) + '_' + str(self.args.min_obj) + str(seed))
        self.reset()

        add_info = ''
        if self.constrain_K != -1:
            add_info = add_info + f'_K_{self.constrain_K}'
        if self.budget_K != -1:
            add_info = add_info + f'_BudgetK_{self.budget_K}'
        # Save to a JSON file
        if self.lm_name =='EA':
            with open(f"new_{self.args.dataset}_fitness_scores_seed_{seed}_{self.config['population_size']}_{self.config['iteration']}_{self.config['max_call']}{add_info}_baseline.json", "w") as f:
                json.dump(self.save_tuple, f)
        else:
            with open(f"new_{self.args.dataset}_fitness_scores_seed_{seed}_{self.config['population_size']}_{self.config['iteration']}_{self.config['max_call']}{add_info}_protein.json", "w") as f:
                json.dump(self.save_tuple, f)
        # self.save_violin(self.save_score, seed)
        self.save_box(self.save_score, seed, add_info)
        
    def save_box(self, fitness_scores_list, seed, add_info):

        steps = [f"Iteration {i+1}" for i in range(len(fitness_scores_list))]
        data = []
        for i, scores in enumerate(fitness_scores_list):
            for score in scores:
                data.append({"Iteration": steps[i], "Fitness": score})

        # Convert to a DataFrame
        df = pd.DataFrame(data)

        # Compute statistics (mean, min, max)
        group_stats = df.groupby("Iteration")["Fitness"].agg(["mean", "min", "max"]).reset_index()

        # Create the box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Iteration", y="Fitness", data=df, palette="muted", whis=np.inf)

        # Overlay statistics
        # plt.scatter(group_stats["Iteration"], group_stats["min"], color="red", label="Min", zorder=3)
        # plt.scatter(group_stats["Iteration"], group_stats["max"], color="blue", label="Max", zorder=3)
        plt.scatter(group_stats["Iteration"], group_stats["mean"], color="green", label="Mean", zorder=3)
        # Add grid, title, labels, and legend
        plt.xticks(rotation=45)
        plt.title("Distribution of Fitness Scores Across Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness Score")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend()
        # if self.lm_name =='EA':
        if self.lm_name =='EA':
            plt.savefig(f"./main/log/distribution_{self.args.dataset}_seed{seed}_{self.config['population_size']}_{self.config['iteration']}_{self.config['max_call']}{add_info}_baseline.png")
        else:
            plt.savefig(f"./main/log/distribution_{self.args.dataset}_seed{seed}_{self.config['population_size']}_{self.config['iteration']}_{self.config['max_call']}{add_info}_protein.png")        # else:
            # plt.savefig(f"./main/molleo/log/box{self.args.dataset}_seed{seed}_{self.config['population_size']}_{self.config['iteration']}_{self.config['max_call']}{add_info}_protein.png")