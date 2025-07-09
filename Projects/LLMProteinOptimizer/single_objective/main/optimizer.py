import os
import yaml
import random
import torch
import numpy as np
import math
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import json
import seaborn as sns
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein

def filter_dataset(top_quantile,data_df, percentile, min_mutant_dist):
    lower_value = data_df.score.quantile(percentile[0])
    upper_value = data_df.score.quantile(percentile[1])
    top_quantile = data_df.score.quantile(top_quantile)
    top_sequences_df = data_df[data_df.score >= top_quantile]  

    filtered_df = data_df[data_df.score.between(lower_value, upper_value)]
    if min_mutant_dist == 0:
        return filtered_df
    get_min_dist = lambda x: np.min([levenshtein(x.strip(), top_seq.strip()) for top_seq in top_sequences_df.sequence]) 
    print('Getting minimum Levenshtein distance to top sequences')
    mutant_dist = filtered_df.sequence.progress_map(get_min_dist)
    return filtered_df[mutant_dist >= min_mutant_dist].reset_index(drop=True)

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
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False)) # increasing order
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
    def __init__(self, args=None):
        # print('call init', len(mol_buffer))
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log

        # self.mol_buffer = mol_buffer
        self.mol_buffer = {}
        # print(self.mol_buffer)
        # self.sa_scorer = tdc.Oracle(name = 'SA')
        # self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0

        self.oracle_name=None


    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):

        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        # print(self.mol_buffer)
        # print(output_file_path)
        # exit()
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
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)

        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        # avg_sa = np.mean(self.sa_scorer(smis))
        # diversity_top100 = self.diversity_evaluator(smis)


        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f} | '
                # f'avg_sa: {avg_sa:.3f} | '
                # f'div: {diversity_top100:.3f}'
                )

        print({
            "avg_top1": avg_top1,
            "avg_top10": avg_top10,
            "avg_top100": avg_top100,
            "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
            "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
            "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
            # "avg_sa": avg_sa,
            # "diversity_top100": diversity_top100,
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
        # print(smi)
        if smi is None:
            return 0
        if len(smi) == 0:
            return 0
        else:
            if smi in self.mol_buffer:
                pass
            elif len(self.mol_buffer) > self.max_oracle_calls:
                return 0
            else:
                fitness = float(self.evaluator(smi))
                #print(fitness, type(fitness))
                if math.isnan(fitness):
                    fitness = 0
                if "docking" in self.args.oracles[0]:
                    fitness *= -1

                self.mol_buffer[smi] = [fitness, len(self.mol_buffer)+1]
            return self.mol_buffer[smi][0]

    def __call__(self, smiles_lst):
        """
        Score
        """
        if type(smiles_lst) == list:
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
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

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None):
        self.model_name = args.mol_lm
        self.args = args
        self.constrain_K = args.constrain_K
        self.budget_K = args.budget_K
        self.n_jobs = args.n_jobs
        self.wild_type={
            'GB1': 'VDGV',
            'TrpB': 'VFVS',
            'Syn-3bfo': 'SKLQICVEPTSQKLMPGSTLVLQCVAVGSPIPHYQWFKNELPLTHETKKLYMVPYVDLEHQGTYWCHVYNDRDSQDSKKVEIIID',
            'AAV':'DEEEIRTTNPVATEQYGSVSTNLQRGNR',
            'GFP':'SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
        }[args.dataset]
        self.oracle = Oracle(args=self.args)
        if args.dataset == 'AAV' or args.dataset == 'GFP':   
            self.all_protein = pd.read_csv(f'../data/{args.dataset}/ground_truth.csv').sort_values(by='score')['sequence'].tolist()   
            self.use_medium_range = args.use_medium_range           
            if self.use_medium_range:
                if not os.path.exists(f'../data/{args.dataset}/gt_medium_range.csv'):
                    df_filtered = filter_dataset(data_df = pd.read_csv(f'../data/{args.dataset}/ground_truth.csv'),top_quantile=0.99,min_mutant_dist=6,percentile=[0.2,0.4])
                    df_filtered.to_csv(f"../data/{args.dataset}/gt_medium_range.csv", index=False)
                self.all_protein = pd.read_csv(f'../data/{args.dataset}/gt_medium_range.csv').sort_values(by='score')['sequence'].tolist()
            self.seq_len = len(self.all_protein[0])
        else:
            self.all_protein = pd.read_csv(f'../data/{args.dataset}/fitness.csv').sort_values(by='fitness')['Combo'].tolist()
            self.wild_type=self.all_protein[-1]
            self.seq_len = len(self.wild_type)
    def filter(self, proteins):
        # print('here',self.seq_len)
        result = []
        # print(self.all_protein)
        for i in proteins:
            if len(i) == self.seq_len:
                # if i in self.all_protein:
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

    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        smis_pass = self.filter(smis)
        if len(smis_pass) == 0:
            top1_pass = -1
        else:
            top1_pass = np.max([scores_dict[s] for s in smis_pass])
        return [np.mean(scores),
                np.mean(scores[:10]),
                np.max(scores),
                # self.diversity_evaluator(smis),
                # np.mean(self.sa_scorer(smis)),
                float(len(smis_pass) / 100),
                top1_pass]

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



    def optimize(self, oracle, config, seed=0, project="test"):

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed

        self.oracle.task_label = self.args.mol_lm + "_" + oracle.name + "_" + str(seed)
        
        self._optimize(oracle, config)
        self.config=config
        if self.args.log_results:
            self.log_result()

        self.save_result(self.args.mol_lm + "_" + oracle.name + "_" + str(seed))
        del self.oracle
        # print('before')
        self.oracle = Oracle(args=self.args)
        # print(len(self.oracle.mol_buffer))
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
        
        # Save to a JSON file
        if self.lm_name =='EA':
            with open(f"new_{self.args.dataset}_fitness_scores_seed_{seed}_{self.config['population_size']}_{self.config['iteration']}_{self.config['max_call']}{add_info}_baseline.json", "w") as f:
                json.dump(self.save_tuple, f)
        else:
            with open(f"new_{self.args.dataset}_fitness_scores_seed_{seed}_{self.config['population_size']}_{self.config['iteration']}_{self.config['max_call']}{add_info}_protein.json", "w") as f:
                json.dump(self.save_tuple, f)
    
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
            plt.savefig(f"./main/molleo/log/distribution_{self.args.dataset}_seed{seed}_{self.config['population_size']}_{self.config['iteration']}_{self.config['max_call']}{add_info}_baseline.png")
        else:
            plt.savefig(f"./main/molleo/log/distribution_{self.args.dataset}_seed{seed}_{self.config['population_size']}_{self.config['iteration']}_{self.config['max_call']}{add_info}_protein.png")        # else:
            # plt.savefig(f"./main/molleo/log/box{self.args.dataset}_seed{seed}_{self.config['population_size']}_{self.config['iteration']}_{self.config['max_call']}{add_info}_protein.png")