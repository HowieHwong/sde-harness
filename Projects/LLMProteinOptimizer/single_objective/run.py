from __future__ import print_function
import os

import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
sys.path.append("..")
from time import time 
import pandas as pd
import torch
from omegaconf import OmegaConf

import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# export CUDA_VISIBLE_DEVICES=2,3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def min_max_normalize(minimum,maximum,score):
    width = maximum-minimum
    normalize_score = (score-minimum)/width
    return normalize_score
def hamming_distance(seq1, seq2, padding_char="-"):
    # if not isinstance(seq1, str):
    #     seq1 = seq1.tolist()  
    #     seq1 = " ".join(map(str,seq1)) 
    # if not isinstance(seq2, str):
    #     seq2 = seq2.tolist()  
    #     seq2 = " ".join(map(str,seq2)) 
    max_length = max(len(seq1), len(seq2))
    padded_seq1 = seq1.ljust(max_length, padding_char)
    padded_seq2 = seq2.ljust(max_length, padding_char)
    
    distance =  sum(ch1 != ch2 for ch1, ch2 in zip(padded_seq1, padded_seq2))
    #normalization by using max_length
    return distance/(max_length+1e-7)

import potts_model

ALPHABET = "ARNDCQEGHILKMFPSTWYV-"
A2N = {a: n for n, a in enumerate(ALPHABET)}
A2N["X"] = 20
# def potts_energy(sequences):
#     """Compute the Potts model energy."""
    
#     if len(np.asarray(sequences).shape) == 1:  # single sequence
#         sequences = np.reshape(sequences, (1, -1))
#     onehot_seq = onehot(sequences, num_classes=20)
#     linear_term = 1.0 * np.einsum(
#         'ij,bij->b', 1.0 onehot_seq, optimize='optimal') + (
#             1.0 - 1.0) * np.einsum(
#                 'ij,bij->b', self._quad_deriv, onehot_seq, optimize='optimal')
#     quadratic_term = self.1.0 * 0.5 * np.einsum(
#         'ijkl,bik,bjl->b',
#         self._weight_matrix,
#         onehot_seq,
#         onehot_seq,
#         optimize='optimal')

#     return linear_term + quadratic_term
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
            return self.landscape.evaluate(sequencns_lst)
        
        if type(sequencns_lst) == list:
            return [self.fitness_dict[s] for s in sequencns_lst]
        return self.fitness_dict[sequencns_lst]


class Oracle_model():
    def __init__(self, dataset):
        self.name = dataset 
        self.df = pd.read_csv(f'../data/{dataset}/ground_truth.csv')
        self.score_max = self.df['score'].max()
        self.score_min = self.df['score'].min()
        self.wild_type_seq = pd.read_csv(f'../data/{dataset}/{dataset}_wild_type.csv').wild_type_sequence.iloc[0]
        self.hamming_dist_weight = 0.1
        self.use_hamming_dist = False

        self.min_max_normalization = True
        if dataset == 'GFP':
            
            from GGS_utils.tokenize import Encoder
            self.oracle = get_model(predictor_dir='../GGS_utils/ckpt/GFP/mutations_7/percentile_0.0_0.3/unsmoothed_smoothed/01_03_2025_23_56',
                                        oracle_dir= '../GGS_utils/ckpt/GFP/mutations_0/percentile_0.0_1.0',)
        elif dataset == 'AAV':
            from GGS_utils.tokenize import Encoder
            self.oracle = get_model(predictor_dir=None,
                            oracle_dir= '../GGS_utils/ckpt/AAV/mutations_0/percentile_0.0_1.0',)

        self.predictor_tokenizer = Encoder()
        
    def __call__(self,  *args, **kwargs):#DF = None,

        seqs = args[0]
        # distance = hamming_distance(self.wild_type_seq, seqs)
        # print(distance)
        if type(seqs) != list:
            seqs = [seqs]
        batch_size = len(seqs)
        # print(batch_size)
        # tokenized_seqs = self.predictor_tokenizer.encode(sampled_seqs).to(device)
        tokenized_seqs = self.predictor_tokenizer.encode(seqs).to(device)

        batches = torch.split(tokenized_seqs, batch_size, 0)
        # tokenized_wt = self.predictor_tokenizer.encode(self.wild_type_seq).to(device)      
        self.oracle.eval()
        scores = []
        for b in batches:
            if b is None:
                continue
            results = self.oracle(b).detach()
            if self.min_max_normalization:
                results = min_max_normalize(self.score_min,self.score_max,results)
            if self.use_hamming_dist:
                str_protein_seq = self.predictor_tokenizer.decode(b)

                hamming_dist = hamming_distance(str_protein_seq[0],self.wild_type_seq)

             
                scores.append(results-hamming_dist)
            else:
                scores.append(results)
            

        if len(scores) == 1:
            return scores[0]
        
        return scores
def get_model(predictor_dir,oracle_dir,use_oracle = True):
    from GGS_utils.predictors import BaseCNN
    if use_oracle:
        oracle_path = os.path.join(oracle_dir, 'cnn_oracle.ckpt')
        oracle_state_dict = torch.load(oracle_path, map_location=device)
        cfg_path = os.path.join(oracle_dir, 'config.yaml')
        with open(cfg_path, 'r') as fp:
            ckpt_cfg = OmegaConf.load(fp.name)
        cnn_oracle = BaseCNN(**ckpt_cfg.model.predictor) #oracle has same architecture as predictor
        cnn_oracle.load_state_dict(
            {k.replace('predictor.', ''): v for k,v in oracle_state_dict['state_dict'].items()})
        cnn_oracle.to(device)
        return cnn_oracle
    else:
        cfg_path = os.path.join(oracle_dir, 'config.yaml')
        with open(cfg_path, 'r') as fp:
            ckpt_cfg = OmegaConf.load(fp.name)
        predictor_path = os.path.join(predictor_dir, 'last.ckpt')
        predictor_state_dict = torch.load(predictor_path, map_location=device)
        predictor = BaseCNN(**ckpt_cfg.model.predictor) 
        predictor.load_state_dict(
            {k.replace('predictor.', ''): v for k,v in predictor_state_dict['state_dict'].items()})
        predictor = predictor.to(device)
        return predictor
    return None

def main():
    start_time = time() 
    parser = argparse.ArgumentParser()
    parser.add_argument('method', default='molleo')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--pickle_directory', help='Directory containing pickle files with the distribution statistics', default=None)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--mol_lm', type=str, default='Llama3', choices=[None, "BioT5", "MoleculeSTM", "GPT-4", "Llama3","EA"])
    parser.add_argument('--bin_size', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--population_size', type=int, default=50)
    parser.add_argument('--max_oracle_calls', type=int, default=-1)
    parser.add_argument('--constrain_K', type=int, default=-1)
    parser.add_argument('--budget_K', type=int, default=-1)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--seed', type=int, nargs="+", default=[1, 2, 12,14,18,43])
    parser.add_argument('--oracles', nargs="+", default=["fitness"]) ###
    parser.add_argument('--dataset', default='GB1')
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_dir', default="./results")
    parser.add_argument('--use_medium_range', action='store_true',default=False)
    # parser.add_argument('--device',type = str, default='0')

    args = parser.parse_args()
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    args.method = args.method.lower() 

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    sys.path.append(path_main)
    
    # print(args.method)
    # Add method name here when adding new ones

    from main.molleo.run import GB_GA_Optimizer as Optimizer



    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.pickle_directory is None:
        args.pickle_directory = path_main
    

    try:
        config_default = yaml.safe_load(open(args.config_default))
    except:
        config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))
    # oracle = Oracle(name = oracle_name)
    if args.population_size != -1:
        config_default['population_size'] = args.population_size
    if args.iteration != -1:
        config_default['iteration'] = args.iteration    
    if args.max_oracle_calls != -1:
        config_default['max_call'] = args.max_oracle_calls  
                
    if config_default['max_call'] == -1:

        args.max_oracle_calls = config_default['population_size'] * (config_default['iteration'] + 1)
        args.max_oracle_calls = config_default['population_size'] * (config_default['iteration']+1)
        config_default['max_call'] = args.max_oracle_calls
    else:
        args.max_oracle_calls = config_default['max_call']
    if args.dataset == 'AAV' or args.dataset == 'GFP':
        oracle = Oracle_model(args.dataset)
    else:
        oracle = Oracle_fitness(args.dataset)

    optimizer = Optimizer(args=args)
    
    for seed in args.seed:
        print('seed', seed)

        optimizer.optimize(oracle=oracle, config=config_default, seed=seed)





    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))


if __name__ == "__main__":
    main()

