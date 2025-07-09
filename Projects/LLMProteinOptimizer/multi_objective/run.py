from __future__ import print_function
import os

import argparse
import yaml
import os
import sys
sys.path.append(os.path.realpath(__file__))
from time import time 

def main():
    start_time = time() 
    parser = argparse.ArgumentParser()
    parser.add_argument('method', default='molleo_multi')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--pickle_directory', help='Directory containing pickle files with the distribution statistics', default=None)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--mol_lm', type=str, default=None, choices=[None, "BioT5", "MoleculeSTM", "GPT-4", "Llama3", "EA"])
    parser.add_argument('--bin_size', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--population_size', type=int, default=96)
    parser.add_argument('--max_oracle_calls', type=int, default=384)
    parser.add_argument('--iteration', type=int, default=4)
    parser.add_argument('--freq_log', type=int, default=100)
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed', type=int, nargs="+", default=[0])
    parser.add_argument('--max_obj', nargs="+", default=["jnk3"]) ### 
    parser.add_argument('--min_obj', nargs="+", default=["sa"]) ### 
    parser.add_argument('--task_mode', type=str, default='1')
    parser.add_argument('--dataset', default='TrpB')
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_dir', default="./results")
    args = parser.parse_args()


    args.method = args.method.lower() 

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    sys.path.append(path_main)
    
    print(args.method)
    # Add method name here when adding new ones

    if args.method == 'molleo_multi':
        from main.molleo_multi.run import GB_GA_Optimizer as Optimizer
    elif args.method == 'molleo_multi_pareto':
        from main.molleo_multi_pareto.run import GB_GA_Optimizer as Optimizer
    else:
        raise ValueError("Unrecognized method name.")


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
        
    if args.population_size != -1:
        config_default['population_size'] = args.population_size
    if args.iteration != -1:
        config_default['iteration'] = args.iteration    
    if args.max_oracle_calls != -1:
        config_default['max_call'] = args.max_oracle_calls  
                
    if config_default['max_call'] == -1:

        args.max_oracle_calls = config_default['population_size'] * config_default['iteration']
        config_default['max_call'] = args.max_oracle_calls
    else:
        args.max_oracle_calls = config_default['max_call']
        
    optimizer = Optimizer(args=args)
    print(config_default)
                
    for seed in args.seed:
        print('seed', seed)
        optimizer.optimize(config=config_default, seed=seed)



    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))
    # print('If the program does not exit, press control+c.')


if __name__ == "__main__":
    main()

