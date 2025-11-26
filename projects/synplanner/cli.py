#!/usr/bin/env python3
"""
LLM-Syn-Planner - LLM-based Retrosynthesis Route Planning
"""
from typing import Tuple, List
import argparse
import os
import time
import logging
import yaml
import pickle

from rdkit import Chem

from src.optimizer.route_optimizer import RouteOptimizer
from src.utils.utils import setup_logging, set_seed
from src.utils.chemistry_utils import MORGAN_FP_GENERATOR


TRAIN_DATA = os.path.join(os.path.dirname(__file__), "dataset", "routes_train.pkl")
VAL_DATA = os.path.join(os.path.dirname(__file__), "dataset", "routes_val.pkl")
USPTO_EASY_DATA = os.path.join(os.path.dirname(__file__), "dataset", "simple_200.pkl")
TEST_HARD_DATA = os.path.join(os.path.dirname(__file__), "dataset", "routes_possible_test_hard.pkl")
PISTACHIO_REACHABLE_DATA = os.path.join(os.path.dirname(__file__), "dataset", "pistachio_reachable_targets.txt")
PISTACHIO_HARD_DATA = os.path.join(os.path.dirname(__file__), "dataset", "pistachio_hard_targets.txt")
FPS_FILE = os.path.join(os.path.dirname(__file__), "dataset", "reference_fps.pkl")


def format_dict_for_logging(
    d: dict, 
    name: str
) -> str:
    """Format a dictionary for logging."""
    formatted_str = f"{name}: {{\n"
    for key, value in d.items():
        formatted_str += f"\t'{key}': {value},\n"
    formatted_str = formatted_str.rstrip(',\n') + "\n}"
    return formatted_str

def get_reference_routes() -> Tuple[List[str], List[List[str]]]:
    """Load reference routes from the training and validation datasets."""
    logging.info(f"Loading reference routes from {TRAIN_DATA} and {VAL_DATA}")
    train_routes = pickle.load(open(TRAIN_DATA, "rb"))
    val_routes = pickle.load(open(VAL_DATA, "rb"))
    total_routes = train_routes + val_routes

    reference_target_list = []
    reference_route_list = []
    for route in total_routes:
        reference_target_list.append(route[0].split(">>")[0])
        reference_route_list.append(route)

    return reference_target_list, reference_route_list

def get_reference_morgan_fps(reference_target_list: List[str]) -> List[int]:
    """Load or compute Morgan fingerprints for all reference molecules (molecules with reference synthetic routes)."""
    if os.path.exists(FPS_FILE):
        logging.info("Loading reference molecules' fingerprints from file...")
        all_fps = pickle.load(open(FPS_FILE, "rb"))
    else:
        logging.info(f"Computing Morgan fingerprints of reference molecules... will be stored in ./dataset/reference_fps.pkl")
        all_fps = [MORGAN_FP_GENERATOR(smi) for smi in reference_target_list]
        pickle.dump(all_fps, open(FPS_FILE, "wb"))

    return all_fps

def check_args(args: argparse.Namespace) -> None:
    """Check arguments are valid/complete."""
    assert os.path.exists("./dataset"), "dataset directory not found, please download the required data following the README"
    assert not (args.target_smiles is None and args.dataset is None), "Either target_smiles or dataset must be specified"
    
    if args.dataset is not None:
        valid_datasets = ["uspto-easy", "uspto-190", "pistachio-reachable", "pistachio-hard"]
        assert args.dataset in valid_datasets, f"dataset must be one of {valid_datasets}, got {args.dataset}"
        
    assert args.model in ["gpt-4o", "gpt-5-mini", "gpt-5", "gpt-5-chat-latest", "claude-sonnet-4-5", "grok-4", "deepseek-reasoner"], f"model must be one of ['gpt-4o', 'gpt-5-mini', 'gpt-5', 'gpt-5-chat-latest', 'claude-sonnet-4-5', 'grok-4', 'deepseek-reasoner'], got {args.model}"
    if args.model in ["gpt-4o", "gpt-5-mini", "gpt-5", "gpt-5-chat-latest"]: 
        assert os.getenv("OPENAI_API_KEY") is not None, f"Specified to use {args.model} but OPENAI_API_KEY is not set"
        if args.model in ["gpt-5-mini", "gpt-5", "gpt-5-chat-latest"]: assert args.temperature == 1.0, f"gpt-5-mini, gpt-5, and gpt-5-chat-latest only support temperature=1.0 (user input is {args.temperature})"
    if args.model == "claude-sonnet-4-5": assert os.getenv("ANTHROPIC_API_KEY") is not None, f"Specified to use {args.model} but ANTHROPIC_API_KEY is not set"
    if args.model == "grok-4": assert os.getenv("XAI_API_KEY") is not None, f"Specified to use {args.model} but XAI_API_KEY is not set"
    if args.model == "deepseek-reasoner": assert os.getenv("DEEPSEEK_API_KEY") is not None, f"Specified to use {args.model} but DEEPSEEK_API_KEY is not set"

def get_targets(dataset: str) -> List[str]:
    """Get targets from the specified dataset."""
    if dataset == "uspto-easy":
        with open(USPTO_EASY_DATA, "rb") as file:
            return pickle.load(file)

    elif dataset == "uspto-190":
        with open(TEST_HARD_DATA, "rb") as file:
            return [route[0].split(">>")[0] for route in pickle.load(file)]

    elif dataset == "pistachio-reachable":
        with open(PISTACHIO_REACHABLE_DATA, "r") as file:
            return [eval(line)[0] for line in file]

    elif dataset == "pistachio-hard":
        with open(PISTACHIO_HARD_DATA, "r") as file:
            return [eval(line)[0] for line in file]

def main() -> None:
    """Main entry point for the LLM-Syn-Planner CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_smiles", default=None, type=str, required=False)
    parser.add_argument("--dataset", default=None, type=str, required=False)
    parser.add_argument("--model", default="gpt-5", type=str, required=False)
    parser.add_argument("--config_default", default="./src/hparams_default.yaml")
    parser.add_argument("--output_dir", type=str, default="./synplanner_results")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_oracle_calls", type=int, default=100)
    parser.add_argument("--freq_log", type=int, default=5)
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    # Check arguments are valid/complete
    check_args(args)
    
    start_time = time.perf_counter()

    config_default = yaml.safe_load(open(args.config_default))
    # Set temperature
    config_default["temperature"] = args.temperature

    # TODO: Could add temperature to output directory name
    if args.dataset is not None: 
        args.output_dir = os.path.join(args.output_dir, args.model, args.dataset, str(args.max_oracle_calls))
    else: 
        args.output_dir = os.path.join(args.output_dir, args.model, str(args.max_oracle_calls))
    os.makedirs(args.output_dir, exist_ok=True)

    setup_logging(os.path.join(args.output_dir, "log.log"))
    logging.info("=" * 60)
    logging.info("🧪 LLM-Syn-Planner - Starting synthesis route planning")
    logging.info("=" * 60)
    logging.info(format_dict_for_logging(vars(args), "📋 Arguments"))
    logging.info(format_dict_for_logging(config_default, "⚙️  Hyperparameters"))
    logging.info("=" * 60)

    # Get reference retrosynthesis routes
    reference_target_list, reference_route_list = get_reference_routes()

    # Get targets either from input SMILES, input file containing SMILES, or from the specified dataset
    if args.dataset is not None: 
        input_smiles = get_targets(args.dataset)
    else: 
        input_smiles = [line.strip() for line in open(args.target_smiles)] if os.path.isfile(args.target_smiles) else [args.target_smiles]

    input_valid_smiles = [s for s in input_smiles if Chem.MolFromSmiles(s) is not None]
    if not input_valid_smiles: raise ValueError("No valid SMILES strings found in input")
    logging.info(f"Found {len(input_valid_smiles)}/{len(input_smiles)} valid SMILES strings in input")

    # Get reference molecule fingerprints
    all_fps = get_reference_morgan_fps(reference_target_list)

    # Run the optimizer
    for seed in args.seed:
        set_seed(seed)
        total_llm_cost = 0
        for target in input_valid_smiles:
            try:
                logging.info("=" * 60)
                logging.info(f"Searching synthesis routes for: {target}")
                logging.info("=" * 60)
                optimizer = RouteOptimizer(args)  # Re-initialization for clean state
                optimizer.optimize(
                    target=target, 
                    route_list=reference_route_list, 
                    all_fps=all_fps, 
                    config=config_default, 
                    seed=seed
                )
                total_llm_cost += optimizer.oracle.cost
            except Exception as e:
                logging.error(f"Error during synthesis route search for {target}: {e}")
                continue

    end_time = time.perf_counter()
    logging.info(f"LLM-Syn-Planner ran for {end_time - start_time:.2f} seconds for {len(input_valid_smiles)} target(s)")
    logging.info(f"Total LLM cost: {total_llm_cost:.2f}")

if __name__ == "__main__":
    main()
