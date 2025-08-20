#!/usr/bin/env python3
"""
LLM-Syn-Planner - LLM-based Retrosynthesis Pathway Design
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
from src.utils.utils import setup_logging
from src.utils.chemistry_utils import MORGAN_FP_GENERATOR


TRAIN_DATA = os.path.join(os.path.dirname(__file__), "dataset", "routes_train.pkl")
VAL_DATA = os.path.join(os.path.dirname(__file__), "dataset", "routes_val.pkl")
FPS_FILE = os.path.join(os.path.dirname(__file__), "dataset", "reference_fps.pkl")


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

def main() -> None:
    """Main entry point for the LLM-Syn-Planner CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_smiles", default=None, type=str, required=True)
    parser.add_argument("--config_default", default="./src/hparams_default.yaml")
    parser.add_argument("--output_dir", type=str, default="./synplanner_results")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_oracle_calls", type=int, default=100)
    parser.add_argument("--freq_log", type=int, default=5)
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

    start_time = time.perf_counter()

    config_default = yaml.safe_load(open(args.config_default))

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "log.log"))
    logging.info("=" * 60)
    logging.info("🧪 LLM-Syn-Planner - Starting synthetic route planning")
    logging.info("=" * 60)
    logging.info(f"📋 Arguments: {args}")
    logging.info(f"⚙️  Hyperparameters: {config_default}")
    logging.info("=" * 60)

    # Get reference retrosynthesis routes
    reference_target_list, reference_route_list = get_reference_routes()

    # Get input SMILES strings
    input_smiles = [line.strip() for line in open(args.target_smiles)] if os.path.isfile(args.target_smiles) else [args.target_smiles]
    input_valid_smiles = [s for s in input_smiles if Chem.MolFromSmiles(s) is not None]
    if not input_valid_smiles: raise ValueError("No valid SMILES strings found in input")
    logging.info(f"Found {len(input_valid_smiles)}/{len(input_smiles)} valid SMILES strings in input")

    # Get reference molecule fingerprints
    all_fps = get_reference_morgan_fps(reference_target_list)

    # Run the optimizer
    for seed in args.seed:
        for idx in range(len(input_valid_smiles)):
            logging.info("=" * 60)
            logging.info(f"Searching synthetic routes for: {input_valid_smiles[idx]}")
            logging.info("=" * 60)
            optimizer = RouteOptimizer(args)  # Re-initialization for clean state
            optimizer.optimize(
                target=input_valid_smiles[idx], 
                route_list=reference_route_list, 
                all_fps=all_fps, 
                config=config_default, 
                seed=seed
            )

    end_time = time.perf_counter()
    logging.info(f"LLM-Syn-Planner ran for {end_time - start_time:.2f} seconds for {len(input_valid_smiles)} target(s)")

if __name__ == "__main__":
    main()
