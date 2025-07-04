"""
Data loading tool

Provide data loading, processing, and validation functions.
"""

import sys
import os
from typing import Tuple, Dict, Any, Optional

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from sci_demo.generation import Generation
from sci_demo.prompt import Prompt
from sci_demo.oracle import Oracle

from prompt import PROMPT_G
from _utils import (
    make_text_for_existing_tmcs,
    retrive_tmc_from_message,
    find_tmc_in_space,
)
import pandas as pd


def load_data(
    ligands_file: str = "data/1M-space_50-ligands-full.csv",
    fitness_file: str = "data/ground_truth_fitness_values.csv",
) -> Tuple[str, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Load data files

    Args:
        ligands_file: Ligand file path
        fitness_file: Fitness file path

    Returns:
        (df_ligands_str, df_tmc, df_ligands, LIG_CHARGE) tuple
    """
    # Check if files exist
    if not os.path.exists(ligands_file):
        raise FileNotFoundError(f"Ligand file not found: {ligands_file}")
    if not os.path.exists(fitness_file):
        raise FileNotFoundError(f"Fitness file not found: {fitness_file}")

    # Read ligand file
    with open(ligands_file, "r") as fo:
        df_ligands_str = fo.read()

    # Read fitness file content
    df_tmc = pd.read_csv(fitness_file)
    df_ligands = pd.read_csv(ligands_file)

    # Create ligand charge dictionary LIG_CHARGE
    LIG_CHARGE = {row["id"]: row["charge"] for _, row in df_ligands.iterrows()}

    return df_ligands_str, df_tmc, df_ligands, LIG_CHARGE


def setup_generator(api_key: Optional[str] = None) -> Generation:
    """
    Setup generator instance

    Args:
        api_key: API key, if None, get from environment variable

    Returns:
        Generator instance
    """
    return Generation(
        openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )


def create_prompt(
    df_ligands_str: str, text_tmc: str, num_samples: int, num_provided_samples: int
) -> Prompt:
    """
    Create prompt template

    Args:
        df_ligands_str: Ligand file content
        text_tmc: TMC sample text
        num_samples: Generated sample number
        num_provided_samples: Provided sample number

    Returns:
        Prompt instance
    """
    return Prompt(
        custom_template=PROMPT_G,
        default_vars={
            "CSV_FILE_CONTENT": df_ligands_str,
            "CURRENT_SAMPLES": text_tmc,
            "NUM_PROVIDED_SAMPLES": num_provided_samples,
            "NUM_SAMPLES": num_samples,
        },
    )


def setup_oracle(
    df_tmc: pd.DataFrame, tmc_samples: pd.DataFrame, mode: str = "multi_round"
) -> Oracle:
    """
    Setup oracle

    Args:
        df_tmc: TMC data frame
        tmc_samples: TMC sample data frame
        mode: Evaluation mode ("multi_round" or "single_round")

    Returns:
        Oracle instance
    """
    oracle = Oracle()

    if mode == "multi_round":

        def improvement_rate_metric(
            history: dict, reference: any, current_iteration: int, **kwargs
        ) -> float:
            # Initialize tmc_explorer with serializable format
            if "tmc_explorer" not in history:
                # Convert initial tmc_samples to serializable format
                history["tmc_explorer"] = [tmc_samples.to_dict('records')]
            
            current_round_tmc = find_tmc_in_space(
                reference, retrive_tmc_from_message(history["outputs"][-1], 10)
            )
            if current_round_tmc is None or current_round_tmc.empty:
                history["tmc_explorer"].append([])
            else:
                # Convert DataFrame to serializable format
                history["tmc_explorer"].append(current_round_tmc.to_dict('records'))
            
            # Convert back to DataFrame for calculations
            all_tmc_records = []
            for tmc_list in history["tmc_explorer"]:
                if tmc_list:  # Only add non-empty lists
                    all_tmc_records.extend(tmc_list)
            
            if not all_tmc_records:
                return 0.0
            
            all_tmc = pd.DataFrame(all_tmc_records)
            top10_avg_gap = all_tmc["gap"].nlargest(10).mean()
            return top10_avg_gap

        oracle.register_multi_round_metric("top10_avg_gap", improvement_rate_metric)
    else:

        def top10_avg_gap(response: str, reference: any) -> float:
            current_round_tmc = find_tmc_in_space(
                reference, retrive_tmc_from_message(response, 10)
            )
            if current_round_tmc is None or current_round_tmc.empty:
                return 0
            else:
                return current_round_tmc["gap"].nlargest(10).mean()

        oracle.register_metric("top10_avg_gap", top10_avg_gap)

    return oracle


def validate_data_files(
    ligands_file: str = "data/1M-space_50-ligands-full.csv",
    fitness_file: str = "data/ground_truth_fitness_values.csv",
) -> bool:
    """
    Validate data files

    Args:
        ligands_file: Ligand file path
        fitness_file: Fitness file path

    Returns:
        Boolean: if the data files are valid
    """
    try:
        # Check if files exist
        if not os.path.exists(ligands_file):
            print(f"Error: Ligand file not found: {ligands_file}")
            return False
        if not os.path.exists(fitness_file):
            print(f"Error: Fitness file not found: {fitness_file}")
            return False

        # Check if files are readable
        df_ligands = pd.read_csv(ligands_file)
        df_tmc = pd.read_csv(fitness_file)

        # Check if required columns exist
        required_ligand_cols = ["id", "charge"]
        required_tmc_cols = ["lig1", "lig2", "lig3", "lig4", "gap"]

        missing_ligand_cols = [
            col for col in required_ligand_cols if col not in df_ligands.columns
        ]
        missing_tmc_cols = [
            col for col in required_tmc_cols if col not in df_tmc.columns
        ]

        if missing_ligand_cols:
            print(f"Error: Ligand file missing required columns: {missing_ligand_cols}")
            return False
        if missing_tmc_cols:
            print(f"Error: Fitness file missing required columns: {missing_tmc_cols}")
            return False

        print("✅ Data files validated successfully")
        return True

    except Exception as e:
        print(f"Error: Data file validation failed: {e}")
        return False
