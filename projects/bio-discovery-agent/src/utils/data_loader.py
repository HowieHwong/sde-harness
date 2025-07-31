"""Data loading utilities for BioDiscoveryAgent."""
import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any


def read_task_prompt(json_file_path: str) -> Tuple[str, str]:
    """
    Read task prompt from JSON file.
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        Tuple of (task_description, measurement)
    """
    with open(json_file_path, 'r') as f:
        prompt_data = json.load(f)
    
    task_description = prompt_data['Task']
    measurement = prompt_data['Measurement']
    
    return task_description, measurement


def load_dataset(data_name: str, base_path: str = "datasets") -> Dict[str, Any]:
    """
    Load dataset by name.
    
    Args:
        data_name: Name of the dataset
        base_path: Base path for datasets
        
    Returns:
        Dictionary containing dataset information
    """
    dataset_info = {}
    
    # Load task prompt
    prompt_path = os.path.join(base_path, "task_prompts", f"{data_name}.json")
    if os.path.exists(prompt_path):
        task_description, measurement = read_task_prompt(prompt_path)
        dataset_info["task_description"] = task_description
        dataset_info["measurement"] = measurement
    
    # Load ground truth
    ground_truth_path = os.path.join(base_path, f"ground_truth_{data_name}.csv")
    if os.path.exists(ground_truth_path):
        dataset_info["ground_truth"] = pd.read_csv(ground_truth_path)
    
    # Load top movers
    topmovers_path = os.path.join(base_path, f"topmovers_{data_name}.npy")
    if os.path.exists(topmovers_path):
        dataset_info["topmovers"] = np.load(topmovers_path, allow_pickle=True)
    
    return dataset_info


def load_essential_genes(file_path: str = "CEGv2.txt") -> List[str]:
    """
    Load essential genes from file.
    
    Args:
        file_path: Path to essential genes file
        
    Returns:
        List of essential gene names
    """
    if os.path.exists(file_path):
        essential_df = pd.read_csv(file_path, delimiter='\t')
        return essential_df['GENE'].tolist()
    return []


def validate_data_files(data_name: str, base_path: str = "datasets") -> bool:
    """
    Validate that required data files exist for a dataset.
    
    Args:
        data_name: Name of the dataset
        base_path: Base path for datasets
        
    Returns:
        True if all required files exist
    """
    required_files = [
        os.path.join(base_path, "task_prompts", f"{data_name}.json"),
        os.path.join(base_path, f"topmovers_{data_name}.npy")
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Missing required file: {file_path}")
            return False
    
    return True


def format_gene_list(genes: List[str]) -> str:
    """
    Format a list of genes for display.
    
    Args:
        genes: List of gene names
        
    Returns:
        Formatted string of genes
    """
    return ", ".join(genes)