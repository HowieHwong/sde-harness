"""Dataset loader for equation discovery tasks."""

import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import h5py
import datasets
from huggingface_hub import snapshot_download

from dataclasses import dataclass


@dataclass
class EquationData:
    """Data class for equation discovery problems."""
    name: str
    symbols: List[str]
    symbol_descs: List[str]
    symbol_properties: List[str]
    expression: str
    train_data: np.ndarray
    test_data: Optional[np.ndarray] = None
    ood_test_data: Optional[np.ndarray] = None


class LLMSRDatasetLoader:
    """Loader for LLMSR benchmark datasets from HuggingFace."""
    
    def __init__(self, dataset_name: str = "lsrtransform"):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_name: Name of the dataset to load
        """
        self.dataset_name = dataset_name
        self.repo_id = "nnheui/llm-srbench"
        self._dataset_dir = None
        self._problems = []
        
    def setup(self):
        """Download and setup the dataset."""
        print(f"Setting up dataset: {self.dataset_name}")
        
        # Download dataset from HuggingFace
        self._dataset_dir = Path(snapshot_download(repo_id=self.repo_id, repo_type="dataset"))
        
        # Load dataset based on type
        if self.dataset_name == "lsrtransform":
            self._load_lsr_transform_dataset()
        elif self.dataset_name in ["bio_pop_growth", "chem_react", "matsci", "phys_osc"]:
            self._load_synthetic_dataset()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        print(f"Loaded {len(self._problems)} problems from {self.dataset_name}")
    
    def _load_lsr_transform_dataset(self):
        """Load the LSR transform dataset."""
        ds = datasets.load_dataset(self.repo_id)['lsr_transform']
        sample_h5file_path = self._dataset_dir / "lsr_bench_data.hdf5"
        
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {
                    k: v[...].astype(np.float64) 
                    for k, v in sample_file[f'/lsr_transform/{e["name"]}'].items()
                }
                
                problem = EquationData(
                    name=e['name'],
                    symbols=e['symbols'],
                    symbol_descs=e['symbol_descs'],
                    symbol_properties=e['symbol_properties'],
                    expression=e['expression'],
                    train_data=samples.get('train', samples.get('train_data')),
                    test_data=samples.get('test', samples.get('id_test_data')),
                    ood_test_data=samples.get('ood_test', samples.get('ood_test_data'))
                )
                self._problems.append(problem)
    
    def _load_synthetic_dataset(self):
        """Load synthetic datasets."""
        dataset_key = f'lsr_synth_{self.dataset_name}'
        ds = datasets.load_dataset(self.repo_id)[dataset_key]
        sample_h5file_path = self._dataset_dir / "lsr_bench_data.hdf5"
        
        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {
                    k: v[...].astype(np.float64) 
                    for k, v in sample_file[f'/lsr_synth/{self.dataset_name}/{e["name"]}'].items()
                }
                
                problem = EquationData(
                    name=e['name'],
                    symbols=e['symbols'],
                    symbol_descs=e['symbol_descs'],
                    symbol_properties=e['symbol_properties'],
                    expression=e['expression'],
                    train_data=samples.get('train', samples.get('train_data')),
                    test_data=samples.get('test', samples.get('id_test_data')),
                    ood_test_data=samples.get('ood_test', samples.get('ood_test_data'))
                )
                self._problems.append(problem)
    
    @property
    def problems(self) -> List[EquationData]:
        """Get all problems in the dataset."""
        return self._problems
    
    def get_problem_by_name(self, name: str) -> Optional[EquationData]:
        """Get a specific problem by name."""
        for problem in self._problems:
            if problem.name == name:
                return problem
        return None
    
    def get_problem_data(self, problem: EquationData) -> Dict[str, Any]:
        """
        Get formatted data for a problem.
        
        Args:
            problem: The equation problem
            
        Returns:
            Dictionary containing formatted data for the oracle
        """
        # Extract variable names and descriptions
        var_names = []
        var_descs = []
        
        # First variable is the output
        var_names.append(problem.symbols[0])
        var_descs.append(problem.symbol_descs[0])
        
        # Add input variables
        for i, (symbol, desc, prop) in enumerate(zip(problem.symbols[1:], 
                                                   problem.symbol_descs[1:], 
                                                   problem.symbol_properties[1:])):
            if "V" in prop:  # Variable property
                var_names.append(symbol)
                var_descs.append(desc)
        
        # Clean variable names
        var_names = [name.strip("$").strip("\\").replace(" ", "_").replace("text", "") 
                    for name in var_names]
        
        # Prepare input data
        if problem.train_data.ndim == 2:
            # Single input variable
            inputs = problem.train_data[:, 1:]  # All columns except first
            outputs = problem.train_data[:, 0]   # First column is output
        else:
            # Multiple input variables
            inputs = [problem.train_data[:, i] for i in range(1, problem.train_data.shape[1])]
            outputs = problem.train_data[:, 0]
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'var_names': var_names,
            'var_descs': var_descs,
            'problem_name': problem.name,
            'expression': problem.expression
        }
    
    def get_test_data(self, problem: EquationData) -> Optional[Dict[str, Any]]:
        """Get test data for a problem."""
        if problem.test_data is None:
            return None
        
        # Extract variable names (same as training data)
        var_names = []
        var_descs = []
        
        var_names.append(problem.symbols[0])
        var_descs.append(problem.symbol_descs[0])
        
        for i, (symbol, desc, prop) in enumerate(zip(problem.symbols[1:], 
                                                   problem.symbol_descs[1:], 
                                                   problem.symbol_properties[1:])):
            if "V" in prop:
                var_names.append(symbol)
                var_descs.append(desc)
        
        var_names = [name.strip("$").strip("\\").replace(" ", "_").replace("text", "") 
                    for name in var_names]
        
        # Prepare test data
        if problem.test_data.ndim == 2:
            inputs = problem.test_data[:, 1:]
            outputs = problem.test_data[:, 0]
        else:
            inputs = [problem.test_data[:, i] for i in range(1, problem.test_data.shape[1])]
            outputs = problem.test_data[:, 0]
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'var_names': var_names,
            'var_descs': var_descs,
            'problem_name': problem.name,
            'expression': problem.expression
        }
