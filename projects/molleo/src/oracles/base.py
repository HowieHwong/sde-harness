"""Base oracle class for molecular property evaluation"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import sys
import os

# Add SDE harness to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from sde_harness.core import Oracle


class MolecularOracle(Oracle):
    """Base class for molecular property oracles"""
    
    def __init__(self, property_name: str):
        super().__init__()
        self.property_name = property_name
        self.call_count = 0
        self.history = []
        
    def evaluate_molecule(self, smiles: str) -> float:
        """Evaluate a single molecule"""
        score = self._evaluate_molecule_impl(smiles)
        self.call_count += 1
        self.history.append({
            "input": smiles,
            "score": score,
            "call_count": self.call_count
        })
        return score
    
    @abstractmethod
    def _evaluate_molecule_impl(self, smiles: str) -> float:
        """Implementation of molecule evaluation (to be overridden)"""
        pass
        
    def evaluate(self, response: Any, reference: Any = None) -> float:
        """
        Evaluate response from generation model
        
        Args:
            response: SMILES string or list of SMILES
            reference: Optional reference data
            
        Returns:
            Score (float)
        """
        if isinstance(response, str):
            # Single SMILES
            score = self.evaluate_molecule(response)
        elif isinstance(response, list):
            # List of SMILES - return average
            scores = [self.evaluate_molecule(smi) for smi in response]
            score = sum(scores) / len(scores) if scores else 0.0
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")
            
        self.call_count += 1
        self.history.append({
            "input": response,
            "score": score,
            "call_count": self.call_count
        })
        
        return score
        
    def reset(self):
        """Reset oracle state"""
        self.call_count = 0
        self.history = []