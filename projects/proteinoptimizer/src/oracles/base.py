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


class ProteinOracle(Oracle):
    """Base class for molecular property oracles"""
    
    def __init__(self, property_name: str):
        super().__init__()
        self.property_name = property_name
        self.call_count = 0
        self.history = []
        
    def evaluate_protein(self, sequence: str) -> float:
        """Evaluate a single protein sequence"""
        score = self._evaluate_protein_impl(sequence)
        self.call_count += 1
        self.history.append({
            "input": sequence,
            "score": score,
            "call_count": self.call_count
        })
        return score
    
    @abstractmethod
    def _evaluate_protein_impl(self, sequence: str) -> float:
        """Implementation of protein sequence evaluation (to be overridden)"""
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
            # Single protein sequence
            score = self.evaluate_protein(response)
        elif isinstance(response, list):
            # List of protein sequences - return average
            scores = [self.evaluate_protein(seq) for seq in response]
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