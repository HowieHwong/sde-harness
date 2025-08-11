"""Base oracle class for molecular property evaluation"""

from __future__ import annotations
from abc import abstractmethod
from typing import Any

from sde_harness.core import Oracle

class ProteinOracle(Oracle):
    """
    Base class for all protein oracles in this project. It inherits from the
    SDE Harness Oracle to be compatible with the Workflow system, and adds
    protein-specific evaluation logic.
    """
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.history = []

    def evaluate_protein(self, sequence: str) -> float:
        """
        Public method to evaluate a protein sequence. It increments the call
        counter and then calls the specific implementation.
        """
        score = self._evaluate_protein_impl(sequence)
        self.call_count += 1
        self.history.append({"input": sequence, "score": score, "call_count": self.call_count})
        return score

    def evaluate(self, response: Any, reference: Any = None) -> float:
        """
        SDE-Harness compatible evaluate method. The workflow calls this method.
        It expects the response to be a dictionary containing the sequence.
        """
        if isinstance(response, dict) and "text" in response:
            sequence = response["text"]
        elif isinstance(response, str):
            sequence = response
        else:
            raise TypeError(f"Unsupported response type for oracle evaluation: {type(response)}")
        
        return self.evaluate_protein(sequence)

    @abstractmethod
    def _evaluate_protein_impl(self, sequence: str) -> float:
        """
        Subclasses must implement this method to define their evaluation logic.
        """
        raise NotImplementedError