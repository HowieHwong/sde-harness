from __future__ import annotations
from typing import List

from src.oracles.base import ProteinOracle
from src.utils.potts_model import PottsModel

def min_max_normalize(minimum,maximum,score):
    width = maximum-minimum
    normalize_score = (score-minimum)/width
    return normalize_score

class HammingDistanceOracle(ProteinOracle):
    """Oracle to calculate Hamming distance to a reference sequence."""

    def __init__(self, reference_sequence: str):
        super().__init__()
        self.reference_sequence = reference_sequence
        self.reference_as_list = list(reference_sequence)

    def _evaluate_protein_impl(self, sequence: str) -> float:
        """Calculate Hamming distance.
        Note: this is a score to be MINIMISED.
        """
        dist = sum(
            c1 != c2 for c1, c2 in zip(list(sequence), self.reference_as_list)
        )/len(sequence)
        return float(dist)


class PottsObjective(ProteinOracle):
    """Oracle to evaluate a sequence with a Potts model."""

    def __init__(self, potts_model: PottsModel):
        super().__init__()
        self._potts_model = potts_model

    def _evaluate_protein_impl(self, sequence: str) -> float:
        """Predict fitness using the Potts model."""
        return min_max_normalize(-3, 3, self._potts_model.predict(sequence))