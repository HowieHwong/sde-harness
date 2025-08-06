from __future__ import annotations
from typing import List, Tuple
import numpy as np
from ..oracles.protein_oracles import ProteinOracle
from ..utils.potts_model import PottsModel

class HammingDistanceOracle(ProteinOracle):
    """Calculates Hamming distance to a reference sequence."""
    def __init__(self, reference_sequence: str):
        super().__init__("hamming_distance")
        self.reference_sequence = reference_sequence

    def _evaluate_protein_impl(self, sequence: str) -> float:
        dist = sum(c1 != c2 for c1, c2 in zip(self.reference_sequence, sequence))
        return float(dist)

class PottsObjective(ProteinOracle):
    """Wrapper for PottsModel to be used as an objective."""
    def __init__(self, potts_model: PottsModel):
        super().__init__("potts_energy")
        self._potts = potts_model

    def _evaluate_protein_impl(self, sequence: str) -> float:
        from ..oracles.protein_oracles import AA_TO_NUM
        seq_numeric = [AA_TO_NUM.get(aa, 20) for aa in sequence]
        return float(self._potts.evaluate(np.array(seq_numeric))) 