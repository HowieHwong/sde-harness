from __future__ import annotations

from typing import List, Tuple

class WeightedSumOracle:
    """Combine multiple single-objective oracles via weighted sum."""

    def __init__(self, objectives: List[Tuple[object, float]]):
        """
        Args:
            objectives: List of (oracle, weight). A positive weight means we
                maximise the oracle score, negative weight means minimise.
        """
        self._objectives = objectives
        self.call_count = 0

    def evaluate_protein(self, seq: str) -> float:
        total = 0.0
        for oracle, w in self._objectives:
            score = oracle.evaluate_protein(seq)
            total += w * score
        self.call_count += 1
        return total 