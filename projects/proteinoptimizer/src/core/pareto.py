from __future__ import annotations

from typing import List, Tuple
import numpy as np


def dominates(a: List[float], b: List[float]) -> bool:
    """Return True if a dominates b.
    Assumes obj1 (Potts) is maximized, obj2 (Hamming) is minimized.
    """
    # Potts score (higher is better) and Hamming distance (lower is better)
    score_a, dist_a = a
    score_b, dist_b = b
    
    # Check for Pareto dominance
    return (score_a >= score_b and dist_a <= dist_b) and \
           (score_a > score_b or dist_a < dist_b)


def non_dominated_sort(scores: List[List[float]]) -> List[int]:
    """Return indices of the non-dominated set (Pareto front)."""
    n = len(scores)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if dominates(scores[i], scores[j]):
                dominated[j] = True
            elif dominates(scores[j], scores[i]):
                dominated[i] = True
                break
    return [i for i, d in enumerate(dominated) if not d] 