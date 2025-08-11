"""Core components for protein optimization"""

# Expose the ProteinOptimizer for sequence-based tasks
from .protein_optimizer import ProteinOptimizer

__all__ = [
    "ProteinOptimizer",
]