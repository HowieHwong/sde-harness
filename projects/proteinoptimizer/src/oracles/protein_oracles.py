"""Protein oracles for sequence-based optimization"""

import pandas as pd
from .base import ProteinOracle
import os
import sys

# Attempt to import local Potts model implementation.
try:
    from ..utils import potts_model  # type: ignore
except Exception:
    potts_model = None

# 21-character alphabet used by the original Potts implementation (20 AA + gap)
ALPHABET = "ARNDCQEGHILKMFPSTWYV-"
AA_TO_NUM = {aa: idx for idx, aa in enumerate(ALPHABET)}

class Syn3bfoOracle(ProteinOracle):
    """Oracle for Syn-3bfo dataset"""

    def __init__(self, data_path: str = ''):
        """
        Initialize Syn-3bfo oracle

        Args:
            data_path: Path to the fitness.csv file
        """
        super().__init__("Syn-3bfo")
        if data_path == '':
            # Expect dataset copied under this repo: <project_root>/data/Syn-3bfo/fitness.csv
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            # → /.../sde-harness
            data_path = os.path.abspath(
                os.path.join(project_root, "proteinoptimizer/data/Syn-3bfo/fitness.csv")
            )

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")

        # Load fitness CSV (ground-truth experimental scores)
        self.data = pd.read_csv(data_path)
        self.lookup = self.data.set_index('Combo')['fitness'].to_dict()
        self.ALPHABET=ALPHABET
        # Try to load Potts landscape (if files & potts_model available)
        self._potts_landscape = None
        if potts_model is not None:
            # Expected file name pattern: {last4}_1_A_model_state_dict.npz, e.g. 3bfo_1_A_model_state_dict.npz
            tag = self.property_name[-4:] if len(self.property_name) >= 4 else ""
            potts_file = os.path.join(
                os.path.dirname(data_path), f"{tag}_1_A_model_state_dict.npz"
            )
            if os.path.exists(potts_file):
                try:
                    self._potts_landscape = potts_model.load_from_mogwai_npz(
                        potts_file, coupling_scale=1.0
                    )
                except Exception:
                    # Fail silently – fall back to lookup table
                    self._potts_landscape = None

    def _evaluate_protein_impl(self, sequence: str) -> float:
        """Evaluate a single protein sequence.

        Priority:
            1. Potts model (if loaded)
            2. Experimental lookup table
            3. Default 0.0
        """
        # Use Potts model if available
        if self._potts_landscape is not None:
            try:
                seq_numeric = [AA_TO_NUM.get(aa, 20) for aa in sequence]
                import numpy as np  # local import to avoid top-level dependency if unnecessary
                return float(self._potts_landscape.evaluate(np.array(seq_numeric)))
            except Exception:
                pass  # fall back to lookup

        # Fallback to lookup table
        return float(self.lookup.get(sequence, 0.0))

    def get_initial_population(self, size: int) -> list[str]:
        """Get an initial population from the dataset"""
        return self.data.sample(n=size)['Combo'].tolist()

    def list_single_round_metrics(self) -> list[str]:
        """Expose the primary metric of this oracle."""
        return ["fitness"] 