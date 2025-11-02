from __future__ import annotations

import os
from typing import List
import pandas as pd

from src.oracles.base import ProteinOracle
from src.utils.potts_model import PottsModel, load_from_mogwai_npz
import numpy as np

ALPHABET = "ARNDCQEGHILKMFPSTWYV-"
A2N = {a: n for n, a in enumerate(ALPHABET)}
A2N["X"] = 20

def min_max_normalize(minimum,maximum,score):
    width = maximum-minimum
    normalize_score = (score-minimum)/width
    return float(normalize_score)

class FitnessOracle(ProteinOracle):
    """
    An oracle that reads fitness scores from a CSV file.
    """
    def __init__(self, name: str, data_path: str, potts_model_path: str = None):
        super().__init__()
        self.name = name
        
        # Resolve data path relative to this file's location
        if not os.path.isabs(data_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, data_path)
            if potts_model_path:
                potts_model_path = os.path.join(base_dir, potts_model_path)
        self.data = pd.read_csv(data_path).sort_values(by='fitness')
        self.lookup = self.data.set_index("Combo")["fitness"].to_dict()
        
        self.score_max = self.data['fitness'].max()
        self.score_min = self.data['fitness'].min()

        self._potts_landscape: PottsModel | None = None
        # print(potts_model_path, data_path)
        if potts_model_path and os.path.exists(potts_model_path):
             if not os.path.isabs(potts_model_path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                potts_model_path = os.path.join(base_dir, potts_model_path)
             self._potts_landscape = load_from_mogwai_npz(potts_model_path)

    def get_initial_population(self, size: int) -> List[str]:
        return np.random.choice(self.data["Combo"].tolist()[:int(0.6*len(self.data["Combo"]))], size)
        # return self.data.sample(size)["Combo"].tolist()

    def _evaluate_protein_impl(self, sequence: str) -> float:
        if self._potts_landscape is not None:
            # print(sequence)
            sequencns_lst = np.array([A2N[i] for i in sequence])
            return min_max_normalize(-3, 3, self._potts_landscape.evaluate(sequencns_lst)[0])
        return min_max_normalize(self.score_min, self.score_max, float(self.lookup.get(sequence, 0.0)))

    def list_single_round_metrics(self) -> list[str]:
        return ["fitness"]

class GB1Oracle(FitnessOracle):
    def __init__(self):
        super().__init__(
            name="gb1",
            data_path="../../data/GB1/fitness.csv",
        )

class TrpBOracle(FitnessOracle):
    def __init__(self):
        super().__init__(
            name="trpb",
            data_path="../../data/TrpB/fitness.csv",
        )

class Syn3bfoOracle(FitnessOracle):
    def __init__(self):
        super().__init__(
            name="syn-3bfo",
            data_path="../../data/Syn-3bfo/fitness.csv",
            potts_model_path="../../data/Syn-3bfo/3bfo_1_A_model_state_dict.npz",
        ) 