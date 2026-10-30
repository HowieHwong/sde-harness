"""Dataset loader for LLM-SRBench equation discovery tasks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datasets
import h5py
import numpy as np
from huggingface_hub import snapshot_download


# Output-variable descriptions used in the prompts of the LLM-SRBench /
# SDE-Harness paper runs. They differ slightly from the HuggingFace metadata
# (e.g. "Acceleration in Nonl-linear Harmonic Oscillator"); keep them by default
# so prompts are identical to the reported experiments.
PAPER_OUTPUT_DESCRIPTIONS = {
    "phys_osc": "Acceleration in non-linear harmonic oscillator",
    "bio_pop_growth": "Population growth rate in biology",
}


@dataclass
class EquationData:
    """Data class for equation discovery problems.

    Data arrays have shape (n_points, n_symbols) with the output in column 0
    followed by the inputs in the order of `symbols[1:]`.
    """
    name: str
    symbols: List[str]
    symbol_descs: List[str]
    symbol_properties: List[str]
    expression: str
    train_data: np.ndarray
    test_data: Optional[np.ndarray] = None
    ood_test_data: Optional[np.ndarray] = None


class LLMSRDatasetLoader:
    """Loader for LLM-SRBench datasets from HuggingFace (`nnheui/llm-srbench`)."""

    def __init__(self, dataset_name: str = "lsrtransform", use_paper_descriptions: bool = True):
        """
        Args:
            dataset_name: One of lsrtransform, bio_pop_growth, chem_react, matsci, phys_osc.
            use_paper_descriptions: Use the output descriptions of the paper runs.
        """
        self.dataset_name = dataset_name
        self.use_paper_descriptions = use_paper_descriptions
        self.repo_id = "nnheui/llm-srbench"
        self._dataset_dir = None
        self._problems: List[EquationData] = []

    def setup(self):
        """Download and load the dataset."""
        print(f"Setting up dataset: {self.dataset_name}")
        self._dataset_dir = Path(snapshot_download(repo_id=self.repo_id, repo_type="dataset"))

        if self.dataset_name == "lsrtransform":
            self._load("lsr_transform", "/lsr_transform")
        elif self.dataset_name in ["bio_pop_growth", "chem_react", "matsci", "phys_osc"]:
            self._load(f"lsr_synth_{self.dataset_name}", f"/lsr_synth/{self.dataset_name}")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        print(f"Loaded {len(self._problems)} problems from {self.dataset_name}")

    def _load(self, split: str, h5_group: str):
        ds = datasets.load_dataset(self.repo_id)[split]
        sample_h5file_path = self._dataset_dir / "lsr_bench_data.hdf5"

        with h5py.File(sample_h5file_path, "r") as sample_file:
            for e in ds:
                samples = {
                    k: v[...].astype(np.float64)
                    for k, v in sample_file[f'{h5_group}/{e["name"]}'].items()
                }
                symbol_descs = list(e['symbol_descs'])
                if self.use_paper_descriptions and self.dataset_name in PAPER_OUTPUT_DESCRIPTIONS:
                    symbol_descs[0] = PAPER_OUTPUT_DESCRIPTIONS[self.dataset_name]

                self._problems.append(EquationData(
                    name=e['name'],
                    symbols=list(e['symbols']),
                    symbol_descs=symbol_descs,
                    symbol_properties=list(e['symbol_properties']),
                    expression=e['expression'],
                    train_data=samples.get('train', samples.get('train_data')),
                    test_data=samples.get('test', samples.get('id_test_data')),
                    ood_test_data=samples.get('ood_test', samples.get('ood_test_data')),
                ))

    @property
    def problems(self) -> List[EquationData]:
        """Get all problems in the dataset."""
        return self._problems

    def get_problem_by_name(self, name: str) -> Optional[EquationData]:
        """Get a specific problem by name."""
        for problem in self._problems:
            if problem.name == name:
                return problem
        return None

    @staticmethod
    def get_variables(problem: EquationData) -> Tuple[List[str], List[str]]:
        """Return (var_names, var_descs): output first, then the input variables ("V")."""
        var_names = [problem.symbols[0]]
        var_descs = [problem.symbol_descs[0]]
        for symbol, desc, prop in zip(problem.symbols[1:], problem.symbol_descs[1:],
                                      problem.symbol_properties[1:]):
            if "V" in prop:
                var_names.append(symbol)
                var_descs.append(desc)
        var_names = [name.strip("$").strip("\\").replace(" ", "_").replace("text", "")
                     for name in var_names]
        return var_names, var_descs

    @staticmethod
    def to_inputs_outputs(data: np.ndarray) -> Dict[str, Any]:
        """Convert an (n_points, n_symbols) array to the {'inputs', 'outputs'} dict."""
        return {
            "inputs": [data[:, i] for i in range(1, data.shape[1])],
            "outputs": data[:, 0],
        }

    def get_problem_data(self, problem: EquationData) -> Dict[str, Any]:
        """Training data and variable metadata of a problem."""
        var_names, var_descs = self.get_variables(problem)
        return {
            **self.to_inputs_outputs(problem.train_data),
            'var_names': var_names,
            'var_descs': var_descs,
            'problem_name': problem.name,
            'expression': problem.expression,
        }
