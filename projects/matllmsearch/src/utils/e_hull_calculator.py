"""Calculate energy to the hull for crystal structures"""

from __future__ import annotations

from typing import List, Tuple, Union, Dict, Any
import pickle
import gzip
from tqdm import tqdm

import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram


def _install_composition_compat_shim() -> None:
    """Rebuild cached fields missing on Compositions unpickled from an older
    pymatgen. The bundled phase diagram was serialized before ``num_atoms``
    was cached in ``_n_atoms``; without a ``__setstate__`` those objects raise
    AttributeError on every ``get_e_above_hull`` call. Recompute on access."""
    if getattr(Composition, "_ehull_compat_shim", False):
        return

    original_getattr = getattr(Composition, "__getattr__", None)

    def __getattr__(self, name):
        if name == "_n_atoms":
            n_atoms = float(sum(abs(v) for v in self._data.values()))
            object.__setattr__(self, "_n_atoms", n_atoms)
            return n_atoms
        if original_getattr is not None:
            return original_getattr(self, name)
        raise AttributeError(name)

    Composition.__getattr__ = __getattr__
    Composition._ehull_compat_shim = True


class EHullCalculator:
    """Calculate energy to the hull using patched phase diagram"""
    
    def __init__(self, ppd_path: str) -> None:
        """Initialize with path to patched phase diagram""" 
        print("Initialize EHullCalcul with patched phase diagram.")
        self.ppd_mp = self.load_gzip(ppd_path)
        
    def __call__(
        self,
        se_list: List[dict],
    ) -> List[dict]:
        """Get energy to the hull from list of dict containing structures
        and energy."""
        return self.get_e_hull(se_list)
        
    def get_e_hull(self, se_list: List[dict]) -> List[dict]:
        """Get energy to the hull from list of structures and energies"""
        entries = self.build_up_entry(se_list)
        e_hull = self.compute_e_hull(entries, self.ppd_mp)

        # Add e_hull to se_list.
        seh_list = []
        for i, se_dict in enumerate(se_list):
            se_dict['e_hull'] = e_hull[i]
            seh_list.append(se_dict)
            
        # Sort seh_list by e_hull.
        seh_list = sorted(seh_list, key=lambda k: k['e_hull'])
        return seh_list
    
    @staticmethod
    def build_up_entry(se_list: List[dict]) -> List[ComputedStructureEntry]:
        """Build list of computed structure entries"""    
        entries = []
        for se_dict in se_list:
            structure = se_dict.get('structure')
            energy = se_dict.get('energy')
            entry = ComputedStructureEntry(structure=structure, energy=energy)
            entries.append(entry)
        return entries
    
    @staticmethod
    def compute_e_hull(entries: List[ComputedStructureEntry], ppd: PatchedPhaseDiagram) -> Dict[str, float]:
        """Compute energy to the hull for each entry"""
        if not isinstance(entries, list):
            raise TypeError(f'entries must be a list, but got {type(entries)}')
        if not isinstance(ppd, PatchedPhaseDiagram):
            raise TypeError(f'ppd must be a PatchedPhaseDiagram, but got {type(ppd)}')
        e_hull = []
        for entry in tqdm(entries):
            e_hull.append(ppd.get_e_above_hull(entry, allow_negative = True))
        return e_hull
        
    @staticmethod
    def load_gzip(gzip_path: str) -> Any:
        """Load gzip file (PatchedPhaseDiagram)."""
        if not isinstance(gzip_path, str):
            raise TypeError(f'gzip_path must be a string, but got {type(gzip_path)}')
        _install_composition_compat_shim()
        with gzip.open(gzip_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def __repr__(self):
        return f'EHullCalculator(phase_diagram={self.ppd_mp})'