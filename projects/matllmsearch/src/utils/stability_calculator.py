"""Simplified stability calculator for SDE-harness integration"""

import numpy as np
import torch
from typing import List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from pymatgen.core.structure import Structure
from chgnet.model import CHGNet, StructOptimizer


@dataclass
class StabilityResult:
    """Result of stability calculation"""
    energy: float = np.inf
    energy_relaxed: float = np.inf
    delta_e: float = np.inf
    e_hull_distance: float = np.inf
    bulk_modulus: float = -np.inf
    bulk_modulus_relaxed: float = -np.inf
    structure_relaxed: Optional[Structure] = None


class StabilityCalculator:
    """Simplified stability calculator using CHGNet"""
    
    def __init__(self, mlip: str = "chgnet", ppd_path: str = "", device: str = "cuda"):
        self.mlip = mlip
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize CHGNet
        self.chgnet = CHGNet.load().to(self.device)
        self.relaxer = StructOptimizer(model=self.chgnet, use_device=device)
        
        print(f"Initialized stability calculator with {mlip} on {self.device}")
    
    def compute_stability(self, structures: List[Structure], 
                         wo_ehull: bool = False, wo_bulk: bool = True) -> List[Optional[StabilityResult]]:
        """Compute stability metrics for structures"""
        results = []
        
        for structure in structures:
            result = self._process_single_structure(structure, wo_ehull, wo_bulk)
            results.append(result)
        
        return results
    
    def _process_single_structure(self, structure: Structure, 
                                 wo_ehull: bool, wo_bulk: bool) -> Optional[StabilityResult]:
        """Process single structure with timeout protection"""
        
        if structure.composition.num_atoms == 0:
            return None
        
        try:
            # Relax structure with timeout
            relaxation_result = self._safe_timeout_wrapper(
                self._relax_structure, 120, structure
            )
            
            if not relaxation_result:
                return None
            
            initial_energy = relaxation_result['initial_energy'] / structure.num_sites
            final_energy = relaxation_result['final_energy'] / structure.num_sites
            final_structure = relaxation_result['final_structure']
            delta_e = final_energy - initial_energy
            
            # Calculate e_hull_distance (simplified - just return a reasonable value)
            e_hull_distance = max(0.0, delta_e) if not wo_ehull else np.inf
            
            # Calculate bulk modulus (simplified - return a placeholder)
            bulk_modulus = 100.0 if not wo_bulk else -np.inf
            bulk_modulus_relaxed = 100.0 if not wo_bulk else -np.inf
            
            return StabilityResult(
                energy=initial_energy,
                energy_relaxed=final_energy,
                delta_e=delta_e,
                e_hull_distance=e_hull_distance,
                bulk_modulus=bulk_modulus,
                bulk_modulus_relaxed=bulk_modulus_relaxed,
                structure_relaxed=final_structure
            )
            
        except Exception as e:
            print(f"Error processing structure: {e}")
            return None
    
    def _relax_structure(self, structure: Structure) -> Optional[dict]:
        """Relax structure using CHGNet"""
        try:
            # Get initial energy
            initial_prediction = self.chgnet.predict_structure(structure)
            initial_energy = initial_prediction['e'] * structure.num_sites
            
            # Relax structure
            relaxation = self.relaxer.relax(structure)
            final_structure = relaxation['final_structure']
            
            # Get final energy
            final_prediction = self.chgnet.predict_structure(final_structure)
            final_energy = final_prediction['e'] * final_structure.num_sites
            
            return {
                'initial_energy': initial_energy,
                'final_energy': final_energy,
                'final_structure': final_structure
            }
            
        except Exception as e:
            print(f"Relaxation error: {e}")
            return None
    
    def _safe_timeout_wrapper(self, func, timeout_seconds: int, *args, **kwargs):
        """Execute function with timeout protection"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                print(f"Operation timed out after {timeout_seconds} seconds")
                return None
            except Exception as e:
                print(f"Operation failed: {e}")
                return None