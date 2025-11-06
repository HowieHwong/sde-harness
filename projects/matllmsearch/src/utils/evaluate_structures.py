#!/usr/bin/env python
# coding: utf-8
"""
Evaluation script for generated crystal structures.

This script evaluates a list of generated structures and computes:
- Diversity metrics (composition diversity, structural diversity)
- Novelty metrics (SUN score - Structures Unique and Novel)
- Success rate metrics (stability rate, validity rate)

Usage:
    python evaluate_structures.py --input structures.json --output results.csv
    python evaluate_structures.py --input structures.csv --format poscar --training-data training_structures.csv
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ..evaluators.materials_oracle import MaterialsOracle, MaterialsEvaluation
from .stability_calculator import StabilityCalculator


class StructureEvaluator:
    """Comprehensive evaluator for crystal structures"""
    
    def __init__(self, 
                 mlip: str = "chgnet",
                 ppd_path: str = "data/2023-02-07-ppd-mp.pkl.gz",
                 device: str = "cuda",
                 training_structures: Optional[List[Structure]] = None):
        """
        Initialize structure evaluator.
        
        Args:
            mlip: Machine learning interatomic potential to use
            ppd_path: Path to patched phase diagram for E-hull calculation
            device: Device for computation ('cuda' or 'cpu')
            training_structures: List of training structures for novelty calculation
        """
        self.oracle = MaterialsOracle(mlip=mlip, ppd_path=ppd_path, device=device)
        self.matcher = StructureMatcher()
        self.training_structures = training_structures or []
        
        # Build training structure lookup for novelty
        if self.training_structures:
            print(f"Building lookup for {len(self.training_structures)} training structures...")
            self.training_formulas = defaultdict(list)
            for struct in self.training_structures:
                try:
                    formula = struct.composition.reduced_formula
                    self.training_formulas[formula].append(struct)
                except:
                    continue
            print(f"Organized into {len(self.training_formulas)} unique formulas")
    
    def parse_structure(self, structure_data: Any, fmt: str = "auto") -> Optional[Structure]:
        """
        Parse structure from various formats.
        
        Args:
            structure_data: Structure data (string, dict, or Structure object)
            fmt: Format type ('poscar', 'cif', 'json', 'auto')
            
        Returns:
            Parsed Structure or None if parsing fails
        """
        if isinstance(structure_data, Structure):
            return structure_data
        
        if isinstance(structure_data, dict):
            # Try to find structure in dict
            for key in ['structure', 'poscar', 'cif', 'Structure', 'StructureRelaxed']:
                if key in structure_data:
                    return self.parse_structure(structure_data[key], fmt)
            # If dict contains structure directly (as JSON string)
            if 'structure' in structure_data:
                structure_str = structure_data['structure']
                if isinstance(structure_str, str):
                    try:
                        # Try parsing as JSON first
                        structure_dict = json.loads(structure_str)
                        return Structure.from_dict(structure_dict)
                    except:
                        # Try parsing as string format
                        for fmt_try in ['json', 'poscar', 'cif']:
                            try:
                                return Structure.from_str(structure_str, fmt=fmt_try)
                            except:
                                continue
        
        if isinstance(structure_data, str):
            if fmt == "auto":
                # Try different formats
                for fmt_try in ['json', 'poscar', 'cif']:
                    try:
                        return Structure.from_str(structure_data, fmt=fmt_try)
                    except:
                        continue
            else:
                try:
                    return Structure.from_str(structure_data, fmt=fmt)
                except:
                    pass
        
        return None
    
    def load_structures_from_file(self, file_path: str, fmt: str = "auto") -> List[Structure]:
        """
        Load structures from file.
        
        Args:
            file_path: Path to input file (CSV, JSON, or text file)
            fmt: Format type for structures in file
            
        Returns:
            List of parsed structures
        """
        file_path = Path(file_path)
        structures = []
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            # Try to find structure column (prefer StructureRelaxed, then Structure)
            struct_cols = []
            for preferred in ['StructureRelaxed', 'Structure', 'structure_relaxed', 'structure']:
                if preferred in df.columns:
                    struct_cols.append(preferred)
            # Also check for any column with 'structure' in name
            struct_cols.extend([col for col in df.columns 
                          if 'structure' in col.lower() and col not in struct_cols])
            
            if struct_cols:
                struct_col = struct_cols[0]
                print(f"Using column '{struct_col}' for structures")
                for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing structures"):
                    struct_str = row[struct_col]
                    if pd.isna(struct_str):
                        continue
                    struct = self.parse_structure(struct_str, fmt)
                    if struct:
                        structures.append(struct)
                    elif idx < 3:  # Show first few errors
                        print(f"Warning: Failed to parse structure at row {idx}")
            else:
                print("Warning: No structure column found in CSV")
                print(f"Available columns: {list(df.columns)}")
                # Try to parse entire row as structure
                for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing structures"):
                    struct = self.parse_structure(row.to_dict(), fmt)
                    if struct:
                        structures.append(struct)
        
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in tqdm(data, desc="Parsing structures"):
                    struct = self.parse_structure(item, fmt)
                    if struct:
                        structures.append(struct)
            elif isinstance(data, dict):
                # Try to find structures in dict
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            struct = self.parse_structure(item, fmt)
                            if struct:
                                structures.append(struct)
                    else:
                        struct = self.parse_structure(value, fmt)
                        if struct:
                            structures.append(struct)
        
        else:
            # Try to read as text file (POSCAR format)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    struct = self.parse_structure(content, fmt='poscar')
                    if struct:
                        structures.append(struct)
            except:
                pass
        
        print(f"Loaded {len(structures)} structures from {file_path}")
        return structures
    
    def _validate_structure(self, structure: Structure) -> bool:
        """
        Validate if structure is structurally valid.
        
        Args:
            structure: Structure to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if structure has valid composition
            if structure.composition.num_atoms <= 0:
                return False
            
            # Check if structure has reasonable volume
            if structure.volume <= 0 or structure.volume >= 30 * structure.composition.num_atoms:
                return False
            
            # Check if structure is 3D periodic
            if not structure.is_3d_periodic:
                return False
            
            # Check for valid lattice
            if structure.lattice.volume <= 0:
                return False
            
            return True
        except Exception:
            return False
    
    def _validate_composition(self, composition: Composition) -> bool:
        """
        Validate if composition is valid.
        
        Args:
            composition: Composition to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check for valid elements
            if len(composition.elements) == 0:
                return False
            
            # Check for reasonable stoichiometry (no negative or zero counts)
            if any(count <= 0 for count in composition.values()):
                return False
            
            # Check for reasonable number of elements (typically 1-10)
            if len(composition.elements) > 10:
                return False
            
            return True
        except Exception:
            return False
    
    def calculate_composition_diversity(self, structures: List[Structure]) -> Dict[str, Any]:
        """
        Calculate composition diversity metrics.
        
        Args:
            structures: List of structures
            
        Returns:
            Dictionary with diversity metrics
        """
        if not structures:
            return {
                'composition_diversity': 0.0,
                'unique_compositions': 0,
                'total_structures': 0,
                'composition_ratio': 0.0
            }
        
        compositions = [s.composition.reduced_formula for s in structures]
        unique_compositions = len(set(compositions))
        total = len(structures)
        
        return {
            'composition_diversity': unique_compositions / total if total > 0 else 0.0,
            'unique_compositions': unique_compositions,
            'total_structures': total,
            'composition_ratio': unique_compositions / total if total > 0 else 0.0,
            'composition_counts': dict(Counter(compositions))
        }
    
    def calculate_validity(self, structures: List[Structure]) -> Dict[str, Any]:
        """
        Calculate structural and composition validity metrics.
        
        Args:
            structures: List of structures
            
        Returns:
            Dictionary with validity metrics
        """
        if not structures:
            return {
                'structural_validity': 0.0,
                'composition_validity': 0.0,
                'valid_structures': 0,
                'valid_compositions': 0,
                'total_structures': 0
            }
        
        total = len(structures)
        struct_valid_count = 0
        comp_valid_count = 0
        
        for struct in structures:
            if self._validate_structure(struct):
                struct_valid_count += 1
            if self._validate_composition(struct.composition):
                comp_valid_count += 1
        
        return {
            'structural_validity': struct_valid_count / total if total > 0 else 0.0,
            'composition_validity': comp_valid_count / total if total > 0 else 0.0,
            'valid_structures': struct_valid_count,
            'valid_compositions': comp_valid_count,
            'total_structures': total
        }
    
    def calculate_structural_diversity(self, structures: List[Structure]) -> Dict[str, Any]:
        """
        Calculate structural diversity using StructureMatcher.
        
        Args:
            structures: List of structures
            
        Returns:
            Dictionary with structural diversity metrics
        """
        if len(structures) < 2:
            return {
                'structural_diversity': 0.0,
                'unique_structures': len(structures),
                'total_structures': len(structures)
            }
        
        unique_structures = []
        for i, struct in enumerate(tqdm(structures, desc="Calculating structural diversity")):
            is_unique = True
            for unique_struct in unique_structures:
                try:
                    if self.matcher.fit(struct, unique_struct):
                        is_unique = False
                        break
                except:
                    continue
            
            if is_unique:
                unique_structures.append(struct)
        
        total = len(structures)
        unique_count = len(unique_structures)
        
        return {
            'structural_diversity': unique_count / total if total > 0 else 0.0,
            'unique_structures': unique_count,
            'total_structures': total,
            'structural_ratio': unique_count / total if total > 0 else 0.0
        }
    
    def calculate_composition_novelty(self, structures: List[Structure]) -> Dict[str, Any]:
        """
        Calculate composition novelty - novel compositions compared to training data.
        
        Args:
            structures: List of generated structures
            
        Returns:
            Dictionary with composition novelty metrics
        """
        if not structures:
            return {
                'composition_novelty': 0.0,
                'novel_compositions': 0,
                'total_compositions': 0
            }
        
        if not self.training_structures:
            return {
                'composition_novelty': 0.0,
                'novel_compositions': 0,
                'total_compositions': len(structures),
                'note': 'No training data provided'
            }
        
        # Get unique compositions from generated structures
        generated_compositions = set()
        for struct in structures:
            try:
                generated_compositions.add(struct.composition.reduced_formula)
            except:
                continue
        
        # Get unique compositions from training data
        training_compositions = set()
        for struct in self.training_structures:
            try:
                training_compositions.add(struct.composition.reduced_formula)
            except:
                continue
        
        # Find novel compositions
        novel_compositions = generated_compositions - training_compositions
        
        total_compositions = len(generated_compositions)
        novelty_score = len(novel_compositions) / total_compositions if total_compositions > 0 else 0.0
        
        return {
            'composition_novelty': novelty_score,
            'novel_compositions': len(novel_compositions),
            'total_compositions': total_compositions,
            'training_compositions': len(training_compositions)
        }
    
    def calculate_structural_novelty(self, structures: List[Structure]) -> Dict[str, Any]:
        """
        Calculate structural novelty - novel structures (different structures even with same composition).
        
        Args:
            structures: List of generated structures
            
        Returns:
            Dictionary with structural novelty metrics
        """
        if not structures:
            return {
                'structural_novelty': 0.0,
                'novel_structures': 0,
                'total_structures': 0
            }
        
        if not self.training_structures:
            return {
                'structural_novelty': 0.0,
                'novel_structures': 0,
                'total_structures': len(structures),
                'note': 'No training data provided'
            }
        
        print("Calculating structural novelty...")
        novel_count = 0
        
        for i, struct in enumerate(tqdm(structures, desc="Checking structural novelty")):
            is_novel = True
            formula = struct.composition.reduced_formula
            
            # Check against training structures with same formula
            if formula in self.training_formulas:
                for train_struct in self.training_formulas[formula]:
                    try:
                        if self.matcher.fit(struct, train_struct):
                            is_novel = False
                            break
                    except:
                        continue
            
            if is_novel:
                novel_count += 1
        
        total = len(structures)
        novelty_score = novel_count / total if total > 0 else 0.0
        
        return {
            'structural_novelty': novelty_score,
            'novel_structures': novel_count,
            'total_structures': total
        }
    
    def calculate_sun_score_neurips(self, structures: List[Structure], evaluations: List[MaterialsEvaluation]) -> Dict[str, Any]:
        """
        Compute SUN score aligned with NeurIPS evaluation helper:
        1) Take stable subset (E-hull < 0.1 eV/atom) from evaluated structures
        2) Find unique structures within the stable subset
        3) Count how many of those unique stable structures are novel w.r.t. training/reference set
        4) SUN score = novel_unique_stable / total_generated (denominator = all generated structures)

        Returns dict with keys: sun_score, novel_unique_stable, unique_stable, total_generated
        """
        total_generated = len(structures)
        if not structures or not evaluations:
            return {
                'sun_score': 0.0,
                'novel_unique_stable': 0,
                'unique_stable': 0,
                'total_generated': total_generated,
            }

        # Build stable subset (use valid evals with e_hull_distance < 0.1)
        stable_structs: List[Structure] = []
        for struct, ev in zip(structures, evaluations):
            try:
                if ev and ev.valid and ev.e_hull_distance is not None and ev.e_hull_distance < 0.1:
                    stable_structs.append(struct)
            except Exception:
                continue

        if not stable_structs:
            return {
                'sun_score': 0.0,
                'novel_unique_stable': 0,
                'unique_stable': 0,
                'total_generated': total_generated,
            }

        # Step 1: find unique within stable set using Structure.matches like NeurIPS (per-formula grouping)
        unique_stable: List[Structure] = []
        for s in stable_structs:
            is_unique = True
            for u in unique_stable:
                try:
                    # Use pymatgen Structure.matches with scaling enabled, no supercells
                    if s.matches(u, scale=True, attempt_supercell=False):
                        is_unique = False
                        break
                except Exception:
                    continue
            if is_unique:
                unique_stable.append(s)

        # Step 2: compare unique stable to training/reference set
        novel_count = 0
        if self.training_structures:
            # Build quick lookup by reduced formula
            training_lookup = self.training_formulas if hasattr(self, 'training_formulas') and self.training_formulas else {}
            if not training_lookup:
                training_lookup = defaultdict(list)
                for ts in self.training_structures:
                    try:
                        training_lookup[ts.composition.reduced_formula].append(ts)
                    except Exception:
                        continue

            for s in unique_stable:
                is_novel = True
                formula = s.composition.reduced_formula
                if formula in training_lookup:
                    for ref in training_lookup[formula]:
                        try:
                            if s.matches(ref, scale=True, attempt_supercell=False):
                                is_novel = False
                                break
                        except Exception:
                            continue
                if is_novel:
                    novel_count += 1
        else:
            # If no training/reference set, treat all unique stable as novel per NeurIPS fallback
            novel_count = len(unique_stable)

        sun_score = novel_count / total_generated if total_generated > 0 else 0.0
        return {
            'sun_score': sun_score,
            'novel_unique_stable': novel_count,
            'unique_stable': len(unique_stable),
            'total_generated': total_generated,
        }

    def calculate_overall_novelty(self, structures: List[Structure]) -> Dict[str, Any]:
        """
        Calculate overall novelty (both composition-novel AND structural-novel) WITHOUT stability requirement.
        
        Overall novelty = (structures that are composition-novel AND structural-novel) / (total generated structures)
        
        No stability filter is applied. Denominator is total structures (all generated structures).

        Returns dict with 'overall_novelty', 'both_novel_count', and 'total_structures'.
        """
        total = len(structures)
        if total == 0:
            return {
                'overall_novelty': 0.0,
                'both_novel_count': 0,
                'total_structures': 0,
            }

        both_novel_count = 0
        # If no training data, nothing can be non-novel; treat as zero to avoid inflating
        if not self.training_structures:
            return {
                'overall_novelty': 0.0,
                'both_novel_count': 0,
                'total_structures': total,
            }

        # Pre-built training_formulas is expected
        for struct in structures:
            try:
                formula = struct.composition.reduced_formula
            except Exception:
                continue

            # Composition novel check
            comp_is_novel = formula not in self.training_formulas

            if not comp_is_novel:
                # If composition exists in reference, cannot be both-novel
                continue

            # Structural novel check
            struct_is_novel = True
            refs = self.training_formulas.get(formula, [])
            for ref in refs:
                try:
                    if struct.matches(ref, scale=True, attempt_supercell=False):
                        struct_is_novel = False
                        break
                except Exception:
                    continue

            if struct_is_novel:
                both_novel_count += 1

        return {
            'overall_novelty': both_novel_count / total if total > 0 else 0.0,
            'both_novel_count': both_novel_count,
            'total_structures': total,
        }

    def calculate_sun(self, structures: List[Structure], 
                     evaluations: Optional[List[MaterialsEvaluation]] = None) -> Dict[str, Any]:
        """
        Calculate SUN (Structures Unique and Novel) rate.
        
        SUN rate = (structures that are stable (E-hull < 0.0) AND composition-novel AND structural-novel) / (total generated structures)
        
        Requires evaluations to be provided. Only counts structures with E-hull distance < 0.0 (stable).
        Denominator is always total structures (all generated structures).

        Returns dict with 'sun_score', 'both_novel_count', and 'total_structures'.
        """
        total = len(structures)
        if total == 0:
            return {
                'sun_score': 0.0,
                'both_novel_count': 0,
                'total_structures': 0,
            }

        both_novel_count = 0
        # If no training data, nothing can be non-novel; treat as zero to avoid inflating
        if not self.training_structures:
            return {
                'sun_score': 0.0,
                'both_novel_count': 0,
                'total_structures': total,
            }

        # Filter by stability - REQUIRED for SUN score
        if evaluations is None or len(evaluations) != len(structures):
            # Without evaluations, cannot calculate SUN (requires stability info)
            return {
                'sun_score': 0.0,
                'both_novel_count': 0,
                'total_structures': total,
            }

        # Only consider structures with E-hull < 0.0 (stable)
        stable_indices = []
        for i, (struct, ev) in enumerate(zip(structures, evaluations)):
            try:
                if ev and ev.valid and ev.e_hull_distance is not None:
                    if not (np.isnan(ev.e_hull_distance) or np.isinf(ev.e_hull_distance)):
                        if ev.e_hull_distance < 0.0:
                            stable_indices.append(i)
            except Exception:
                continue
        
        # Only check stable structures for novelty
        structures_to_check = [structures[i] for i in stable_indices]

        # Pre-built training_formulas is expected
        for struct in structures_to_check:
            try:
                formula = struct.composition.reduced_formula
            except Exception:
                continue

            # Composition novel check
            comp_is_novel = formula not in self.training_formulas

            if not comp_is_novel:
                # If composition exists in reference, cannot be both-novel
                continue

            # Structural novel check
            # reference set for the same formula; if formula absent, treat as novel structurally
            struct_is_novel = True
            refs = self.training_formulas.get(formula, [])
            for ref in refs:
                try:
                    if struct.matches(ref, scale=True, attempt_supercell=False):
                        struct_is_novel = False
                        break
                except Exception:
                    continue

            if struct_is_novel:
                both_novel_count += 1

        return {
            'sun_score': both_novel_count / total if total > 0 else 0.0,
            'both_novel_count': both_novel_count,
            'total_structures': total,
        }

    def calculate_novelty(self, structures: List[Structure]) -> Dict[str, Any]:
        """
        Calculate overall novelty (SUN score) - Structures Unique and Novel.
        
        SUN score = (structures unique in generated set AND not in training set) / (total generated structures)
        
        Also computes composition and structural novelty separately.
        
        Args:
            structures: List of generated structures
            
        Returns:
            Dictionary with novelty metrics
        """
        if not structures:
            return {
                'sun_score': 0.0,
                'novel_structures': 0,
                'unique_structures': 0,
                'total_structures': 0
            }
        
        print(f"Calculating novelty for {len(structures)} structures...")
        
        # Step 1: Find unique structures within generated set
        print("Step 1: Finding unique structures within generated set...")
        unique_structures = []
        for i, struct in enumerate(tqdm(structures, desc="Finding unique structures")):
            is_unique = True
            for unique_struct in unique_structures:
                try:
                    if self.matcher.fit(struct, unique_struct):
                        is_unique = False
                        break
                except:
                    continue
            
            if is_unique:
                unique_structures.append(struct)
        
        print(f"Found {len(unique_structures)} unique structures out of {len(structures)}")
        
        # Step 2: Compare with training set
        if not self.training_structures:
            print("No training structures provided, skipping novelty comparison")
            return {
                'novel_structures': len(unique_structures),
                'unique_structures': len(unique_structures),
                'total_structures': len(structures),
                'novelty_without_training': True
            }
        
        print(f"Step 2: Comparing {len(unique_structures)} unique structures with {len(self.training_structures)} training structures...")
        novel_structures = []
        
        for struct in tqdm(unique_structures, desc="Checking novelty"):
            is_novel = True
            formula = struct.composition.reduced_formula
            
            # Only check against structures with same formula
            if formula in self.training_formulas:
                for train_struct in self.training_formulas[formula]:
                    try:
                        if self.matcher.fit(struct, train_struct):
                            is_novel = False
                            break
                    except:
                        continue
            
            if is_novel:
                novel_structures.append(struct)
        
        print(f"Found {len(novel_structures)} novel structures out of {len(unique_structures)} unique structures")
        
        # Calculate composition and structural novelty separately
        comp_novelty = self.calculate_composition_novelty(structures)
        struct_novelty = self.calculate_structural_novelty(structures)
        
        return {
            'novel_structures': len(novel_structures),
            'unique_structures': len(unique_structures),
            'total_structures': len(structures),
            'novelty_ratio': len(novel_structures) / len(unique_structures) if unique_structures else 0.0,
            'composition_novelty': comp_novelty,
            'structural_novelty': struct_novelty
        }
    
    def calculate_success_rate(self, evaluations: List[MaterialsEvaluation]) -> Dict[str, Any]:
        """
        Calculate success rate metrics based on stability and validity.
        Includes M3GNet metastability (E-hull < 0.1 eV/atom).
        
        Args:
            evaluations: List of MaterialsEvaluation objects
            
        Returns:
            Dictionary with success rate metrics
        """
        if not evaluations:
            return {
                'validity_rate': 0.0,
                'metastability_0': 0.0,
                'metastability_0.03': 0.0,
                'metastability_0.10': 0.0,
                'm3gnet_metastability': 0.0,
                'stability_rate_0.03': 0.0,  # Keep for backward compatibility
                'stability_rate_0.10': 0.0,  # Keep for backward compatibility
                'success_rate': 0.0,
                'total_structures': 0
            }
        
        total = len(evaluations)
        valid_count = sum(1 for eval in evaluations if eval.valid)
        valid_evals = [eval for eval in evaluations if eval.valid]
        
        # Stability rates (metastability thresholds)
        stable_0 = sum(1 for eval in valid_evals 
                      if not (np.isnan(eval.e_hull_distance) or np.isinf(eval.e_hull_distance))
                      and eval.e_hull_distance < 0.0)
        stable_003 = sum(1 for eval in valid_evals 
                        if not (np.isnan(eval.e_hull_distance) or np.isinf(eval.e_hull_distance))
                        and eval.e_hull_distance < 0.03)
        stable_01 = sum(1 for eval in valid_evals 
                       if not (np.isnan(eval.e_hull_distance) or np.isinf(eval.e_hull_distance))
                       and eval.e_hull_distance < 0.10)
        
        # Metastability rates
        metastability_0 = stable_0 / total if total > 0 else 0.0
        metastability_003 = stable_003 / total if total > 0 else 0.0
        metastability_01 = stable_01 / total if total > 0 else 0.0
        # M3GNet metastability: E-hull < 0.1 eV/atom (same as metastability_01)
        m3gnet_metastability = metastability_01
        
        # Success rate: valid AND stable (< 0.1 eV/atom)
        success_count = sum(1 for eval in valid_evals 
                           if not (np.isnan(eval.e_hull_distance) or np.isinf(eval.e_hull_distance))
                           and eval.e_hull_distance <= 0.10)
        
        return {
            'validity_rate': valid_count / total if total > 0 else 0.0,
            'valid_structures': valid_count,
            'metastability_0': metastability_0,
            'metastability_0.03': metastability_003,
            'metastability_0.10': metastability_01,
            'm3gnet_metastability': m3gnet_metastability,
            'stable_structures_0': stable_0,
            'stable_structures_0.03': stable_003,
            'stable_structures_0.10': stable_01,
            'stability_rate_0.03': metastability_003,  # Keep for backward compatibility
            'stability_rate_0.10': metastability_01,  # Keep for backward compatibility
            'success_rate': success_count / total if total > 0 else 0.0,
            'success_structures': success_count,
            'total_structures': total
        }
    
    def evaluate(self, structures: List[Structure], 
                 calculate_stability: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of structures.
        
        Args:
            structures: List of structures to evaluate
            calculate_stability: Whether to calculate stability metrics (can be slow)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print(f"Evaluating {len(structures)} structures...")
        
        results = {
            'total_structures': len(structures)
        }
        
        # Validity metrics
        print("\n" + "="*60)
        print("Calculating validity metrics...")
        print("="*60)
        validity = self.calculate_validity(structures)
        results['structural_validity'] = validity['structural_validity']
        results['composition_validity'] = validity['composition_validity']
        results['validity'] = validity
        
        # Composition diversity
        print("\n" + "="*60)
        print("Calculating composition diversity...")
        print("="*60)
        comp_div = self.calculate_composition_diversity(structures)
        results['composition_diversity'] = comp_div['composition_diversity']
        results['composition_diversity_details'] = comp_div
        
        # Structural diversity
        print("\n" + "="*60)
        print("Calculating structural diversity...")
        print("="*60)
        struct_div = self.calculate_structural_diversity(structures)
        results['structural_diversity'] = struct_div['structural_diversity']
        results['structural_diversity_details'] = struct_div
        
        # Stability and success rate (if requested)
        if calculate_stability:
            print("\n" + "="*60)
            print("Calculating stability metrics (this may take a while)...")
            print("="*60)
            evaluations = self.oracle.evaluate(structures)
            success_metrics = self.calculate_success_rate(evaluations)
            results['success_rate'] = success_metrics
            
            # Add detailed stability metrics
            valid_evals = [e for e in evaluations if e.valid]
            if valid_evals:
                e_hull_distances = [e.e_hull_distance for e in valid_evals 
                                  if not (np.isnan(e.e_hull_distance) or np.isinf(e.e_hull_distance))]
                if e_hull_distances:
                    results['stability_stats'] = {
                        'min_e_hull': min(e_hull_distances),
                        'max_e_hull': max(e_hull_distances),
                        'mean_e_hull': np.mean(e_hull_distances),
                        'median_e_hull': np.median(e_hull_distances),
                        'std_e_hull': np.std(e_hull_distances)
                    }
            
            # Add M3GNet metastability to results
            results['m3gnet_metastability'] = success_metrics['m3gnet_metastability']

            # Compute composition/structural novelty, overall novelty, and SUN rate
            novelty = self.calculate_novelty(structures)
            overall_novelty_result = self.calculate_overall_novelty(structures)  # No stability requirement
            sun_result = self.calculate_sun(structures, evaluations=evaluations)  # With stability requirement (E-hull < 0.0)
            results['overall_novelty'] = overall_novelty_result['overall_novelty']
            results['novelty'] = {
                'sun_score': sun_result['sun_score'],
                'both_novel_count': overall_novelty_result['both_novel_count'],
                'sun_both_novel_count': sun_result['both_novel_count'],
                'total_structures': overall_novelty_result['total_structures'],
                'composition_novelty': novelty['composition_novelty'],
                'structural_novelty': novelty['structural_novelty'],
            }
            results['composition_novelty'] = novelty['composition_novelty']['composition_novelty']
            results['structural_novelty'] = novelty['structural_novelty']['structural_novelty']
        else:
            results['success_rate'] = {
                'note': 'Stability calculation skipped',
                'total_structures': len(structures)
            }

            # Without stability, compute overall novelty only (SUN requires stability)
            novelty = self.calculate_novelty(structures)
            overall_novelty_result = self.calculate_overall_novelty(structures)  # No stability requirement
            results['overall_novelty'] = overall_novelty_result['overall_novelty']
            results['novelty'] = {
                'sun_score': 0.0,  # Cannot calculate SUN without stability
                'both_novel_count': overall_novelty_result['both_novel_count'],
                'total_structures': overall_novelty_result['total_structures'],
                'composition_novelty': novelty['composition_novelty'],
                'structural_novelty': novelty['structural_novelty'],
            }
            results['composition_novelty'] = novelty['composition_novelty']['composition_novelty']
            results['structural_novelty'] = novelty['structural_novelty']['structural_novelty']
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate generated crystal structures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate structures from JSON file
  python evaluate_structures.py --input structures.json --output results.json

  # Evaluate structures from CSV with POSCAR format
  python evaluate_structures.py --input structures.csv --format poscar --output results.csv

  # Evaluate with training data for novelty calculation
  python evaluate_structures.py --input structures.json --training-data training.json --output results.json
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input file containing structures (CSV, JSON, or text file)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file for results (JSON or CSV)')
    parser.add_argument('--format', type=str, default='auto',
                       choices=['auto', 'poscar', 'cif', 'json'],
                       help='Structure format (default: auto-detect)')
    parser.add_argument('--training-data', type=str, default=None,
                       help='File containing training structures for novelty calculation')
    parser.add_argument('--mlip', type=str, default='chgnet',
                       help='Machine learning interatomic potential (default: chgnet)')
    parser.add_argument('--ppd-path', type=str, default='data/2023-02-07-ppd-mp.pkl.gz',
                       help='Path to patched phase diagram file')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for computation (default: cuda)')
    parser.add_argument('--no-stability', action='store_true',
                       help='Skip stability calculation for faster evaluation')
    
    args = parser.parse_args()
    
    # Load structures
    print("Loading structures...")
    evaluator = StructureEvaluator(
        mlip=args.mlip,
        ppd_path=args.ppd_path,
        device=args.device
    )
    
    structures = evaluator.load_structures_from_file(args.input, fmt=args.format)
    
    if not structures:
        print(f"Error: No valid structures found in {args.input}")
        sys.exit(1)
    
    # Load training structures if provided
    if args.training_data:
        print(f"Loading training structures from {args.training_data}...")
        training_structures = evaluator.load_structures_from_file(args.training_data, fmt=args.format)
        evaluator.training_structures = training_structures
        evaluator.training_formulas = defaultdict(list)
        for struct in training_structures:
            try:
                formula = struct.composition.reduced_formula
                evaluator.training_formulas[formula].append(struct)
            except:
                continue
        print(f"Loaded {len(training_structures)} training structures")
    
    # Evaluate
    results = evaluator.evaluate(structures, calculate_stability=not args.no_stability)
    
    # Save results
    output_path = Path(args.output)
    if output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                return obj
            
            json.dump(convert_to_serializable(results), f, indent=2)
    else:
        # Save as CSV (flatten results)
        flat_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_results[f"{key}_{subkey}"] = subvalue
            else:
                flat_results[key] = value
        
        df = pd.DataFrame([flat_results])
        df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total structures: {results['total_structures']}")
    print(f"  Composition diversity: {results['composition_diversity']['composition_diversity']:.4f}")
    print(f"  Structural diversity: {results['structural_diversity']['structural_diversity']:.4f}")
    print(f"  SUN score (novelty): {results['novelty']['sun_score']:.4f}")
    if 'success_rate' in results and 'validity_rate' in results['success_rate']:
        print(f"  Validity rate: {results['success_rate']['validity_rate']:.4f}")
        print(f"  Success rate (<0.1 eV): {results['success_rate']['success_rate']:.4f}")
        print(f"  Stability rate (<0.03 eV): {results['success_rate']['stability_rate_0.03']:.4f}")
        print(f"  Stability rate (<0.10 eV): {results['success_rate']['stability_rate_0.10']:.4f}")


if __name__ == "__main__":
    main()

