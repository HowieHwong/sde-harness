#!/usr/bin/env python
# coding: utf-8
"""
Evaluation script for generated crystal structures.

Computes diversity, novelty (SUN score), and success rate metrics.
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .materials_oracle import MaterialsOracle, MaterialsEvaluation
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
        
        # Build training structure lookup for novelty if provided
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
            for key in ['structure', 'poscar', 'cif', 'Structure', 'StructureRelaxed']:
                if key in structure_data:
                    return self.parse_structure(structure_data[key], fmt)
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
            struct_cols = []
            for preferred in ['StructureRelaxed', 'Structure', 'structure_relaxed', 'structure']:
                if preferred in df.columns:
                    struct_cols.append(preferred)
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
                    elif idx < 3:
                        print(f"Warning: Failed to parse structure at row {idx}")
            else:
                print("Warning: No structure column found in CSV")
                print(f"Available columns: {list(df.columns)}")
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
        """Validate if structure is structurally valid"""
        try:
            if structure.composition.num_atoms <= 0:
                return False
            
            if structure.volume <= 0 or structure.volume >= 30 * structure.composition.num_atoms:
                return False
            
            if not structure.is_3d_periodic:
                return False
            
            if structure.lattice.volume <= 0:
                return False
            
            return True
        except Exception:
            return False
    
    def _validate_composition(self, composition: Composition) -> bool:
        """Validate if composition is valid"""
        try:
            if len(composition.elements) == 0:
                return False
            
            # Check for reasonable stoichiometry (no negative or zero counts)
            if any(count <= 0 for count in composition.values()):
                return False
            
            # Check for reasonable number of elements
            if len(composition.elements) > 10:
                return False
            
            return True
        except Exception:
            return False
    
    def calculate_composition_diversity(self, structures: List[Structure]) -> Dict[str, Any]:
        """Calculate composition diversity metrics"""
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
        """Calculate structural and composition validity metrics"""
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
        """Calculate structural diversity using StructureMatcher"""
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
        """Calculate composition novelty compared to training data"""
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
        """Calculate structural novelty (different structures even with same composition)"""
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
    
    def calculate_overall_novelty(self, structures: List[Structure]) -> Dict[str, Any]:
        """
        Calculate overall novelty (composition-novel AND structural-novel) without stability requirement.
        """
        total = len(structures)
        if total == 0:
            return {
                'overall_novelty': 0.0,
                'both_novel_count': 0,
                'total_structures': 0,
            }

        both_novel_count = 0
        if not self.training_structures:
            return {
                'overall_novelty': 0.0,
                'both_novel_count': 0,
                'total_structures': total,
            }

        for struct in structures:
            try:
                formula = struct.composition.reduced_formula
            except Exception:
                continue

            # Composition novel check
            comp_is_novel = formula not in self.training_formulas

            if not comp_is_novel:
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
        """
        total = len(structures)
        if total == 0:
            return {
                'sun_score': 0.0,
                'both_novel_count': 0,
                'total_structures': 0,
            }

        both_novel_count = 0
        if not self.training_structures:
            return {
                'sun_score': 0.0,
                'both_novel_count': 0,
                'total_structures': total,
            }

        if evaluations is None or len(evaluations) != len(structures):
            return {
                'sun_score': 0.0,
                'both_novel_count': 0,
                'total_structures': total,
            }

        # structures with Ed < 0.0 ((meta)stable)
        stable_indices = []
        for i, (struct, ev) in enumerate(zip(structures, evaluations)):
            try:
                if ev and ev.valid and ev.e_hull_distance is not None:
                    if not (np.isnan(ev.e_hull_distance) or np.isinf(ev.e_hull_distance)):
                        if ev.e_hull_distance < 0.0:
                            stable_indices.append(i)
            except Exception:
                continue
        
        structures_to_check = [structures[i] for i in stable_indices]

        for struct in structures_to_check:
            try:
                formula = struct.composition.reduced_formula
            except Exception:
                continue

            # Composition novel check
            comp_is_novel = formula not in self.training_formulas

            if not comp_is_novel:
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
            'sun_score': both_novel_count / total if total > 0 else 0.0,
            'both_novel_count': both_novel_count,
            'total_structures': total,
        }

    def calculate_novelty(self, structures: List[Structure]) -> Dict[str, Any]:
        """Calculate overall novelty (SUN score) and separate composition/structural novelty"""
        if not structures:
            return {
                'sun_score': 0.0,
                'novel_structures': 0,
                'unique_structures': 0,
                'total_structures': 0
            }
        
        print(f"Calculating novelty for {len(structures)} structures...")
        
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
        """Calculate success rate metrics based on stability and validity"""
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
        
        metastability_0 = stable_0 / total if total > 0 else 0.0
        metastability_003 = stable_003 / total if total > 0 else 0.0
        metastability_01 = stable_01 / total if total > 0 else 0.0
        
        # M3GNet metastability
        m3gnet_stable_01 = sum(1 for eval in valid_evals 
                              if not (np.isnan(eval.e_hull_distance_m3gnet) or np.isinf(eval.e_hull_distance_m3gnet))
                              and eval.e_hull_distance_m3gnet < 0.10)
        m3gnet_metastability = m3gnet_stable_01 / total if total > 0 else 0.0
        if m3gnet_stable_01 == 0 and all(eval.e_hull_distance_m3gnet == np.inf for eval in valid_evals):
            # Fallback to CHGNet if M3GNet not computed
            m3gnet_metastability = metastability_01
        
        # Success rate: valid AND metastable (< 0.1 eV/atom)
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
            'm3gnet_stable_structures_0.10': m3gnet_stable_01,
            'stable_structures_0': stable_0,
            'stable_structures_0.03': stable_003,
            'stable_structures_0.10': stable_01,
            'stability_rate_0.03': metastability_003,
            'stability_rate_0.10': metastability_01,
            'success_rate': success_count / total if total > 0 else 0.0,
            'success_structures': success_count,
            'total_structures': total
        }
    
    def evaluate(self, structures: List[Structure], 
                 calculate_stability: bool = True) -> Dict[str, Any]:
        """Comprehensive evaluation of structures"""
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
        
        # Stability and success rate
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
                
                # Add M3GNet statistics if available
                e_hull_distances_m3gnet = [e.e_hull_distance_m3gnet for e in valid_evals 
                                          if not (np.isnan(e.e_hull_distance_m3gnet) or np.isinf(e.e_hull_distance_m3gnet))]
                if e_hull_distances_m3gnet:
                    results['stability_stats_m3gnet'] = {
                        'min_e_hull': min(e_hull_distances_m3gnet),
                        'max_e_hull': max(e_hull_distances_m3gnet),
                        'mean_e_hull': np.mean(e_hull_distances_m3gnet),
                        'median_e_hull': np.median(e_hull_distances_m3gnet),
                        'std_e_hull': np.std(e_hull_distances_m3gnet)
                    }
            
            # Add M3GNet metastability to results
            results['m3gnet_metastability'] = success_metrics['m3gnet_metastability']

            # Compute composition/structural novelty, overall novelty, and SUN rate
            novelty = self.calculate_novelty(structures)
            overall_novelty_result = self.calculate_overall_novelty(structures)
            sun_result = self.calculate_sun(structures, evaluations=evaluations)
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

            novelty = self.calculate_novelty(structures)
            overall_novelty_result = self.calculate_overall_novelty(structures)
            results['overall_novelty'] = overall_novelty_result['overall_novelty']
            results['novelty'] = {
                'sun_score': 0.0,
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

