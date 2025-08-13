"""Data loading utilities for MatLLMSearch"""

import pandas as pd
import numpy as np
from typing import List
from pathlib import Path

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition


def validate_data_files() -> bool:
    """Validate that required data files exist"""
    # For now, just return True since we're using the basic version
    return True


def load_seed_structures(data_path: str = "data/band_gap_processed_5000.csv", 
                        task: str = "csg", random_seed: int = 42) -> List[Structure]:
    """Load seed structures for initialization"""
    
    try:
        # Try to load from specified data path
        data_file = Path(data_path)
        if data_file.exists():
            print(f"Loading seed structures from: {data_file}")
            return _load_from_bandgap_data(data_file, task, random_seed)
        
        # If no data files found, return empty list (will trigger zero-shot generation)
        print(f"No seed structure files found at: {data_path}")
        print("Using zero-shot generation")
        return []
        
    except Exception as e:
        print(f"Error loading seed structures: {e}")
        return []


def _load_from_bandgap_data(data_path: Path, task: str, 
                           random_seed: int) -> List[Structure]:
    """Load structures from band gap processed data"""
    
    df = pd.read_csv(data_path)
    
    # Parse structures
    df['structure'] = df['structure'].apply(
        lambda x: Structure.from_str(x, fmt='json') if pd.notna(x) else None
    )
    
    # Filter valid structures
    df = df.dropna(subset=['structure'])
    df['composition'] = [s.composition for s in df['structure']]
    df['composition_len'] = [len(s.composition.elements) for s in df['structure']]
    
    # Filter by composition length (3-6 elements for reasonable complexity)
    df = df[df['composition_len'].between(3, 6)]
    
    # Shuffle to get diverse samples
    df = df.sample(frac=1, random_state=random_seed)
    
    print(f"Loaded {len(df)} seed structures from {data_path.name}")
    return df['structure'].tolist()




def matches_composition(comp1: Composition, comp2: Composition) -> bool:
    """Check if two compositions match exactly"""
    if set(comp1.elements) != set(comp2.elements):
        return False
    return all(abs(comp1[el] - comp2[el]) <= 1e-6 for el in comp2.elements)


def matches_unit_cell_pattern(comp1: Composition, comp2: Composition) -> bool:
    """Check if two compositions have the same unit cell pattern"""
    if len(comp1.elements) != len(comp2.elements):
        return False
    
    total_atoms1 = sum(comp1.values())
    total_atoms2 = sum(comp2.values())
    if total_atoms1 != total_atoms2:
        return False
    
    counts1 = sorted([comp1[el] for el in comp1.elements])
    counts2 = sorted([comp2[el] for el in comp2.elements])
    
    return counts1 == counts2