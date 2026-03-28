"""
Data utilities for loading and processing molecular data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re


class MolecularDataLoader:
    """Load and process molecular data from CSV files."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file containing molecular data
        """
        self.data_path = data_path
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """Load the molecular data from CSV."""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.data)} molecular records")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = pd.DataFrame()
    
    def get_molecule_by_id(self, molecule_id: str) -> Optional[Dict]:
        """
        Retrieve a molecule by its ID with robust matching.
        """
        if self.data is None or self.data.empty:
            return None
        # Ensure column exists
        if 'molecule_id' not in self.data.columns:
            return None
        # Coerce to string and strip whitespace for robust matching
        col = self.data['molecule_id'].astype(str).str.strip()
        key = str(molecule_id).strip()
        mask = col == key
        if not mask.any():
            return None
        return self.data.loc[mask].iloc[0].to_dict()
    
    def get_random_molecule(self) -> Optional[Dict]:
        """
        Get a random molecule from the dataset.
        
        Returns:
            Dictionary containing random molecule data or None if dataset is empty
        """
        if self.data is None or self.data.empty:
            return None
        
        return self.data.sample(n=1).iloc[0].to_dict()
    
    def search_by_smiles(self, smiles: str) -> Optional[Dict]:
        """
        Search for a molecule by SMILES string.
        
        Args:
            smiles: SMILES string to search for
            
        Returns:
            Dictionary containing molecule data or None if not found
        """
        if self.data is None or self.data.empty:
            return None
        
        molecule = self.data[self.data['SMILES'] == smiles]
        if molecule.empty:
            return None
        
        return molecule.iloc[0].to_dict()
    
    def get_nmr_data(self, SMILES: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get NMR data for a specific molecule.
        
        Args:
            SMILES: The molecule ID
            
        Returns:
            Tuple of (H_NMR, C_NMR) strings or (None, None) if not found
        """
        molecule = self.get_molecule_by_id(SMILES)
        if molecule is None:
            return None, None
        
        return molecule.get('H_NMR'), molecule.get('C_NMR')
    
    def get_functional_groups(self, SMILES: str) -> Dict[str, bool]:
        """
        Get functional group information for a molecule.
        
        Args:
            SMILES: The molecule ID
            
        Returns:
            Dictionary mapping functional group names to boolean values
        """
        molecule = self.get_molecule_by_id(SMILES)
        if molecule is None:
            return {}
        
        # Extract functional group columns (excluding metadata columns)
        functional_groups = {}
        for col in molecule.keys():
            if col not in [ 'SMILES','H_NMR', 'C_NMR']:
                functional_groups[col] = bool(molecule[col])
        
        return functional_groups
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self.data is None or self.data.empty:
            return {}
        
        info = {
            'total_molecules': len(self.data),
            'columns': list(self.data.columns),
            'functional_groups': [col for col in self.data.columns 
                                if col not in ['SMILES', 'H_NMR', 'C_NMR']],
            'sample_molecules': self.data['SMILES'].head(10).tolist()
        }
        
        return info


class NMRProcessor:
    """Process and analyze NMR data."""
    
    @staticmethod
    def parse_nmr_peaks(nmr_string: str) -> List[Dict]:
        """
        Parse NMR string into structured peak data.
        
        Args:
            nmr_string: NMR data string (e.g., "NMR: 0.87 (3H, t, J = 6.5 Hz)")
            
        Returns:
            List of dictionaries containing peak information
        """
        if not nmr_string or 'NMR:' not in nmr_string:
            return []
        
        # Remove "NMR:" prefix and clean up
        peaks_str = nmr_string.replace('NMR:', '').strip()
        
        # Split by commas and process each peak
        peaks = []
        for peak_str in peaks_str.split(','):
            peak_str = peak_str.strip()
            if not peak_str:
                continue
            
            # Extract chemical shift, multiplicity, and coupling constants
            # Pattern: chemical_shift (integration, multiplicity, J = coupling)
            # Handle variations in spacing around "J ="
            match = re.match(r'(\d+\.?\d*)\s*\(([^)]+)\)', peak_str)
            if match:
                chemical_shift = float(match.group(1))
                details = match.group(2)
                
                # Parse details
                parts = details.split(',')
                integration = parts[0].strip() if len(parts) > 0 else ""
                multiplicity = parts[1].strip() if len(parts) > 1 else ""
                coupling = ""
                
                # Look for coupling constant in remaining parts
                for part in parts[2:]:
                    if 'J' in part and '=' in part:
                        # Extract the coupling value
                        coupling_match = re.search(r'J\s*=\s*([\d.]+)', part)
                        if coupling_match:
                            coupling = coupling_match.group(1)
                        break
                
                peaks.append({
                    'chemical_shift': chemical_shift,
                    'integration': integration,
                    'multiplicity': multiplicity,
                    'coupling': coupling,
                    'raw': peak_str
                })
        
        return peaks
    
    @staticmethod
    def split_nmr_peaks(nmr_string):
        """
        Splits an NMR data string into individual peaks, including nested peaks.
        
        Parameters:
            nmr_string (str): The NMR data string.
            
        Returns:
            list: A list of individual peaks as strings.
        """
        import re
        
        # Normalize spaces in the input
        nmr_string = re.sub(r"\s+", " ", nmr_string.strip())
        # remove δ if present
        nmr_string = nmr_string.replace('δ', '')
        
        # First, split the string into main peaks
        main_peaks = []
        current_peak = ''
        parentheses_count = 0

        for char in nmr_string:
            if char == '(':
                parentheses_count += 1
            elif char == ')':
                parentheses_count -= 1
            elif char == ',' and parentheses_count == 0:
                main_peaks.append(current_peak.strip())
                current_peak = ''
                continue
            current_peak += char

        if current_peak:
            main_peaks.append(current_peak.strip())

        return main_peaks

    @staticmethod
    def extract_13c_data(peak):
        """
        Extracts information from an individual 13C NMR peak.
        
        Parameters:
            peak (str): A single NMR peak string.
            
        Returns:
            list: A list containing chemical shift, carbon count, type.
        """
        import re
        
        result = []
        # Regex patterns
        shift_range_pattern = r"([\d\.]+-[\d\.]+)\s*\((\d+)C,\s*(.*)\)"
        shift_single_pattern = r"^([\d\.]+)\s*\((\d+)C,\s*(.*)\)"
        inside_peak_pattern = r"^([\d\.]+)\s*\((.*)\)"

        # Extract chemical shift(s)
        range_match = re.match(shift_range_pattern, peak)
        if range_match:
            number = int(range_match.group(2))
            inside_peaks = range_match.group(3)
            peaklist = inside_peaks.split(', ')
            n_inside_peaks = len(peaklist)
            normalized_number = number / n_inside_peaks
            for i in peaklist:
                inside_match = re.match(inside_peak_pattern, i)
                if inside_match:
                    shift = float(inside_match.group(1))
                    mult = inside_match.group(2)
                    result.append([shift, normalized_number, mult])
        else:
            single_match = re.match(shift_single_pattern, peak)
            if single_match:
                shift = float(single_match.group(1))
                number = int(single_match.group(2))
                mult = single_match.group(3)
                result = [[shift, number, mult]]

        return result

    @staticmethod
    def extract_1h_data(peak):
        """
        Extracts information from an individual 1H NMR peak.
        
        Parameters:
            peak (str): A single NMR peak string.
            
        Returns:
            list: A list containing chemical shift, hydrogen count, type.
        """
        import re
        
        result = []
        # Regex patterns
        shift_range_pattern = r"([\d\.]+-[\d\.]+)\s*\((\d+)H,\s*(.*)\)"
        shift_single_pattern = r"^([\d\.]+)\s*\(([\d+])H,\s*([a-z]+),\s*(.*)\)"
        shift_singlet_pattern = r"^([\d\.]+)\s*\((\d+)H,\s*(.*)\)"
        inside_peak_pattern = r"^([\d\.]+)\s*\(\s*([a-z]+),(.*)\)"

        # Extract chemical shift(s)
        range_match = re.match(shift_range_pattern, peak)
        if range_match:
            peaklist = []
            current_peak = ''
            parentheses_count = 0
            number = int(range_match.group(2))
            inside_peaks = range_match.group(3)

            for char in inside_peaks:
                if char == '(':
                    parentheses_count += 1
                elif char == ')':
                    parentheses_count -= 1
                elif char == ',' and parentheses_count == 0:
                    peaklist.append(current_peak.strip())
                    current_peak = ''
                    continue
                current_peak += char

            if current_peak:
                peaklist.append(current_peak.strip())

            n_inside_peaks = len(peaklist)
            normalized_number = number / n_inside_peaks
            for i in peaklist:
                inside_match = re.match(inside_peak_pattern, i)
                if inside_match:
                    shift = float(inside_match.group(1))
                    mult = inside_match.group(2)
                    result.append([shift, normalized_number, mult])
        else:
            single_match = re.match(shift_single_pattern, peak)
            if single_match:
                shift = float(single_match.group(1))
                number = int(single_match.group(2))
                mult = single_match.group(3)
                result = [[shift, number, mult]]
            else:
                single_match = re.match(shift_singlet_pattern, peak)
                if single_match:
                    shift = float(single_match.group(1))
                    number = int(single_match.group(2))
                    mult = single_match.group(3)
                    result = [[shift, number, mult]]

        return result

    @staticmethod
    def calculate_nmr_similarity_advanced(nmr1: str, nmr2: str, nucleus: str = "H", tolerance: float = 0.05) -> float:
        """
        Advanced NMR similarity calculation based on the notebook approach.
        
        Args:
            nmr1: First NMR spectrum string
            nmr2: Second NMR spectrum string
            nucleus: "H" for proton NMR, "C" for carbon NMR
            tolerance: Chemical shift tolerance in ppm
            
        Returns:
            Similarity score between 0 and 1
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Process NMR peaks
            peaks1 = NMRProcessor.split_nmr_peaks(nmr1)
            peaks2 = NMRProcessor.split_nmr_peaks(nmr2)
            
            logger.debug(f"Split into {len(peaks1)} and {len(peaks2)} peaks")
            
            # Extract structured data
            data1 = []
            data2 = []
            
            if nucleus == "H":
                for peaks in peaks1:
                    data1.extend(NMRProcessor.extract_1h_data(peaks))
                for peaks in peaks2:
                    data2.extend(NMRProcessor.extract_1h_data(peaks))
            else:  # C-NMR
                for peaks in peaks1:
                    data1.extend(NMRProcessor.extract_13c_data(peaks))
                for peaks in peaks2:
                    data2.extend(NMRProcessor.extract_13c_data(peaks))
            
            logger.debug(f"Extracted {len(data1)} and {len(data2)} data points")
            
            if not data1 or not data2:
                logger.warning("No data extracted from one or both NMR strings")
                return 0.0
            
            # Calculate similarity using the notebook approach
            total_score = 0
            max_possible_score = 0
            
            # Use the shorter dataset length to avoid index errors
            min_length = min(len(data1), len(data2))
            
            for i in range(min_length):
                entry = data1[i]
                target_entry = data2[i]
                
                # Column 1: Chemical shift - exact match within threshold
                if len(entry) > 0 and len(target_entry) > 0:
                    col1_diff = abs(entry[0] - target_entry[0])
                    col1_score = 1 if col1_diff <= tolerance else 0
                    total_score += col1_score
                    max_possible_score += 1
                    
                    # Column 2: Integration - absolute difference (smaller distance gets higher score)
                    if len(entry) > 1 and len(target_entry) > 1:
                        col_diff = abs(entry[1] - target_entry[1])
                        col_score = 1 / (1 + col_diff)  # Normalize distance
                        total_score += col_score
                        max_possible_score += 1
                    
                    # Column 3: Multiplicity - exact match
                    if len(entry) > 2 and len(target_entry) > 2:
                        col3_score = 1 if entry[2] == target_entry[2] else 0
                        total_score += col3_score
                        max_possible_score += 1
            
            if max_possible_score == 0:
                return 0.0
            
            # Normalize by the maximum possible score
            normalized_score = total_score / max_possible_score
            logger.debug(f"Advanced similarity: {total_score}/{max_possible_score} = {normalized_score:.3f}")
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error in advanced NMR similarity calculation: {e}")
            return 0.0

    @staticmethod
    def calculate_nmr_similarity(nmr1: str, nmr2: str, tolerance: float = 0.5) -> float:
        """
        Calculate similarity between two NMR spectra using the advanced method.
        
        Args:
            nmr1: First NMR spectrum string
            nmr2: Second NMR spectrum string
            tolerance: Chemical shift tolerance in ppm
            
        Returns:
            Similarity score between 0 and 1
        """
        # Try to determine if this is H-NMR or C-NMR based on content
        if 'H' in nmr1 and 'H' in nmr2:
            nucleus = "H"
        elif 'C' in nmr1 and 'C' in nmr2:
            nucleus = "C"
        else:
            # Default to H-NMR if we can't determine
            nucleus = "H"
        
        return NMRProcessor.calculate_nmr_similarity_advanced(nmr1, nmr2, nucleus, tolerance)
    
    @staticmethod
    def extract_key_features(nmr_string: str) -> Dict:
        """
        Extract key features from NMR data.
        
        Args:
            nmr_string: NMR data string
            
        Returns:
            Dictionary containing key NMR features
        """
        peaks = NMRProcessor.parse_nmr_peaks(nmr_string)
        
        if not peaks:
            return {}
        
        features = {
            'num_peaks': len(peaks),
            'chemical_shift_range': {
                'min': min(p['chemical_shift'] for p in peaks),
                'max': max(p['chemical_shift'] for p in peaks)
            },
            'aromatic_peaks': len([p for p in peaks if p['chemical_shift'] > 6.0]),
            'aliphatic_peaks': len([p for p in peaks if p['chemical_shift'] <= 6.0]),
            'peaks': peaks
        }
        
        return features
