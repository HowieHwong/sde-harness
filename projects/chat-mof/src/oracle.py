"""
MOF Oracle class - evaluates MOF candidates by looking up their properties
"""

import sys
import os
import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from pathlib import Path

# Add sde_harness to path  
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from sde_harness.core import Oracle


class MOFOracle(Oracle):
    """
    MOF-specific Oracle for evaluating generated MOF names.
    Looks up MOF properties in the ChatMOF database.
    """
    
    def __init__(
        self,
        database_path: Optional[str] = None,
        surface_area_column: str = "Accessible Surface Area (m^2/g)",
        mof_name_column: str = "name"
    ):
        """
        Initialize MOF Oracle with database access.
        
        Args:
            database_path: Path to MOF database file (Excel/CSV)
            surface_area_column: Column name for surface area values
            mof_name_column: Column name for MOF names
        """
        super().__init__()
        
        # Default to local data directory if not specified
        if database_path is None:
            database_path = Path(__file__).parent.parent / "data" / "coremof.xlsx"
        
        self.database_path = Path(database_path)
        self.surface_area_column = surface_area_column
        self.mof_name_column = mof_name_column
        
        # Load database
        self.df = self._load_database()
        
        # Register MOF-specific metrics
        self.register_metric("surface_area", self._surface_area_metric)
        self.register_metric("found_in_db", self._found_in_db_metric)
        self.register_metric("above_threshold", self._above_threshold_metric)
        
    def _load_database(self) -> pd.DataFrame:
        """
        Load MOF database from file.
        
        Returns:
            DataFrame containing MOF data
        """
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.database_path}")
            
        if self.database_path.suffix == ".xlsx":
            df = pd.read_excel(self.database_path)
        elif self.database_path.suffix == ".csv":
            df = pd.read_csv(self.database_path)
        else:
            raise ValueError(f"Unsupported file format: {self.database_path.suffix}")
            
        return df
    
    def _clean_mof_name(self, name: str) -> str:
        """
        Clean MOF name for matching (similar to ChatMOF's approach).
        
        Args:
            name: Raw MOF name
            
        Returns:
            Cleaned MOF name
        """
        # Remove common suffixes that might interfere with matching
        remove_list = ['_clean_h', '_clean', '_charged', '_manual', '_ion_b', '_auto', '_SL']
        str_remove_list = r"|".join(remove_list)
        cleaned = re.sub(rf"({str_remove_list})", "", name)
        
        # Remove extra whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def lookup_mof(self, mof_name: str) -> Optional[Dict[str, Any]]:
        """
        Look up a MOF by name in the database.
        
        Args:
            mof_name: Name of the MOF to look up
            
        Returns:
            Dict with MOF properties if found, None otherwise
        """
        cleaned_name = self._clean_mof_name(mof_name)
        
        # Try exact match first
        mask = self.df[self.mof_name_column].str.contains(
            re.escape(cleaned_name), case=False, na=False
        )
        matches = self.df[mask]
        
        if len(matches) > 0:
            # Return first match as dict
            return matches.iloc[0].to_dict()
        
        # Try partial matching if exact match fails
        mask = self.df[self.mof_name_column].str.contains(
            cleaned_name, case=False, na=False, regex=False
        )
        matches = self.df[mask]
        
        if len(matches) > 0:
            return matches.iloc[0].to_dict()
            
        return None
    
    def evaluate_mof_candidates(
        self, 
        mof_names: List[str],
        threshold: float = 1000.0
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a list of MOF candidates.
        
        Args:
            mof_names: List of MOF names to evaluate
            threshold: Surface area threshold for success
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for name in mof_names:
            result = {
                'mof_name': name,
                'found': False,
                'surface_area': None,
                'above_threshold': False,
                'properties': None
            }
            
            # Look up in database
            mof_data = self.lookup_mof(name)
            
            if mof_data is not None:
                result['found'] = True
                result['properties'] = mof_data
                
                # Extract surface area if available
                if self.surface_area_column in mof_data:
                    surface_area = mof_data[self.surface_area_column]
                    if pd.notna(surface_area):
                        result['surface_area'] = float(surface_area)
                        result['above_threshold'] = surface_area >= threshold
            
            results.append(result)
            
        return results
    
    def get_high_surface_area_mofs(
        self, 
        threshold: float = 1000.0, 
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get MOFs with high surface areas from the database.
        
        Args:
            threshold: Minimum surface area threshold
            top_n: Number of top MOFs to return
            
        Returns:
            List of high surface area MOFs
        """
        # Filter by surface area
        valid_data = self.df[pd.notna(self.df[self.surface_area_column])]
        high_sa_mofs = valid_data[valid_data[self.surface_area_column] >= threshold]
        
        # Sort by surface area and take top N
        sorted_mofs = high_sa_mofs.sort_values(
            by=self.surface_area_column, 
            ascending=False
        ).head(top_n)
        
        return sorted_mofs.to_dict('records')
    
    # Metric functions for sde_harness Oracle
    def _surface_area_metric(self, prediction: str, reference: Any = None, **kwargs) -> float:
        """Metric: return surface area if MOF is found, 0 otherwise."""
        mof_data = self.lookup_mof(prediction)
        if mof_data and self.surface_area_column in mof_data:
            surface_area = mof_data[self.surface_area_column]
            if pd.notna(surface_area):
                return float(surface_area)
        return 0.0
    
    def _found_in_db_metric(self, prediction: str, reference: Any = None, **kwargs) -> float:
        """Metric: 1.0 if MOF found in database, 0.0 otherwise."""
        mof_data = self.lookup_mof(prediction)
        return 1.0 if mof_data is not None else 0.0
    
    def _above_threshold_metric(self, prediction: str, reference: Any = None, threshold: float = 1000.0, **kwargs) -> float:
        """Metric: 1.0 if MOF surface area above threshold, 0.0 otherwise."""
        mof_data = self.lookup_mof(prediction)
        if mof_data and self.surface_area_column in mof_data:
            surface_area = mof_data[self.surface_area_column]
            if pd.notna(surface_area) and surface_area >= threshold:
                return 1.0
        return 0.0