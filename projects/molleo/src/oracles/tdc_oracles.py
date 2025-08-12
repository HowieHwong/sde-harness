"""TDC (Therapeutics Data Commons) oracle implementations"""

from typing import Dict, Any
import tdc
from tdc import Oracle as TDCOracleBase
from .base import MolecularOracle


class TDCOracle(MolecularOracle):
    """Oracle using TDC scoring functions"""
    
    # Mapping of common oracle names
    ORACLE_MAPPING = {
        'jnk3': 'JNK3',
        'gsk3b': 'GSK3β',
        'drd2': 'DRD2',
        'qed': 'QED',
        'sa': 'SA',
        'logp': 'LogP',
        'penalized_logp': 'Penalized logP',
    }
    
    def __init__(self, oracle_name: str, **kwargs):
        """
        Initialize TDC oracle
        
        Args:
            oracle_name: Name of the oracle (e.g., 'jnk3', 'qed', 'sa')
            **kwargs: Additional arguments for the oracle
        """
        super().__init__(oracle_name)
        
        # Map oracle name if needed
        tdc_name = self.ORACLE_MAPPING.get(oracle_name, oracle_name)
        
        # Initialize TDC oracle
        self.tdc_oracle = TDCOracleBase(tdc_name, **kwargs)
        
    def _evaluate_molecule_impl(self, smiles: str) -> float:
        """Evaluate a single molecule using TDC oracle"""
        try:
            score = self.tdc_oracle(smiles)
            return float(score)
        except Exception as e:
            print(f"Error evaluating {smiles}: {e}")
            return 0.0
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get oracle statistics"""
        if self.history:
            scores = [h['score'] for h in self.history]
            return {
                'call_count': self.call_count,
                'mean_score': sum(scores) / len(scores),
                'max_score': max(scores),
                'min_score': min(scores),
                'num_evaluations': len(self.history),
            }
        return {
            'call_count': 0,
            'mean_score': 0,
            'max_score': 0,
            'min_score': 0,
            'num_evaluations': 0,
        }