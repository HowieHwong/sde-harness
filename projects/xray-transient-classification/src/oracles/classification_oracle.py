"""
Classification oracle for X-ray transient classification.
"""

import json
import re
from typing import Any, Callable, Dict, Optional

from sde_harness.core import Oracle

from ..utils.data_loader import get_ground_truth


class ClassificationOracle:
    """Oracle for evaluating X-ray transient classifications."""
    
    def __init__(self, observation: Dict[str, Any]):
        self.observation = observation
        self.ground_truth = get_ground_truth(observation)
    
    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response."""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _check_luminosity_in_range(self, value: float, lum_range: list) -> float:
        """
        Check if a luminosity value is within the expected range.
        
        Args:
            value: The predicted luminosity value
            lum_range: [min, max] expected range
        
        Returns:
            1.0 if within range, 0.0 otherwise
        """
        if value is None or not lum_range:
            return 0.0
        
        try:
            value = float(value)
        except (ValueError, TypeError):
            return 0.0
        
        if value <= 0:
            return 0.0
        
        lum_min, lum_max = lum_range[0], lum_range[1]
        return 1.0 if lum_min <= value <= lum_max else 0.0
    
    def _top1_correct(self, response: str) -> float:
        """Check if classification is correct."""
        if not self.ground_truth:
            return 0.0
        
        parsed = self._parse_response(response)
        if not parsed:
            return 0.0
        
        correct = self.ground_truth.get('classification', '')
        predicted = parsed.get('classification', '')
        
        return 1.0 if predicted == correct else 0.0
    
    def _alternatives_identified(self, response: str) -> float:
        """Check if a valid alternative was listed."""
        if not self.ground_truth:
            return 0.0
        
        valid_alts = self.ground_truth.get('valid_alternatives', [])
        if not valid_alts:
            return 1.0
        
        parsed = self._parse_response(response)
        if not parsed:
            return 0.0
        
        predicted_alts = parsed.get('alternatives', [])
        
        for predicted in predicted_alts:
            if predicted in valid_alts:
                return 1.0
        
        return 0.0
    
    def _luminosity_powerlaw(self, response: str) -> float:
        """Check if power law luminosity calculation is correct."""
        if not self.ground_truth:
            return 0.0
        
        lum_range = self.ground_truth.get('luminosity_powerlaw_erg_s_range')
        if not lum_range:
            return 0.0
        
        parsed = self._parse_response(response)
        if not parsed:
            return 0.0
        
        value = parsed.get('luminosity_powerlaw_erg_s')
        return self._check_luminosity_in_range(value, lum_range)
    
    def _luminosity_blackbody(self, response: str) -> float:
        """Check if blackbody luminosity calculation is correct."""
        if not self.ground_truth:
            return 0.0
        
        lum_range = self.ground_truth.get('luminosity_blackbody_erg_s_range')
        if not lum_range:
            return 0.0
        
        parsed = self._parse_response(response)
        if not parsed:
            return 0.0
        
        value = parsed.get('luminosity_blackbody_erg_s')
        return self._check_luminosity_in_range(value, lum_range)
    
    def get_single_round_metrics(self) -> Dict[str, Callable[[str], float]]:
        """Get metric functions."""
        return {
            'top1_correct': self._top1_correct,
            'alternatives_identified': self._alternatives_identified,
            'luminosity_powerlaw': self._luminosity_powerlaw,
            'luminosity_blackbody': self._luminosity_blackbody,
        }
    
    def evaluate(self, response: str) -> Dict[str, float]:
        """Evaluate a response."""
        metrics = self.get_single_round_metrics()
        return {name: func(response) for name, func in metrics.items()}
    
    def create_sde_oracle(self) -> Oracle:
        """Create SDE-Harness Oracle."""
        metrics = self.get_single_round_metrics()
        
        wrapped_metrics = {}
        for name, func in metrics.items():
            def make_wrapper(f):
                def wrapper(response: str, **kwargs) -> float:
                    return f(response)
                return wrapper
            wrapped_metrics[name] = make_wrapper(func)
        
        return Oracle(metrics=wrapped_metrics)
