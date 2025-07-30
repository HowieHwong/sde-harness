"""
Base evaluator class for SDE-Harness projects.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable


class EvaluatorBase(ABC):
    """
    Base class for project-specific evaluators.
    
    Provides standard evaluation patterns for scientific discovery projects.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.metrics = {}
        self.multi_round_metrics = {}
    
    def register_metric(self, name: str, metric_fn: Callable):
        """
        Register a single-round metric.
        
        Args:
            name: Name of the metric
            metric_fn: Function that computes the metric
                      Signature: (prediction, reference, **kwargs) -> float
        """
        self.metrics[name] = metric_fn
    
    def register_multi_round_metric(self, name: str, metric_fn: Callable):
        """
        Register a multi-round metric.
        
        Args:
            name: Name of the metric
            metric_fn: Function that computes the metric
                      Signature: (history, reference, current_iteration, **kwargs) -> float
        """
        self.multi_round_metrics[name] = metric_fn
    
    @abstractmethod
    def setup_metrics(self):
        """
        Setup project-specific metrics.
        
        This method should register all metrics needed for the project.
        """
        pass
    
    def evaluate(self, 
                 prediction: str, 
                 reference: str, 
                 history: Dict[str, Any] = None,
                 current_iteration: int = 1,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate prediction against reference.
        
        Args:
            prediction: Generated prediction
            reference: Reference/ground truth
            history: Historical data for multi-round metrics
            current_iteration: Current iteration number
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dict of metric name to score
        """
        scores = {}
        
        # Compute single-round metrics
        for name, metric_fn in self.metrics.items():
            try:
                scores[name] = metric_fn(prediction, reference, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to compute metric '{name}': {e}")
                scores[name] = 0.0
        
        # Compute multi-round metrics if history is available
        if history and current_iteration > 1:
            for name, metric_fn in self.multi_round_metrics.items():
                try:
                    scores[name] = metric_fn(
                        history, reference, current_iteration, **kwargs
                    )
                except Exception as e:
                    print(f"Warning: Failed to compute multi-round metric '{name}': {e}")
                    scores[name] = 0.0
        
        return scores