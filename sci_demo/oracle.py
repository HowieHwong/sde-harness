from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class Oracle:
    """
    Oracle for validating and evaluating model outputs with customizable metrics.
    Enhanced with multi-round support for iterative workflows.

    Use register_metric to add new metric functions. Metrics should accept (prediction, reference, **kwargs) and return a numeric score.
    Multi-round metrics can access history data to compute trends and improvements.
    """
    def __init__(self, metrics: Optional[Dict[str, Callable]] = None):
        # metrics: mapping from metric name to function
        self.metrics: Dict[str, Callable[[Any, Any], float]] = metrics or {}
        # multi-round metrics: functions that take history data
        self.multi_round_metrics: Dict[str, Callable] = {}

    def register_metric(self, name: str, func: Callable[[Any, Any], float]) -> None:
        """
        Register a new metric.

        Args:
            name: Unique name for the metric
            func: A function taking (prediction, reference, **kwargs) and returning a float
        """
        if name in self.metrics:
            raise ValueError(f"Metric '{name}' is already registered.")
        self.metrics[name] = func

    def register_multi_round_metric(self, name: str, func: Callable) -> None:
        """
        Register a new multi-round metric that can access historical data.

        Args:
            name: Unique name for the metric
            func: A function taking (history, reference, current_iteration, **kwargs) and returning a float
                  where history is a dict with 'prompts', 'outputs', 'scores' lists
        """
        if name in self.multi_round_metrics:
            raise ValueError(f"Multi-round metric '{name}' is already registered.")
        self.multi_round_metrics[name] = func

    def unregister_metric(self, name: str) -> None:
        """
        Unregister an existing metric by name.
        """
        if name in self.metrics:
            del self.metrics[name]
        elif name in self.multi_round_metrics:
            del self.multi_round_metrics[name]
        else:
            raise KeyError(f"Metric '{name}' is not registered.")

    def list_metrics(self) -> List[str]:
        """
        Return a list of registered metric names.
        """
        return list(self.metrics.keys()) + list(self.multi_round_metrics.keys())

    def list_single_round_metrics(self) -> List[str]:
        """
        Return a list of single-round metric names.
        """
        return list(self.metrics.keys())

    def list_multi_round_metrics(self) -> List[str]:
        """
        Return a list of multi-round metric names.
        """
        return list(self.multi_round_metrics.keys())

    def compute(
        self,
        prediction: Any,
        reference: Any,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute specified metrics on a single example.

        Args:
            prediction: Model output
            reference: Ground truth or expected output
            metrics: List of metric names to compute. If None, compute all registered single-round metrics.
            **kwargs: Additional args passed to metric functions

        Returns:
            Dict mapping metric name to computed score
        """
        to_compute = metrics or self.list_single_round_metrics()
        results: Dict[str, float] = {}
        for name in to_compute:
            if name not in self.metrics:
                if name in self.multi_round_metrics:
                    raise ValueError(f"Metric '{name}' is a multi-round metric. Use compute_with_history() instead.")
                else:
                    raise KeyError(f"Metric '{name}' not registered.")
            func = self.metrics[name]
            try:
                score = func(prediction, reference, **kwargs)
            except TypeError:
                # Metric doesn't accept kwargs
                score = func(prediction, reference)
            results[name] = score
        return results

    def compute_with_history(
        self,
        prediction: Any,
        reference: Any,
        history: Dict[str, List[Any]],
        current_iteration: int = 1,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute metrics including multi-round metrics that can access historical data.

        Args:
            prediction: Current model output
            reference: Ground truth or expected output
            history: Dictionary containing 'prompts', 'outputs', 'scores' lists from previous iterations
            current_iteration: Current iteration number
            metrics: List of metric names to compute. If None, compute all registered metrics.
            **kwargs: Additional args passed to metric functions

        Returns:
            Dict mapping metric name to computed score
        """
        to_compute = metrics or self.list_metrics()
        results: Dict[str, float] = {}
        
        # Compute single-round metrics
        single_round_metrics = [m for m in to_compute if m in self.metrics]
        if single_round_metrics:
            single_results = self.compute(prediction, reference, single_round_metrics, **kwargs)
            results.update(single_results)
        
        # Compute multi-round metrics
        multi_round_metrics = [m for m in to_compute if m in self.multi_round_metrics]
        for name in multi_round_metrics:
            func = self.multi_round_metrics[name]
            try:
                score = func(history, reference, current_iteration, prediction=prediction, **kwargs)
            except TypeError:
                # Metric doesn't accept all kwargs
                score = func(history, reference, current_iteration)
            results[name] = score
        
        return results

    def evaluate_batch(
        self,
        predictions: List[Any],
        references: List[Any],
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """
        Compute metrics on a batch of examples.

        Args:
            predictions: List of model outputs
            references: List of ground truths
            metrics: List of metric names to compute. If None, compute all registered single-round metrics.
            **kwargs: Additional args passed to metric functions

        Returns:
            Dict mapping metric name to list of scores per example
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")
        batch_results: Dict[str, List[float]] = {m: [] for m in (metrics or self.list_single_round_metrics())}
        for pred, ref in zip(predictions, references):
            single = self.compute(pred, ref, metrics, **kwargs)
            for name, score in single.items():
                batch_results[name].append(score)
        return batch_results

    def compute_trend_metrics(self, history: Dict[str, List[Any]], metric_name: str) -> Dict[str, float]:
        """
        Compute trend-based metrics from historical scores.

        Args:
            history: Dictionary containing historical data
            metric_name: Name of the metric to analyze trends for

        Returns:
            Dict with trend metrics like improvement_rate, consistency, etc.
        """
        if not history.get("scores"):
            return {}
        
        scores = []
        for score_dict in history["scores"]:
            if isinstance(score_dict, dict) and metric_name in score_dict:
                scores.append(score_dict[metric_name])
        
        if len(scores) < 2:
            return {"trend_available": 0.0}
        
        scores = np.array(scores)
        
        # Calculate various trend metrics
        trend_metrics = {
            "improvement_rate": float(scores[-1] - scores[0]) / len(scores),
            "total_improvement": float(scores[-1] - scores[0]),
            "consistency": 1.0 - float(np.std(scores)) if len(scores) > 1 else 1.0,
            "best_score": float(np.max(scores)),
            "worst_score": float(np.min(scores)),
            "average_score": float(np.mean(scores)),
            "is_improving": float(scores[-1] > scores[0]) if len(scores) > 1 else 0.0,
            "monotonic_improvement": float(all(scores[i] <= scores[i+1] for i in range(len(scores)-1))),
        }
        
        return trend_metrics

# Built-in multi-round metrics
def improvement_rate_metric(history: Dict[str, List[Any]], reference: Any, current_iteration: int, **kwargs) -> float:
    """
    Calculate the rate of improvement across iterations for a specific metric.
    """
    target_metric = kwargs.get("target_metric", "accuracy")
    if not history.get("scores") or len(history["scores"]) < 2:
        return 0.0
    
    scores = []
    for score_dict in history["scores"]:
        if isinstance(score_dict, dict) and target_metric in score_dict:
            scores.append(score_dict[target_metric])
    
    if len(scores) < 2:
        return 0.0
    
    return (scores[-1] - scores[0]) / len(scores)

def consistency_metric(history: Dict[str, List[Any]], reference: Any, current_iteration: int, **kwargs) -> float:
    """
    Calculate consistency of outputs across iterations.
    """
    target_metric = kwargs.get("target_metric", "accuracy")
    if not history.get("scores"):
        return 1.0
    
    scores = []
    for score_dict in history["scores"]:
        if isinstance(score_dict, dict) and target_metric in score_dict:
            scores.append(score_dict[target_metric])
    
    if len(scores) < 2:
        return 1.0
    
    return 1.0 - np.std(scores)

def convergence_metric(history: Dict[str, List[Any]], reference: Any, current_iteration: int, **kwargs) -> float:
    """
    Calculate how well the model is converging to a stable solution.
    """
    if not history.get("outputs") or len(history["outputs"]) < 2:
        return 0.0
    
    # Simple convergence based on output similarity
    recent_outputs = history["outputs"][-3:] if len(history["outputs"]) >= 3 else history["outputs"]
    
    if len(recent_outputs) < 2:
        return 0.0
    
    # Calculate similarity between recent outputs (simple word overlap)
    similarities = []
    for i in range(len(recent_outputs) - 1):
        words1 = set(recent_outputs[i].lower().split())
        words2 = set(recent_outputs[i + 1].lower().split())
        if len(words1) == 0 and len(words2) == 0:
            similarity = 1.0
        elif len(words1) == 0 or len(words2) == 0:
            similarity = 0.0
        else:
            similarity = len(words1.intersection(words2)) / len(words1.union(words2))
        similarities.append(similarity)
    
    return np.mean(similarities)

# Example usage:
if __name__ == "__main__":
    # define a simple accuracy metric
    def accuracy(pred, ref, **kwargs):
        return float(pred == ref)

    # define a placeholder BLEU metric (actual implementation omitted)
    def dummy_bleu(pred, ref, **kwargs):
        return 0.75

    oracle = Oracle()
    oracle.register_metric("accuracy", accuracy)
    oracle.register_metric("bleu", dummy_bleu)
    
    # Register multi-round metrics
    oracle.register_multi_round_metric("improvement_rate", improvement_rate_metric)
    oracle.register_multi_round_metric("consistency", consistency_metric)
    oracle.register_multi_round_metric("convergence", convergence_metric)

    pred = "The cat sat on the mat."
    ref = "The cat is sitting on the mat."

    result = oracle.compute(pred, ref)
    print("Single eval:", result)

    # Test with history
    mock_history = {
        "prompts": ["Describe a cat", "Improve the description"],
        "outputs": ["A cat sits.", "The cat sat on the mat."],
        "scores": [{"accuracy": 0.3, "bleu": 0.5}, {"accuracy": 0.7, "bleu": 0.8}]
    }
    
    result_with_history = oracle.compute_with_history(
        pred, ref, mock_history, current_iteration=3,
        target_metric="accuracy"
    )
    print("Multi-round eval:", result_with_history)
    
    # Test trend metrics
    trend_metrics = oracle.compute_trend_metrics(mock_history, "accuracy")
    print("Trend metrics:", trend_metrics)
