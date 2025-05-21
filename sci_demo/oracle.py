from typing import Any, Callable, Dict, List, Optional, Tuple


class Oracle:
    """
    Oracle for validating and evaluating model outputs with customizable metrics.

    Use register_metric to add new metric functions. Metrics should accept (prediction, reference, **kwargs) and return a numeric score.
    """
    def __init__(self, metrics: Optional[Dict[str, Callable]] = None):
        # metrics: mapping from metric name to function
        self.metrics: Dict[str, Callable[[Any, Any], float]] = metrics or {}

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

    def unregister_metric(self, name: str) -> None:
        """
        Unregister an existing metric by name.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' is not registered.")
        del self.metrics[name]

    def list_metrics(self) -> List[str]:
        """
        Return a list of registered metric names.
        """
        return list(self.metrics.keys())

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
            metrics: List of metric names to compute. If None, compute all registered.
            **kwargs: Additional args passed to metric functions

        Returns:
            Dict mapping metric name to computed score
        """
        to_compute = metrics or self.list_metrics()
        results: Dict[str, float] = {}
        for name in to_compute:
            if name not in self.metrics:
                raise KeyError(f"Metric '{name}' not registered.")
            func = self.metrics[name]
            try:
                score = func(prediction, reference, **kwargs)
            except TypeError:
                # Metric doesn't accept kwargs
                score = func(prediction, reference)
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
            metrics: List of metric names to compute. If None, compute all registered.
            **kwargs: Additional args passed to metric functions

        Returns:
            Dict mapping metric name to list of scores per example
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length.")
        batch_results: Dict[str, List[float]] = {m: [] for m in (metrics or self.list_metrics())}
        for pred, ref in zip(predictions, references):
            single = self.compute(pred, ref, metrics, **kwargs)
            for name, score in single.items():
                batch_results[name].append(score)
        return batch_results

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

    pred = "The cat sat on the mat."
    ref = "The cat is sitting on the mat."

    result = oracle.compute(pred, ref)
    print("Single eval:", result)

    preds = [pred, "Hello world"]
    refs = [ref, "Hello world!"]
    batch = oracle.evaluate_batch(preds, refs)
    print("Batch eval:", batch)
