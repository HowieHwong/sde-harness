from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

# Recording
import weave
try:
    from .utils import safe_weave_log
except ImportError:
    # Handle direct execution
    from utils import safe_weave_log

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

    @weave.op()
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
        
        # Log metric registration
        safe_weave_log({
            "metric_registered": {
                "name": name,
                "type": "single_round",
                "total_single_round_metrics": len(self.metrics)
            }
        })

    @weave.op()
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
        
        # Log metric registration
        safe_weave_log({
            "metric_registered": {
                "name": name,
                "type": "multi_round",
                "total_multi_round_metrics": len(self.multi_round_metrics)
            }
        })

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

    @weave.op()
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
        
        # Log computation start
        safe_weave_log({
            "single_round_evaluation": {
                "metrics_to_compute": to_compute,
                "prediction_length": len(str(prediction)) if prediction else 0,
                "reference_length": len(str(reference)) if reference else 0
            }
        })
        
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
        
        # Log individual metric results
        safe_weave_log({
            "single_round_results": {
                "metrics_computed": results,
                "num_metrics": len(results)
            }
        })
        
        return results

    @weave.op()
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
        
        # Log multi-round computation start
        safe_weave_log({
            "multi_round_evaluation": {
                "current_iteration": current_iteration,
                "metrics_to_compute": to_compute,
                "history_length": len(history.get("outputs", [])),
                "prediction_length": len(str(prediction)) if prediction else 0
            }
        })
        
        # Compute single-round metrics
        single_round_metrics = [m for m in to_compute if m in self.metrics]
        if single_round_metrics:
            single_results = self.compute(prediction, reference, single_round_metrics, **kwargs)
            results.update(single_results)
        
        # Compute multi-round metrics
        multi_round_metrics = [m for m in to_compute if m in self.multi_round_metrics]
        multi_round_results = {}
        
        for name in multi_round_metrics:
            func = self.multi_round_metrics[name]
            try:
                score = func(history, reference, current_iteration, prediction=prediction, **kwargs)
            except TypeError:
                # Metric doesn't accept all kwargs
                score = func(history, reference, current_iteration)
            multi_round_results[name] = score
            results[name] = score
        
        # Log multi-round specific results
        safe_weave_log({
            "multi_round_results": {
                "single_round_metrics": {k: v for k, v in results.items() if k in single_round_metrics},
                "multi_round_metrics": multi_round_results,
                "total_metrics_computed": len(results)
            }
        })
        
        return results

    @weave.op()
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
        
        # Log batch evaluation start
        safe_weave_log({
            "batch_evaluation": {
                "batch_size": len(predictions),
                "metrics_to_compute": metrics or self.list_single_round_metrics()
            }
        })
        
        batch_results: Dict[str, List[float]] = {m: [] for m in (metrics or self.list_single_round_metrics())}
        for pred, ref in zip(predictions, references):
            single = self.compute(pred, ref, metrics, **kwargs)
            for name, score in single.items():
                batch_results[name].append(score)
        
        # Log batch results summary
        batch_summary = {}
        for metric, scores in batch_results.items():
            batch_summary[metric] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }
        
        safe_weave_log({
            "batch_results_summary": batch_summary
        })
        
        return batch_results

    @weave.op()
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
        
        # Log trend analysis
        safe_weave_log({
            "trend_analysis": {
                "metric_name": metric_name,
                "num_data_points": len(scores),
                "trend_metrics": trend_metrics
            }
        })
        
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
    # Initialize weave for testing this module only
    weave.init("oracle_module_test")
    
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

    @weave.op()
    def evaluate_response(self, response: str, expected: str = None, criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate a generated response against expected output and criteria."""
        
        # Log evaluation start
        safe_weave_log({
            "evaluation_start": {
                "response_length": len(response),
                "has_expected": expected is not None,
                "criteria_count": len(criteria) if criteria else 0
            }
        })
        
        evaluation_result = {
            "response": response,
            "expected": expected,
            "criteria": criteria or {},
            "scores": {},
            "overall_score": 0.0,
            "passed": False,
            "feedback": []
        }
        
        # Basic evaluation metrics
        scores = {}
        
        # Length evaluation
        if expected:
            length_ratio = len(response) / len(expected) if expected else 0
            scores["length_similarity"] = min(1.0, 1.0 / (abs(length_ratio - 1.0) + 0.1))
        
        # Content similarity (simple word overlap)
        if expected:
            response_words = set(response.lower().split())
            expected_words = set(expected.lower().split())
            overlap = len(response_words.intersection(expected_words))
            scores["content_overlap"] = overlap / len(expected_words) if expected_words else 0
        
        # Custom criteria evaluation
        if criteria:
            for criterion, threshold in criteria.items():
                if criterion == "min_length":
                    scores[criterion] = 1.0 if len(response) >= threshold else 0.0
                elif criterion == "contains_keywords":
                    keywords = threshold if isinstance(threshold, list) else [threshold]
                    found_keywords = sum(1 for kw in keywords if kw.lower() in response.lower())
                    scores[criterion] = found_keywords / len(keywords)
                elif criterion == "factual_accuracy":
                    # Placeholder for more sophisticated factual accuracy check
                    scores[criterion] = 0.8  # Mock score
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        
        evaluation_result.update({
            "scores": scores,
            "overall_score": overall_score,
            "passed": overall_score >= self.threshold,
            "feedback": self._generate_feedback(scores, overall_score)
        })
        
        # Log evaluation results
        safe_weave_log({
            "evaluation_results": {
                "overall_score": overall_score,
                "passed": evaluation_result["passed"],
                "scores": scores
            }
        })
        
        return evaluation_result

    @weave.op()
    def validate_generation_quality(self, generation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the quality of a generation result."""
        
        # Log validation start
        safe_weave_log({
            "validation_start": {
                "generation_result_keys": list(generation_result.keys()),
                "validation_type": "generation_quality"
            }
        })
        
        validation_result = {
            "input_generation": generation_result,
            "quality_scores": {},
            "validation_passed": False,
            "issues_found": [],
            "recommendations": []
        }
        
        generated_text = generation_result.get("generated_text", "")
        
        # Quality checks
        quality_scores = {}
        issues = []
        
        # Length check
        if len(generated_text) < 10:
            issues.append("Generated text is too short")
            quality_scores["length"] = 0.0
        else:
            quality_scores["length"] = min(1.0, len(generated_text) / 100)
        
        # Coherence check (simple sentence structure)
        sentences = generated_text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if avg_sentence_length < 3:
            issues.append("Sentences are too short, may lack coherence")
            quality_scores["coherence"] = 0.3
        else:
            quality_scores["coherence"] = min(1.0, avg_sentence_length / 15)
        
        # Repetition check
        words = generated_text.lower().split()
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words) if words else 0
        if repetition_ratio < 0.5:
            issues.append("High repetition detected")
            quality_scores["diversity"] = repetition_ratio
        else:
            quality_scores["diversity"] = repetition_ratio
        
        # Overall quality score
        overall_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
        
        validation_result.update({
            "quality_scores": quality_scores,
            "overall_quality": overall_quality,
            "validation_passed": overall_quality >= self.threshold and len(issues) == 0,
            "issues_found": issues,
            "recommendations": self._generate_quality_recommendations(quality_scores, issues)
        })
        
        # Log validation results
        safe_weave_log({
            "validation_results": {
                "overall_quality": overall_quality,
                "validation_passed": validation_result["validation_passed"],
                "issues_count": len(issues)
            }
        })
        
        return validation_result

    @weave.op()
    def compare_outputs(self, outputs: List[str], criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compare multiple outputs and rank them."""
        
        # Log comparison start
        safe_weave_log({
            "comparison_start": {
                "output_count": len(outputs),
                "criteria": criteria or {}
            }
        })
        
        comparison_result = {
            "outputs": outputs,
            "individual_scores": [],
            "rankings": [],
            "best_output": None,
            "comparison_summary": {}
        }
        
        # Evaluate each output individually
        individual_results = []
        for i, output in enumerate(outputs):
            result = self.evaluate_response(output, criteria=criteria)
            individual_results.append({
                "index": i,
                "output": output,
                "score": result["overall_score"],
                "detailed_scores": result["scores"]
            })
        
        # Sort by score (descending)
        individual_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Generate rankings
        rankings = []
        for rank, result in enumerate(individual_results, 1):
            rankings.append({
                "rank": rank,
                "output_index": result["index"],
                "score": result["score"],
                "output_preview": result["output"][:100] + "..." if len(result["output"]) > 100 else result["output"]
            })
        
        best_output = individual_results[0] if individual_results else None
        
        comparison_result.update({
            "individual_scores": individual_results,
            "rankings": rankings,
            "best_output": best_output,
            "comparison_summary": {
                "best_score": best_output["score"] if best_output else 0,
                "worst_score": individual_results[-1]["score"] if individual_results else 0,
                "score_range": best_output["score"] - individual_results[-1]["score"] if len(individual_results) > 1 else 0,
                "average_score": sum(r["score"] for r in individual_results) / len(individual_results) if individual_results else 0
            }
        })
        
        # Log comparison results
        safe_weave_log({
            "comparison_results": {
                "best_score": comparison_result["comparison_summary"]["best_score"],
                "average_score": comparison_result["comparison_summary"]["average_score"],
                "output_count": len(outputs)
            }
        })
        
        return comparison_result

    @weave.op()
    def validate_factual_accuracy(self, text: str, facts: List[str] = None) -> Dict[str, Any]:
        """Validate factual accuracy of generated text."""
        
        # Log validation start
        safe_weave_log({
            "factual_validation_start": {
                "text_length": len(text),
                "facts_count": len(facts) if facts else 0
            }
        })
        
        validation_result = {
            "text": text,
            "reference_facts": facts or [],
            "accuracy_score": 0.0,
            "verified_facts": [],
            "potential_inaccuracies": [],
            "confidence": 0.0
        }
        
        if not facts:
            # Simple heuristic validation without reference facts
            potential_issues = []
            
            # Check for absolute statements without qualifiers
            absolute_words = ["all", "never", "always", "impossible", "certain"]
            for word in absolute_words:
                if word in text.lower():
                    potential_issues.append(f"Absolute statement detected: '{word}'")
            
            # Check for unsupported claims (basic patterns)
            claim_patterns = ["studies show", "research proves", "it is proven that"]
            for pattern in claim_patterns:
                if pattern in text.lower():
                    potential_issues.append(f"Unsupported claim: '{pattern}'")
            
            # Simple accuracy scoring based on issues found
            accuracy_score = max(0.0, 1.0 - (len(potential_issues) * 0.2))
            confidence = 0.6  # Lower confidence without reference facts
            
            validation_result.update({
                "accuracy_score": accuracy_score,
                "potential_inaccuracies": potential_issues,
                "confidence": confidence
            })
        
        else:
            # Validate against provided facts
            verified = []
            inaccuracies = []
            
            for fact in facts:
                # Simple substring matching (in real implementation, use NLP/similarity)
                if fact.lower() in text.lower():
                    verified.append(fact)
                else:
                    # Check for contradictions (simplified)
                    fact_words = set(fact.lower().split())
                    text_words = set(text.lower().split())
                    if len(fact_words.intersection(text_words)) > len(fact_words) * 0.5:
                        verified.append(fact)
                    else:
                        inaccuracies.append(f"Fact not supported: {fact}")
            
            accuracy_score = len(verified) / len(facts) if facts else 1.0
            confidence = 0.8  # Higher confidence with reference facts
            
            validation_result.update({
                "accuracy_score": accuracy_score,
                "verified_facts": verified,
                "potential_inaccuracies": inaccuracies,
                "confidence": confidence
            })
        
        # Log validation results
        safe_weave_log({
            "factual_validation_results": {
                "accuracy_score": accuracy_score,
                "confidence": confidence,
                "verified_facts_count": len(validation_result["verified_facts"]),
                "potential_inaccuracies_count": len(validation_result["potential_inaccuracies"])
            }
        })
        
        return validation_result

    def get_oracle_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for oracle evaluations."""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
        
        total_evaluations = len(self.evaluation_history)
        passed_evaluations = sum(1 for eval_result in self.evaluation_history 
                                if eval_result.get("passed", False))
        
        all_scores = [eval_result.get("overall_score", 0) 
                     for eval_result in self.evaluation_history]
        
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        max_score = max(all_scores) if all_scores else 0
        min_score = min(all_scores) if all_scores else 0
        
        current_metrics = {
            "total_evaluations": total_evaluations,
            "passed_evaluations": passed_evaluations,
            "pass_rate": passed_evaluations / total_evaluations if total_evaluations > 0 else 0,
            "average_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "current_threshold": self.threshold,
            "evaluation_types": list(set(eval_result.get("type", "unknown") 
                                       for eval_result in self.evaluation_history))
        }
        
        # Log metrics collection
        safe_weave_log({
            "oracle_metrics": current_metrics
        })
        
        return current_metrics
