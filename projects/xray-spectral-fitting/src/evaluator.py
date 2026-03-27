"""Evaluation of spectral fitting results against ground truth."""

import json
import os
from typing import Any, Dict, Optional

from .compat import EvaluatorBase


class SpectralFitEvaluator(EvaluatorBase):
    """Evaluator for X-ray spectral fitting benchmark results."""

    def __init__(self):
        super().__init__()
        self.setup_metrics()

    def setup_metrics(self):
        self.register_metric("best_reduced_cstat", _best_reduced_cstat)
        self.register_metric("cstat_vs_expected", _cstat_vs_expected)
        self.register_metric("found_expected_model", _found_expected_model)

        self.register_multi_round_metric("convergence_speed", _convergence_speed)


def load_spectrum_metadata(pha_path: str) -> Optional[Dict[str, Any]]:
    """Load metadata.json from the spectrum directory."""
    spectrum_dir = os.path.dirname(pha_path)
    metadata_path = os.path.join(spectrum_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            return json.load(f)
    return None


def evaluate_results(
    results: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate optimization results against ground truth.

    This is the main entry point called from the CLI after optimization completes.
    """
    metrics: Dict[str, Any] = {
        "generations": results["generations"],
        "oracle_calls": results["oracle_calls"],
        "converged": (
            results["best_result"] is not None
            and results["best_result"].get("success", False)
        ),
    }

    if results["best_model"]:
        best_lower = results["best_model"].lower()
        expected_sherpa = ground_truth.get("expected_model_sherpa", "").lower()

        # Record the best model expression
        metrics["best_model"] = results["best_model"]

        # Check if best model has equivalent model classes as expected
        # (ignoring instance names, treating bbody/bbodyrad as equivalent)
        if expected_sherpa:
            import re
            expected_classes = sorted(_normalize_model_classes(re.findall(r'(xs\w+)\.', expected_sherpa)))
            best_classes = sorted(_normalize_model_classes(re.findall(r'(xs\w+)\.', best_lower)))
            metrics["found_expected_model"] = (expected_classes == best_classes)
        else:
            metrics["found_expected_model"] = None

        if results["best_result"] and results["best_result"].get("success"):
            metrics["best_reduced_cstat"] = results["best_result"]["reduced_cstat"]
            # Compare to optimal value of 1.0
            metrics["cstat_vs_optimal"] = abs(results["best_result"]["reduced_cstat"] - 1.0)
            # Store fitted parameters for display
            metrics["best_fit_params"] = results["best_result"].get("params", {})

            # --- Parameter comparison (only if model matches) ---
            if metrics.get("found_expected_model"):
                expected_params = ground_truth.get("expected_parameters", {})
                fitted_params = results["best_result"].get("params", {})
                if expected_params and fitted_params:
                    param_comparison = _compare_parameters(fitted_params, expected_params)
                    metrics["parameter_comparison"] = param_comparison

    return metrics


def _normalize_model_classes(classes: list) -> list:
    """Normalize model class names to treat equivalent models the same."""
    equivalents = {
        # blackbody variants
        "xsbbodyrad": "xsbbody",
        "xszbbody": "xsbbody",
        # absorption variants — all equivalent neutral ISM absorption models
        "xsphabs": "xstbabs",
        "xswabs":  "xstbabs",
        "xstbabs": "xstbabs",
        "xsnhabs": "xstbabs",
    }
    return [equivalents.get(c, c) for c in classes]


def _compare_parameters(
    fitted: Dict[str, float], expected: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Compare fitted parameter values against ground truth using error tolerances.

    ``expected`` has the shape from metadata.json::

        { "abs1.nH": {"value": 0.0001, "error": 0.247, ...}, "bb1.kT": {"value": 1.85, "error": 0.30, ...} }

    ``fitted`` has the shape from Sherpa::

        { "abs1.nH": 0.065, "bb1.kT": 1.82, "bb1.norm": 0.0003 }

    Returns per-parameter comparison with within_tolerance flag based on errors.
    """
    comparisons: Dict[str, Any] = {}
    within_tolerance_count = 0
    total_matched = 0

    for param_name, spec in expected.items():
        expected_val = spec.get("value")
        if expected_val is None:
            continue

        expected_err = spec.get("error", 0)

        # Sherpa parameter names are case-sensitive; try exact match first,
        # then a case-insensitive fuzzy match on the suffix.
        fitted_val = fitted.get(param_name)
        if fitted_val is None:
            suffix = param_name.split(".")[-1].lower()
            for fk, fv in fitted.items():
                if fk.split(".")[-1].lower() == suffix:
                    fitted_val = fv
                    break

        if fitted_val is not None:
            total_matched += 1
            abs_err = abs(fitted_val - expected_val)
            # Within tolerance if difference is within 1-sigma
            tolerance = expected_err if expected_err > 0 else abs(expected_val) * 0.1
            within_tol = abs_err <= tolerance
            if within_tol:
                within_tolerance_count += 1

            comparisons[param_name] = {
                "expected": expected_val,
                "expected_error": expected_err,
                "fitted": round(fitted_val, 6),
                "abs_error": round(abs_err, 6),
                "tolerance": round(tolerance, 6),
                "within_tolerance": within_tol,
                "unit": spec.get("unit", ""),
            }
        else:
            comparisons[param_name] = {
                "expected": expected_val,
                "expected_error": expected_err,
                "fitted": None,
                "within_tolerance": False,
                "note": "parameter not found in best-fit model",
            }

    comparisons["_summary"] = {
        "params_matched": total_matched,
        "params_expected": len(expected),
        "params_within_tolerance": within_tolerance_count,
        "all_within_tolerance": (within_tolerance_count == total_matched) if total_matched > 0 else False,
    }

    return comparisons


# ------------------------------------------------------------------
# Individual metric functions for EvaluatorBase registration
# ------------------------------------------------------------------

def _best_reduced_cstat(prediction, reference, **kwargs):
    result = kwargs.get("best_result")
    if result and result.get("success"):
        return result["reduced_cstat"]
    return float("inf")


def _cstat_vs_expected(prediction, reference, **kwargs):
    if not isinstance(reference, dict):
        return 0.0
    expected_cstat = reference.get("expected_reduced_cstat")
    result = kwargs.get("best_result")
    if expected_cstat and result and result.get("success"):
        return result["reduced_cstat"] - expected_cstat
    return 0.0


def _found_expected_model(prediction, reference, **kwargs):
    """Check if model has exactly the same components as expected."""
    import re
    if not isinstance(reference, dict):
        return 0.0
    expected_sherpa = reference.get("expected_model_sherpa", "").lower()
    if not expected_sherpa:
        return 0.0
    expected_classes = sorted(re.findall(r'(xs\w+)\.', expected_sherpa))
    best_classes = sorted(re.findall(r'(xs\w+)\.', prediction.lower()))
    return float(expected_classes == best_classes)


def _convergence_speed(history, reference, current_iteration, **kwargs):
    """Multi-round metric: how quickly did the optimizer converge?"""
    if not history.get("scores") or len(history["scores"]) < 2:
        return 0.0
    scores = [
        s.get("best_reduced_cstat", float("inf"))
        for s in history["scores"]
        if isinstance(s, dict)
    ]
    if not scores:
        return 0.0
    best_final = min(scores)
    for i, s in enumerate(scores):
        if s <= best_final * 1.05:
            return 1.0 - (i / len(scores))
    return 0.0
