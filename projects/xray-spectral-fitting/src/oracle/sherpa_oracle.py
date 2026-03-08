"""Sherpa-based oracle for X-ray spectral fitting."""

from typing import Any, Dict, List, Optional, Tuple

from ..compat import Oracle


class SherpaOracle(Oracle):
    """
    Oracle that wraps Sherpa spectral fitting.

    Registers domain-specific metrics and provides a ``fit_model`` method
    that loads a PHA spectrum, sets a model expression, and runs a
    Levenberg-Marquardt fit returning C-stat results.
    """

    def __init__(
        self,
        pha_file: str,
        energy_range: Tuple[float, float] = (0.3, 7.0),
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.pha_file = pha_file
        self.energy_range = energy_range
        self.metadata = metadata or {}
        self.call_count = 0
        self.fit_cache: Dict[str, Dict[str, Any]] = {}
        self._observation: Optional[Dict[str, Any]] = None

        self._register_metrics()

    def _register_metrics(self):
        """Register spectral-fitting evaluation metrics."""
        self.register_metric("reduced_cstat", self._metric_reduced_cstat)
        self.register_metric("found_expected_component", self._metric_found_expected_component)
        self.register_metric("found_correct_model_type", self._metric_found_correct_model_type)
        self.register_metric("models_explored", self._metric_models_explored)

    # ------------------------------------------------------------------
    # Metric implementations (prediction=best_model str, reference=ground_truth dict)
    # ------------------------------------------------------------------

    @staticmethod
    def _metric_reduced_cstat(prediction: Any, reference: Any, **kwargs) -> float:
        """Lower is better; returns the reduced C-stat of the best fit."""
        result = kwargs.get("best_result")
        if result and result.get("success"):
            return result["reduced_cstat"]
        return float("inf")

    @staticmethod
    def _metric_found_expected_component(prediction: Any, reference: Any, **kwargs) -> float:
        if not isinstance(reference, dict):
            return 0.0
        expected = reference.get("expected_model_component", "")
        return float(expected.lower() in prediction.lower()) if expected else 0.0

    @staticmethod
    def _metric_found_correct_model_type(prediction: Any, reference: Any, **kwargs) -> float:
        if not isinstance(reference, dict):
            return 0.0
        expected_type = reference.get("expected_model_type", "").lower()
        model_lower = prediction.lower()
        if expected_type == "thermal":
            return float(any(m in model_lower for m in ["bbody", "apec", "bremss", "diskbb"]))
        elif expected_type == "non-thermal":
            return float("powerlaw" in model_lower)
        return 0.0

    @staticmethod
    def _metric_models_explored(prediction: Any, reference: Any, **kwargs) -> float:
        return float(kwargs.get("models_explored", 0))

    # ------------------------------------------------------------------
    # Core fitting
    # ------------------------------------------------------------------

    def fit_model(
        self, model_str: str, params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run a Sherpa fit for *model_str* and return a results dict.

        Results dict keys:
            success (bool), cstat, dof, reduced_cstat, params (if success)
            or error (str) on failure.
        """
        from sherpa.astro import ui
        import json

        self.call_count += 1

        # Create cache key from model + params to allow same model with different params
        params_key = json.dumps(params, sort_keys=True) if params else "{}"
        cache_key = "{model}|{params}".format(model=model_str, params=params_key)

        if cache_key in self.fit_cache:
            return self.fit_cache[cache_key]

        try:
            ui.clean()
            ui.load_pha(self.pha_file)
            ui.ignore("0.:0.3,8.0:")
            ui.notice_id(1, 0.3, 7.0)
            ui.group_counts(1)

            ui.set_xsabund("angr")
            ui.set_xsxsect("vern")
            ui.set_xscosmo(h0=70, q0=0, l0=0.73)
            ui.set_syserror(0)

            ui.set_xsxsect("vern")
            ui.set_stat("cstat")
            ui.set_method("levmar")
            ui.set_analysis("ener")

            ui.set_source(model_str)

            if params:
                for param_name, param_spec in params.items():
                    try:
                        if isinstance(param_spec, dict):
                            if param_spec.get("val") is not None:
                                ui.set_par(param_name, val=param_spec["val"])
                            if param_spec.get("min") is not None:
                                ui.set_par(param_name, min=param_spec["min"])
                            if param_spec.get("max") is not None:
                                ui.set_par(param_name, max=param_spec["max"])
                        else:
                            ui.set_par(param_name, val=param_spec)
                    except Exception:
                        pass

            import warnings
            import io
            import sys

            # Capture warnings during fit
            warning_buffer = io.StringIO()
            old_stderr = sys.stderr
            sys.stderr = warning_buffer

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                ui.fit()

            sys.stderr = old_stderr
            stderr_output = warning_buffer.getvalue()

            fit_result = ui.get_fit_results()

            # Collect warnings
            fit_warnings = []
            for w in caught_warnings:
                fit_warnings.append(str(w.message))
            # Also capture stderr warnings (Sherpa prints some there)
            for line in stderr_output.strip().split("\n"):
                if line.startswith("WARNING:"):
                    fit_warnings.append(line.replace("WARNING: ", ""))

            # Compute AIC and BIC
            import math
            cstat = fit_result.statval
            n_free_params = fit_result.numpoints - fit_result.dof  # k
            n_data_points = fit_result.numpoints  # n
            
            aic = cstat + 2 * n_free_params
            bic = cstat + n_free_params * math.log(n_data_points)

            result = {
                "success": True,
                "cstat": round(fit_result.statval, 2),
                "dof": fit_result.dof,
                "reduced_cstat": round(fit_result.rstat, 4),
                "n_free_params": n_free_params,
                "n_data_points": n_data_points,
                "aic": round(aic, 2),
                "bic": round(bic, 2),
                "params": {
                    n: round(v, 6)
                    for n, v in zip(fit_result.parnames, fit_result.parvals)
                },
            }
            if fit_warnings:
                result["warnings"] = fit_warnings

        except Exception as e:
            result = {"success": False, "error": str(e)}

        # Store with model string as display key, but include initial params info
        result["model_str"] = model_str
        result["initial_params"] = params
        self.fit_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Observation summary
    # ------------------------------------------------------------------

    def get_observation_summary(self) -> Dict[str, Any]:
        """Build observation summary from metadata (excludes ground_truth)."""
        if self._observation is not None:
            return self._observation

        md = self.metadata
        obs = md.get("observation", {})
        src = md.get("source", {})
        sp = md.get("spectral_properties", {})
        tp = md.get("temporal_properties", {})
        en = md.get("energetics", {})
        pc = md.get("persistent_constraints", {})
        mw = md.get("multiwavelength", {})

        self._observation = {
            "telescope": obs.get("telescope", "Unknown"),
            "instrument": obs.get("instrument", "Unknown"),
            "energy_range_keV": obs.get("energy_range_keV", list(self.energy_range)),
            "exposure_ks": obs.get("exposure_ks"),
            "net_counts": obs.get("net_counts"),
            "sn_ratio": obs.get("sn_ratio"),
            "field": src.get("field"),
            "distance_kpc": src.get("distance_kpc"),
            "off_axis_arcmin": src.get("off_axis_arcmin"),
            "localization_uncertainty_arcsec": src.get("localization_uncertainty_arcsec"),
            "phenomenology": md.get("phenomenology", []),
            "persistent_constraints": pc,
            "temporal_properties": tp,
            "spectral_properties": sp,
            "energetics": en,
            "multiwavelength": mw,
        }
        return self._observation

    def reset(self):
        """Reset oracle state."""
        self.call_count = 0
        self.fit_cache.clear()
