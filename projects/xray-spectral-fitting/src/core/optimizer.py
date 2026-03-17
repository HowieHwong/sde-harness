"""
SpectralFitOptimizer – evolutionary optimizer for X-ray spectral model discovery.

Uses ``sde_harness.core.Generation`` for LLM calls and ``SherpaOracle`` for
spectral fitting.  Follows the population-based (offspring → keep-best)
pattern used by MolLEO.
"""

import json
import re
import os
from typing import Any, Dict, List, Optional, Tuple

from ..compat import Generation
from .prompts import SpectralPrompts
from ..oracle import SherpaOracle


def format_observation_summary(obs: Dict[str, Any]) -> str:
    """Format observation data into a human-readable string for prompts."""
    lines: List[str] = []

    # Detection
    lines.append("Detected with {telescope} ({instrument})".format(
        telescope=obs.get("telescope", "Unknown"),
        instrument=obs.get("instrument", "Unknown"),
    ))
    er = obs.get("energy_range_keV")
    if er:
        lines.append("Energy band: {lo}-{hi} keV".format(lo=er[0], hi=er[1]))
    if obs.get("exposure_ks"):
        lines.append("Observation exposure: {e} ks".format(e=obs["exposure_ks"]))
    if obs.get("off_axis_arcmin"):
        lines.append("Off-axis angle: {a}'".format(a=obs["off_axis_arcmin"]))
    if obs.get("localization_uncertainty_arcsec"):
        lines.append("Localization uncertainty: {u}\" (90%)".format(
            u=obs["localization_uncertainty_arcsec"]))
    if obs.get("net_counts"):
        lines.append("Net counts: ~{c}".format(c=obs["net_counts"]))
    if obs.get("sn_ratio"):
        lines.append("S/N: ~{s}".format(s=obs["sn_ratio"]))
    if obs.get("field"):
        lines.append("Field: {f}".format(f=obs["field"]))

    # Temporal properties
    tp = obs.get("temporal_properties", {})
    if tp:
        lines.append("")
        lines.append("Temporal properties:")
        spike = tp.get("initial_spike", {})
        if spike:
            lines.append("  Initial spike:")
            if spike.get("duration_s"):
                lines.append("    Duration: {v} s".format(v=spike["duration_s"]))
            if spike.get("rise_time_s"):
                lines.append("    Rise time: {v} s".format(v=spike["rise_time_s"]))
            if spike.get("count_rate_increase_orders"):
                lines.append("    Count rate increase: {v} orders of magnitude".format(
                    v=spike["count_rate_increase_orders"]))
            if spike.get("peak_count_rate_cts_s"):
                lines.append("    Peak count rate: {v}".format(
                    v=spike["peak_count_rate_cts_s"]))
            if spike.get("fraction_of_total_counts"):
                lines.append("    Fraction of total flare counts: ~{v:.0%}".format(
                    v=spike["fraction_of_total_counts"]))
        tail = tp.get("extended_tail", {})
        if tail:
            lines.append("  Extended tail:")
            if tail.get("duration_s"):
                lines.append("    Duration: ~{v} s".format(v=tail["duration_s"]))
            if tail.get("morphology"):
                lines.append("    Morphology: {v}".format(v=tail["morphology"]))
        if tp.get("overall"):
            lines.append("  Overall: {v}".format(v=tp["overall"]))

    # Spectral properties
    sp = obs.get("spectral_properties", {})
    if sp:
        lines.append("")
        lines.append("Spectral properties:")
        if sp.get("spectral_evolution"):
            lines.append("  {v}".format(v=sp["spectral_evolution"]))

    # Multiwavelength
    mw = obs.get("multiwavelength", {})
    if mw:
        lines.append("")
        lines.append("Multi-wavelength constraints:")
        if mw.get("gamma_ray"):
            lines.append("  Gamma-ray: {v}".format(v=mw["gamma_ray"]))
        if mw.get("gravitational_wave"):
            lines.append("  Gravitational wave: {v}".format(v=mw["gravitational_wave"]))
        if mw.get("optical_counterpart"):
            lines.append("  Optical: {v}".format(v=mw["optical_counterpart"]))
        od = mw.get("optical_depth_5sigma")
        if od:
            parts = ["{b}~{m}".format(b=b, m=m) for b, m in od.items()]
            lines.append("  Optical 5-sigma depth: {v}".format(v=", ".join(parts)))
        bc = mw.get("brightest_in_error_circle")
        if bc:
            lines.append("  Brightest source in error circle: m~{a} (M~+{ab} at assumed distance)".format(
                a=bc.get("apparent_mag", "?"), ab=bc.get("absolute_mag_at_50kpc", "?")))
        if mw.get("stellar_population"):
            lines.append("  Stellar population: {v}".format(v=mw["stellar_population"]))

    # Core phenomenology (summary at the end)
    phenom = obs.get("phenomenology", [])
    if phenom:
        lines.append("")
        lines.append("Core phenomenology:")
        for item in phenom:
            lines.append("  - {v}".format(v=item))

    return "\n".join(lines)


def format_population_summary(population: List[Tuple[str, Dict, str]]) -> str:
    """Format the current population for the iterative prompt."""
    if not population:
        return "No models fitted yet."
    lines: List[str] = []
    for i, (model, result, reasoning) in enumerate(population, 1):
        if result.get("success"):
            params_str = ", ".join(
                "{k}={v:.4f}".format(k=k, v=v) for k, v in result["params"].items()
            )
            lines.append("{i}. {model}".format(i=i, model=model))
            lines.append(
                "   C-stat: {cstat:.1f}/{dof} = {reduced:.3f}".format(
                    cstat=result["cstat"],
                    dof=result["dof"],
                    reduced=result["reduced_cstat"],
                )
            )
            lines.append("   Parameters: {p}".format(p=params_str))
        else:
            lines.append("{i}. {model}".format(i=i, model=model))
            lines.append("   Fit failed: {err}".format(err=result.get("error", "Unknown")))
        lines.append("")
    return "\n".join(lines)


def format_all_results_summary(fit_cache: Dict[str, Dict[str, Any]]) -> str:
    """Format full history of all attempted models for the iterative prompt."""
    if not fit_cache:
        return "No models tried yet."
    successes: List[str] = []
    failures: List[str] = []
    for cache_key, result in fit_cache.items():
        model_str = result.get("model_str", cache_key.split("|")[0])
        initial_params = result.get("initial_params")
        
        if result.get("success"):
            lines_for_model = []
            lines_for_model.append("  {m}".format(m=model_str))
            if initial_params:
                init_str = ", ".join("{k}={v}".format(k=k, v=v) for k, v in initial_params.items())
                lines_for_model.append("    Initial params: {p}".format(p=init_str))
            lines_for_model.append("    C-stat={c:.2f}  DOF={d}  Reduced C-stat={r:.4f}  BIC={bic:.2f}".format(
                c=result["cstat"], d=result["dof"], r=result["reduced_cstat"],
                bic=result.get("bic", 0),
            ))
            lines_for_model.append("    Fitted parameters:")
            for pname, pval in result["params"].items():
                lines_for_model.append("      {k} = {v:.6f}".format(k=pname, v=pval))
            # Add warnings if any
            warnings = result.get("warnings", [])
            if warnings:
                lines_for_model.append("    WARNINGS:")
                for w in warnings:
                    lines_for_model.append("      - {w}".format(w=w))
            successes.append("\n".join(lines_for_model))
        else:
            failures.append("  {m}  ERROR: {e}".format(
                m=model_str, e=result.get("error", "Unknown"),
            ))
    lines: List[str] = []
    if successes:
        lines.append("Previous fit results ({n}):".format(n=len(successes)))
        lines.append("")
        lines.extend(successes)
    if failures:
        if lines:
            lines.append("")
        lines.append("Failed fits ({n}):".format(n=len(failures)))
        lines.extend(failures)
    return "\n".join(lines)


def format_errors_summary(errors: List[Tuple[str, str]]) -> str:
    """Format errors from the last generation for the iterative prompt."""
    if not errors:
        return ""
    lines = []
    for model, error in errors:
        lines.append("- \"{m}\": {e}".format(m=model, e=error))
    return "\n".join(lines)


class SpectralFitOptimizer:
    """
    Evolutionary optimizer for X-ray spectral fitting.

    Each generation the LLM proposes ``offspring_size`` model hypotheses.
    The Sherpa oracle fits each one.  The best ``population_size`` models
    (ranked by reduced C-stat) are kept for the next round.
    """

    def __init__(
        self,
        oracle: SherpaOracle,
        population_size: int = 2,
        offspring_size: int = 4,
        model_name: str = "openai/gpt-4o-2024-08-06",
    ):
        self.oracle = oracle
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.model_name = model_name

        harness_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        )
        self.generator = Generation(
            models_file=os.path.join(harness_root, "config", "models.yaml"),
            credentials_file=os.path.join(harness_root, "config", "credentials.yaml"),
        )

        self.population: List[Tuple[str, Dict, str]] = []
        self.generation_count = 0
        self.last_errors: List[Tuple[str, str]] = []

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _parse_llm_response(self, text: str) -> List[Dict]:
        """Extract a JSON array of model hypotheses from an LLM response."""
        if not (text or "").strip():
            return []

        raw = text.strip()

        def try_parse(s: str) -> Optional[List[Dict]]:
            s = s.strip()
            if not s:
                return None
            # Allow trailing comma before ]
            s = re.sub(r",\s*\]", "]", s)
            s = re.sub(r",\s*}", "}", s)
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return parsed if parsed else None
                if isinstance(parsed, dict) and parsed.get("model"):
                    return [parsed]
            except json.JSONDecodeError:
                pass
            return None

        # 1. Try markdown code block (GPT-5 / reasoning models often wrap JSON)
        for block in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", raw):
            out = try_parse(block.group(1))
            if out:
                return out

        # 2. Find a balanced [...] (first [ to matching ])
        depth = 0
        start = None
        for i, c in enumerate(raw):
            if c == "[":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0 and start is not None:
                    out = try_parse(raw[start : i + 1])
                    if out:
                        return out
                    break

        # 3. Try first regex array (greedy)
        json_match = re.search(r"\[[\s\S]*\]", raw)
        if json_match:
            out = try_parse(json_match.group())
            if out:
                return out

        # 4. Try single object (balanced braces)
        start = raw.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        out = try_parse(raw[start : i + 1])
                        if out:
                            return out
                        break

        return []

    def _generate_hypotheses(self) -> List[Dict]:
        """Use the LLM to propose new model hypotheses."""
        obs_summary = format_observation_summary(self.oracle.get_observation_summary())

        if self.generation_count == 0:
            prompt = SpectralPrompts.get_initial_prompt(obs_summary, self.offspring_size)
        else:
            pop_summary = format_population_summary(self.population)
            all_results = format_all_results_summary(self.oracle.fit_cache)
            errors_summary = format_errors_summary(self.last_errors)
            prompt = SpectralPrompts.get_iterative_prompt(
                obs_summary, pop_summary, all_results, self.offspring_size,
                errors_summary=errors_summary,
            )

        response = self.generator.generate(
            prompt=prompt.build(),
            model_name=self.model_name,
            temperature=1.0,
            max_tokens=2000,
        )

        text = response.get("text", "")
        hypotheses = self._parse_llm_response(text)
        if not hypotheses and text:
            if getattr(self, "_verbose", False):
                print("  [DEBUG] LLM returned 0 hypotheses. Raw response (first 600 chars):")
                print("  ---")
                print(text[:600].replace("\n", "\n  "))
                print("  ---")
            # Save full response to file for inspection (e.g. GPT-5 format issues)
            try:
                debug_path = os.path.join(os.getcwd(), "last_llm_response_debug.txt")
                with open(debug_path, "w") as f:
                    f.write("=== Full LLM response (0 hypotheses parsed) ===\n\n")
                    f.write(text)
                if getattr(self, "_verbose", False):
                    print("  [DEBUG] Full response saved to: {p}".format(p=debug_path))
            except Exception:
                pass
        return hypotheses

    def _generate_summary(self, best_model: str, best_result: Dict, verbose: bool = False) -> str:
        """Generate a final summary explaining why the best model was selected."""
        if verbose:
            print("\n--- Generating Final Summary ---")

        obs = self.oracle.get_observation_summary()
        obs_summary = format_observation_summary(obs)
        all_results = format_all_results_summary(self.oracle.fit_cache)

        fitted_params_str = ", ".join(
            "{k}={v:.4f}".format(k=k, v=v) for k, v in best_result.get("params", {}).items()
        )

        prompt = SpectralPrompts.get_summary_prompt(
            observation_summary=obs_summary,
            all_results_summary=all_results,
            best_model=best_model,
            reduced_cstat=best_result.get("reduced_cstat", 0),
            bic=best_result.get("bic", 0),
            fitted_params=fitted_params_str,
        )

        response = self.generator.generate(
            prompt=prompt.build(),
            model_name=self.model_name,
            temperature=1.0,
            max_tokens=800,
        )

        summary = response["text"].strip()
        if verbose:
            print("  {s}".format(s=summary))
        return summary

    def _generate_classification(self, best_model: str, best_result: Dict, verbose: bool = False) -> Dict:
        """Ask the LLM to classify the source after fitting."""
        if verbose:
            print("\n--- Generating Source Classification ---")

        obs = self.oracle.get_observation_summary()
        obs_summary = format_observation_summary(obs)
        fitted_params_str = ", ".join(
            "{k}={v:.4f}".format(k=k, v=v) for k, v in best_result.get("params", {}).items()
        )

        prompt = SpectralPrompts.get_classification_prompt(
            observation_summary=obs_summary,
            best_model=best_model,
            reduced_cstat=best_result.get("reduced_cstat", 0),
            fitted_params=fitted_params_str,
        )

        response = self.generator.generate(
            prompt=prompt.build(),
            model_name=self.model_name,
            temperature=1.0,
            max_tokens=300,
        )

        text = response.get("text", "").strip()
        try:
            import re as _re
            match = _re.search(r'\{.*\}', text, _re.DOTALL)
            if match:
                data = json.loads(match.group())
                classification = data.get("classification", "").strip()
                reasoning = data.get("reasoning", "").strip()
                if verbose:
                    print("  Classification: {c}".format(c=classification))
                    print("  Reasoning: {r}".format(r=reasoning))
                return {"classification": classification, "reasoning": reasoning}
        except Exception:
            pass
        if verbose:
            print("  Could not parse classification response: {t}".format(t=text[:100]))
        return {"classification": text[:200], "reasoning": ""}

    # ------------------------------------------------------------------
    # Evolutionary loop
    # ------------------------------------------------------------------

    def evolve_one_generation(self) -> List[Tuple[str, Dict, str]]:
        """Generate hypotheses, fit them, and update the population."""
        self.generation_count += 1
        hypotheses = self._generate_hypotheses()

        offspring: List[Tuple[str, Dict, str]] = []
        self.last_errors = []
        for hyp in hypotheses:
            model_str = hyp.get("model", "")
            params = hyp.get("params", {})
            reasoning = hyp.get("reasoning", "")
            if not model_str:
                continue
            result = self.oracle.fit_model(model_str, params)
            offspring.append((model_str, result, reasoning))
            if not result.get("success"):
                self.last_errors.append((model_str, result.get("error", "Unknown error")))

        all_candidates = self.population + offspring

        def _sort_key(item):
            """Sort by BIC (lower = better). BIC penalizes complexity."""
            _, result, _ = item
            if result.get("success"):
                # Use BIC for ranking - balances fit quality vs model complexity
                return (0, result.get("bic", float("inf")))
            return (1, float("inf"))

        all_candidates.sort(key=_sort_key)
        self.population = all_candidates[: self.population_size]
        return offspring

    def optimize(
        self,
        max_generations: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the full evolutionary search for max_generations rounds."""
        if verbose:
            print("=" * 60)
            print("X-RAY SPECTRAL FITTING BENCHMARK")
            print(
                "Config: {off} -> {pop}".format(
                    off=self.offspring_size, pop=self.population_size
                )
            )
            print("Model: {m}".format(m=self.model_name))
            print("=" * 60)

        best_history: List[Tuple[str, Dict]] = []
        generations_log: List[Dict[str, Any]] = []
        classification_log: List[Dict[str, Any]] = []
        self._verbose = verbose

        for gen in range(max_generations):
            if verbose:
                print("\n--- Generation {g} ---".format(g=gen + 1))

            offspring = self.evolve_one_generation()

            for model_str, result, _ in offspring:
                generations_log.append({
                    "generation": gen + 1,
                    "model": model_str,
                    "reduced_cstat": result.get("reduced_cstat"),
                    "bic": result.get("bic"),
                    "success": result.get("success", False),
                    "cstat": result.get("cstat"),
                    "dof": result.get("dof"),
                    "n_free_params": result.get("n_free_params"),
                })

            if verbose:
                print("Generated {n} hypotheses".format(n=len(offspring)))
                for model, result, reasoning in offspring:
                    if result.get("success"):
                        print(
                            "  {m}: C-stat={c}/{d}={r:.3f}  BIC={bic:.1f}".format(
                                m=model,
                                c=result["cstat"],
                                d=result["dof"],
                                r=result["reduced_cstat"],
                                bic=result.get("bic", 0),
                            )
                        )
                        if reasoning:
                            print("    Reasoning: {r}".format(r=reasoning[:80]))
                    else:
                        print(
                            "  {m}: FAILED - {e}".format(
                                m=model, e=str(result.get("error", ""))[:50]
                            )
                        )
                        if reasoning:
                            print("    Reasoning: {r}".format(r=reasoning[:80]))

            if self.population:
                best_model, best_result, _ = self.population[0]
                best_history.append((best_model, best_result))

                # Classify once per generation using current best model
                gen_classification = None
                if best_result.get("success"):
                    gen_classification = self._generate_classification(best_model, best_result, verbose=False)
                    classification_log.append({
                        "generation": gen + 1,
                        "best_model": best_model,
                        "classification": gen_classification.get("classification", "") if gen_classification else "",
                        "reasoning": gen_classification.get("reasoning", "") if gen_classification else "",
                    })

                if verbose:
                    all_tried = [
                        (res.get("model_str", k.split("|")[0]), res)
                        for k, res in self.oracle.fit_cache.items()
                    ]
                    all_tried.sort(key=lambda x: (not x[1].get("success"), x[1].get("bic", float("inf"))))
                    print("\nAll tried models ({n} total, ranked by BIC):".format(n=len(all_tried)))
                    for i, (m, r) in enumerate(all_tried, 1):
                        if r.get("success"):
                            print("  {i}. {m}  reduced_cstat={r:.3f}  BIC={bic:.1f}".format(
                                i=i, m=m, r=r["reduced_cstat"], bic=r.get("bic", 0)))
                        else:
                            print("  {i}. {m}  FAILED".format(i=i, m=m))
                    if gen_classification:
                        print("  Classification: {c}".format(
                            c=gen_classification.get("classification", "N/A")))

        # Final classification = last generation's classification
        final_classification = classification_log[-1].get("classification", "") if classification_log else ""

        return {
            "best_model": self.population[0][0] if self.population else None,
            "best_result": self.population[0][1] if self.population else None,
            "best_reasoning": self.population[0][2] if self.population else None,
            "classification": final_classification,
            "classification_log": classification_log,  # per-generation classification history
            "final_population": [
                {"model": m, "result": r, "reasoning": reason}
                for m, r, reason in self.population
            ],
            "generations": self.generation_count,
            "oracle_calls": self.oracle.call_count,
            "best_history": [{"model": m, "result": r} for m, r in best_history],
            "all_results": self.oracle.fit_cache,
            "generations_log": generations_log,
        }
