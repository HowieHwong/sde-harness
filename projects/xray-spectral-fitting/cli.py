#!/usr/bin/env python3
"""
X-Ray Spectral Fitting – LLM-augmented evolutionary spectral model discovery.

Uses SDE-Harness framework: Generation for LLM, Oracle for Sherpa fitting,
Prompt for template management.

Example usage:
    python cli.py fit --pha data/spectra/lmc_flare/flaresp_grp1.pha --model openai/gpt-4o-2024-08-06 -v
    python cli.py list
"""

import argparse
import json
import os
import sys

# Ensure the project root (sde-harness/) is on PYTHONPATH
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.oracle import SherpaOracle
from src.core import SpectralFitOptimizer
from src.evaluator import evaluate_results, load_spectrum_metadata


# ---- subcommands --------------------------------------------------------

def cmd_fit(args: argparse.Namespace) -> int:
    """Run the evolutionary spectral fitting benchmark."""
    metadata = load_spectrum_metadata(args.pha)
    if metadata:
        ground_truth = metadata.get("ground_truth", {})
        if args.verbose:
            print("Spectrum: {desc}".format(desc=metadata.get("description", "Unknown")))
            print("Ground truth evaluation: enabled")
    else:
        ground_truth = {}
        if args.verbose:
            print("No metadata.json found – running without ground truth evaluation")

    for seed in args.seed:
        print("\n" + "=" * 60)
        print("Running with seed {s}".format(s=seed))
        print("=" * 60 + "\n")

        oracle = SherpaOracle(
            pha_file=args.pha,
            energy_range=(args.emin, args.emax),
            metadata=metadata,
        )

        optimizer = SpectralFitOptimizer(
            oracle=oracle,
            population_size=args.population_size,
            offspring_size=args.offspring_size,
            model_name=args.model,
        )

        results = optimizer.optimize(
            max_generations=args.generations,
            convergence_rounds=args.convergence,
            verbose=args.verbose,
        )

        metrics = evaluate_results(results, ground_truth)

        if args.verbose:
            # Print ranking of all model fits (sorted by BIC)
            print("\n--- Model Fit Ranking (by BIC) ---")
            all_results = results.get("all_results", {})
            ranked = []
            for cache_key, res in all_results.items():
                if res.get("success"):
                    model_str = res.get("model_str", cache_key.split("|")[0])
                    ranked.append((model_str, res.get("bic", float("inf")), res["reduced_cstat"], res.get("n_free_params", 0), res.get("initial_params")))
            ranked.sort(key=lambda x: x[1])  # Sort by BIC
            for i, (model, bic, rcstat, n_params, init_params) in enumerate(ranked, 1):
                print("  {rank}. {model}".format(rank=i, model=model))
                if init_params:
                    init_str = ", ".join("{k}={v}".format(k=k, v=v) for k, v in init_params.items())
                    print("      Initial: {p}".format(p=init_str[:60]))
                print("      BIC={bic:.2f}  reduced_cstat={rcstat:.4f}  params={n}".format(
                    bic=bic, rcstat=rcstat, n=n_params
                ))

            # Print failed models
            failed = [(k, r) for k, r in all_results.items() if not r.get("success")]
            if failed:
                print("\n  Failed models:")
                for cache_key, res in failed:
                    model_str = res.get("model_str", cache_key.split("|")[0])
                    err = res.get("error", "unknown error")
                    print("    - {m}: {e}".format(m=model_str, e=err[:60]))

            # Show best fit results
            best_result = results.get("best_result", {})
            best_reasoning = results.get("best_reasoning", "")
            if best_result and best_result.get("success"):
                print("\n--- Best Fit Results ---")
                print("  Model: {m}".format(m=metrics.get("best_model", "N/A")))
                if best_reasoning:
                    print("  Reasoning: {r}".format(r=best_reasoning))
                print("  C-stat: {c}  DOF: {d}  Reduced C-stat: {r:.4f}".format(
                    c=best_result.get("cstat"),
                    d=best_result.get("dof"),
                    r=best_result.get("reduced_cstat", 0),
                ))
                print("  BIC: {bic:.2f}  (free params: {k})".format(
                    bic=best_result.get("bic", 0),
                    k=best_result.get("n_free_params", 0),
                ))
                print("  Distance from optimal (|rcstat - 1|): {:.4f}".format(
                    metrics.get("cstat_vs_optimal", 0)
                ))
                print("  Fitted parameters:")
                fitted_params = best_result.get("params", {})
                for pname, pval in fitted_params.items():
                    print("    {n}: {v:.6f}".format(n=pname, v=pval))

            print("\n--- Evaluation Summary ---")
            print("  found_expected_model: {v}".format(v=metrics.get("found_expected_model")))
            print("  best_reduced_cstat: {v}".format(v=metrics.get("best_reduced_cstat")))
            print("  cstat_vs_optimal: {v:.4f}".format(v=metrics.get("cstat_vs_optimal", 0)))

            if ground_truth:
                print("\n--- Ground Truth Comparison ---")
                expected_model = ground_truth.get("expected_model_sherpa", "N/A")
                print("  Expected model: {exp}  (or equivalent, e.g. xsbbodyrad = xsbbody)".format(exp=expected_model))
                print("  Found model:    {found}".format(found=metrics.get("best_model", "N/A")))
                if metrics.get("found_expected_model"):
                    print("  Model STATUS: MATCH")
                else:
                    print("  Model STATUS: MISMATCH")

                # Key parameter checks (kT for thermal, PhoIndex/Gamma for powerlaw)
                fitted_params = best_result.get("params", {}) if best_result else {}
                
                print("\n--- Key Parameter Checks ---")
                # Check for kT (thermal models)
                kt_val = None
                for pname, pval in fitted_params.items():
                    if pname.lower().endswith(".kt"):
                        kt_val = pval
                        metrics["kt_value"] = kt_val
                        print("  kT = {:.3f} keV  (ground truth: 1.85 keV, range: 1.75-1.95)".format(kt_val))
                        if 1.75 <= kt_val <= 1.95:
                            print("    RESULT: CORRECT")
                            metrics["kt_correct"] = True
                        else:
                            print("    RESULT: INCORRECT")
                            metrics["kt_correct"] = False
                        break
                
                # Check for PhoIndex/Gamma (powerlaw models)
                gamma_val = None
                for pname, pval in fitted_params.items():
                    pname_lower = pname.lower()
                    if "phoindex" in pname_lower or "gamma" in pname_lower:
                        gamma_val = pval
                        print("  Gamma/PhoIndex = {:.3f}  (ground truth: ~0.5, range: 0.2-0.8)".format(gamma_val))
                        if 0.2 <= gamma_val <= 0.8:
                            print("    RESULT: CORRECT")
                            metrics["gamma_correct"] = True
                        else:
                            print("    RESULT: INCORRECT")
                            metrics["gamma_correct"] = False
                        break
                
                if kt_val is None and gamma_val is None:
                    print("  No kT or Gamma parameter found in best model")

                # Final summary
                print("\n" + "=" * 50)
                print("FINAL BENCHMARK RESULTS")
                print("=" * 50)
                
                print("  Best model: {m}".format(m=metrics.get("best_model", "N/A")))
                if best_reasoning:
                    print("  LLM reasoning: {r}".format(r=best_reasoning))
                
                # Show key fitted parameters
                if fitted_params:
                    kt_found = None
                    gamma_found = None
                    for pname, pval in fitted_params.items():
                        if pname.lower().endswith(".kt"):
                            kt_found = pval
                        if "phoindex" in pname.lower() or "gamma" in pname.lower():
                            gamma_found = pval
                    if kt_found is not None:
                        print("  Fitted kT: {:.3f} keV".format(kt_found))
                    if gamma_found is not None:
                        print("  Fitted Gamma: {:.3f}".format(gamma_found))
                print("")
                
                # Display final LLM summary
                final_summary = results.get("final_summary")
                if final_summary:
                    print("  Final Summary (LLM):")
                    for line in final_summary.split(". "):
                        if line.strip():
                            print("    {l}.".format(l=line.strip().rstrip(".")))
                    print("")
                
                good_fit = metrics.get("cstat_vs_optimal", 1.0) <= 0.05
                correct_model = metrics.get("found_expected_model", False)
                correct_kt = metrics.get("kt_correct", False)
                
                results_achieved = sum([good_fit, correct_model, correct_kt])
                
                best_rcstat = best_result.get("reduced_cstat", 0) if best_result else 0
                best_bic = best_result.get("bic", 0) if best_result else 0
                print("  [{}] Good fit (|reduced_cstat - 1| <= 0.05): rcstat={:.3f}, BIC={:.1f}".format(
                    "PASS" if good_fit else "FAIL",
                    best_rcstat,
                    best_bic
                ))
                print("  [{}] Correct model expression: {}".format(
                    "PASS" if correct_model else "FAIL",
                    metrics.get("best_model", "N/A")
                ))
                print("  [{}] Correct kT (1.75-1.95 keV): {}".format(
                    "PASS" if correct_kt else "FAIL",
                    "{:.3f} keV".format(metrics.get("kt_value", 0)) if metrics.get("kt_value") else "N/A"
                ))
                print("-" * 50)
                print("  SCORE: {}/3 criteria met".format(results_achieved))
                if results_achieved == 3:
                    print("  STATUS: SUCCESS")
                elif results_achieved >= 2:
                    print("  STATUS: PARTIAL SUCCESS")
                else:
                    print("  STATUS: FAILED")
                print("=" * 50)
                
                metrics["final_score"] = results_achieved
                metrics["final_status"] = "success" if results_achieved == 3 else ("partial" if results_achieved >= 2 else "failed")

        if args.output:
            output_data = {
                "results": results,
                "metrics": metrics,
                "ground_truth": ground_truth,
                "metadata": metadata,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            print("\nSaved to {o}".format(o=args.output))

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List available PHA spectra in the data directory."""
    data_dir = os.path.join(os.path.dirname(__file__), "data", "spectra")
    if not os.path.exists(data_dir):
        print("Data directory not found: {d}".format(d=data_dir))
        return 1

    print("Available spectra:")
    for root, _dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".pha"):
                rel = os.path.relpath(os.path.join(root, fname), os.path.dirname(__file__))
                print("  {r}".format(r=rel))
    return 0


# ---- argument parser ----------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="xray-spectral-fitting",
        description="LLM-augmented evolutionary algorithm for X-ray spectral model discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run spectral fitting with GPT-4o
  python cli.py fit --pha data/spectra/lmc_flare/flaresp_grp1.pha -v

  # Use a different model
  python cli.py fit --pha data/spectra/lmc_flare/flaresp_grp1.pha --model anthropic/claude-3-7-sonnet-20250219 -v

  # Custom population (6->3 instead of 4->2)
  python cli.py fit --pha data/spectra/lmc_flare/flaresp_grp1.pha --population-size 3 --offspring-size 6

  # List available spectra
  python cli.py list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- fit ----
    fit_p = subparsers.add_parser("fit", help="Run evolutionary spectral fitting")
    fit_p.add_argument("--pha", type=str, required=True, help="Path to PHA spectrum file")
    fit_p.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-2024-08-06",
        help="LLM model name as defined in config/models.yaml (default: openai/gpt-4o-2024-08-06)",
    )
    fit_p.add_argument(
        "--population-size", type=int, default=2,
        help="Models to keep per generation (default: 2)",
    )
    fit_p.add_argument(
        "--offspring-size", type=int, default=4,
        help="Hypotheses generated per generation (default: 4)",
    )
    fit_p.add_argument(
        "--generations", type=int, default=10, help="Max generations (default: 10)"
    )
    fit_p.add_argument(
        "--convergence", type=int, default=3,
        help="Stop after N unchanged rounds (default: 3)",
    )
    fit_p.add_argument(
        "--emin", type=float, default=0.3, help="Min energy in keV (default: 0.3)"
    )
    fit_p.add_argument(
        "--emax", type=float, default=7.0, help="Max energy in keV (default: 7.0)"
    )
    fit_p.add_argument(
        "--seed", type=int, nargs="+", default=[0], help="Random seed(s) (default: 0)"
    )
    fit_p.add_argument("-o", "--output", type=str, help="Output JSON file")
    fit_p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    fit_p.set_defaults(func=cmd_fit)

    # ---- list ----
    list_p = subparsers.add_parser("list", help="List available spectra")
    list_p.set_defaults(func=cmd_list)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
