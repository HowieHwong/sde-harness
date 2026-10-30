"""LLM-SR evolutionary equation discovery workflow.

Reproduces the search loop of the original LLM-SR implementation used for the
LLM-SRBench / SDE-Harness paper results:

1. The initial linear skeleton is evaluated and registered in every island.
2. Until `max_samples` samples have been drawn: an island is picked, a prompt
   with `functions_per_prompt` previous versions is built from it, the LLM is
   sampled `samples_per_prompt` times, and every valid sample is scored
   (-MSE on the training data after BFGS fitting) and registered back into the
   prompt's island. The search stops early if a perfect score (0) is found.
3. The best skeleton is re-fitted on the training data and evaluated on the ID
   and OOD test sets.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sde_harness.core import Oracle, Workflow

from ..core import code_manipulation
from ..core.buffer import ExperienceBuffer, ExperienceBufferConfig
from ..core.generation import LLMSRGeneration
from ..core.oracle import EquationOracle
from ..core.prompt import build_llm_prompt, build_specification

project_root2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root2 not in sys.path:
    sys.path.insert(0, project_root2)
from data.dataset import LLMSRDatasetLoader, EquationData


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """Return the names of the function to evolve and of the function to run."""
    run_functions = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@evaluate.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'equation', 'evolve'))
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@equation.evolve`.')
    return evolve_functions[0], run_functions[0]


def _to_jsonable(value: Any) -> Any:
    """Convert numpy scalars / NaN to JSON-friendly values."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class LLMSREvolutionaryWorkflow(Workflow):
    """LLM-SR evolutionary search on top of SDE-Harness components."""

    def __init__(self,
                 model_name: str = "openai/gpt-4o-2024-08-06",
                 models_file: Optional[str] = None,
                 credentials_file: Optional[str] = None,
                 max_samples: int = 1000,
                 samples_per_prompt: int = 8,
                 num_islands: int = 10,
                 functions_per_prompt: int = 2,
                 reset_period: int = 4 * 60 * 60,
                 cluster_sampling_temperature_init: float = 0.1,
                 cluster_sampling_temperature_period: int = 30_000,
                 evaluate_timeout_seconds: int = 30,
                 use_paper_descriptions: bool = True):
        """
        Args:
            model_name: Model key in models.yaml.
            models_file / credentials_file: SDE-Harness config files
                (default: `<sde-harness>/config/`).
            max_samples: Total LLM samples per problem (paper: 1000).
            samples_per_prompt: Samples drawn per prompt (paper: 8).
            num_islands: Islands in the experience buffer (paper: 10).
            functions_per_prompt: Previous versions included in each prompt.
            reset_period: Seconds between resets of the weaker islands.
            cluster_sampling_temperature_init / _period: Cluster sampling schedule.
            evaluate_timeout_seconds: Timeout for evaluating one sample.
            use_paper_descriptions: Use the variable descriptions of the paper runs.
        """
        generator = LLMSRGeneration(
            model_name=model_name,
            models_file=models_file,
            credentials_file=credentials_file,
            samples_per_prompt=samples_per_prompt,
        )
        # The equation oracle depends on the problem; it is created in discover_equation().
        super().__init__(
            generator=generator,
            oracle=Oracle(),
            max_iterations=max_samples,
            enable_history_in_prompts=False,
            enable_multi_round_metrics=False,
        )
        self.model_name = model_name
        self.max_samples = max_samples
        self.samples_per_prompt = samples_per_prompt
        self.buffer_config = ExperienceBufferConfig(
            functions_per_prompt=functions_per_prompt,
            num_islands=num_islands,
            reset_period=reset_period,
            cluster_sampling_temperature_init=cluster_sampling_temperature_init,
            cluster_sampling_temperature_period=cluster_sampling_temperature_period,
        )
        self.evaluate_timeout_seconds = evaluate_timeout_seconds
        self.use_paper_descriptions = use_paper_descriptions

        self.dataset_loader: Optional[LLMSRDatasetLoader] = None
        self.results: Dict[str, Dict[str, Any]] = {}

    def setup_dataset(self, dataset_name: str = "lsrtransform"):
        """Load the dataset."""
        self.dataset_loader = LLMSRDatasetLoader(dataset_name, self.use_paper_descriptions)
        self.dataset_loader.setup()

    # ------------------------------------------------------------------ search
    def discover_equation(self,
                          problem: Union[EquationData, str],
                          output_dir: str = "outputs") -> Dict[str, Any]:
        """Run LLM-SR on one problem and evaluate the best equation on the test sets."""
        if isinstance(problem, str):
            if self.dataset_loader is None:
                raise ValueError("Dataset not loaded. Call setup_dataset() first.")
            problem_name = problem
            problem = self.dataset_loader.get_problem_by_name(problem_name)
            if problem is None:
                raise ValueError(f"Problem '{problem_name}' not found in dataset.")

        var_names, var_descs = LLMSRDatasetLoader.get_variables(problem)
        specification, equation_specification = build_specification(var_names, var_descs)
        function_to_evolve, function_to_run = _extract_function_names(specification)
        template = code_manipulation.text_to_program(specification)
        train_data = LLMSRDatasetLoader.to_inputs_outputs(problem.train_data)

        buffer = ExperienceBuffer(self.buffer_config, template, function_to_evolve)
        self.oracle = EquationOracle(
            template, function_to_evolve, function_to_run, equation_specification,
            inputs={"data": train_data}, timeout_seconds=self.evaluate_timeout_seconds,
        )

        problem_dir = Path(output_dir) / problem.name
        problem_dir.mkdir(parents=True, exist_ok=True)
        samples_path = problem_dir / "samples.jsonl"
        samples_path.write_text("")

        print(f"Starting LLM-SR for problem {problem.name}: variables={var_names}")
        best = {"score": -float("inf"), "function": None, "sample_order": None}
        stats = {"valid": 0, "invalid": 0}

        def register(sample: str, island_id: Optional[int], version_generated: Optional[int],
                     sample_order: Optional[int], sample_time: Optional[float]) -> None:
            function, scores_per_test, evaluate_time = self.oracle.analyse(sample, version_generated)
            score = None
            if scores_per_test is not None:
                buffer.register_program(function, island_id, scores_per_test)
                score = sum(scores_per_test.values()) / len(scores_per_test)
                stats["valid"] += 1
                if score > best["score"]:
                    best.update(score=score, function=function, sample_order=sample_order)
            else:
                stats["invalid"] += 1
            with open(samples_path, "a") as f:
                f.write(json.dumps(_to_jsonable({
                    "sample_order": sample_order,
                    "island_id": island_id,
                    "score": score,
                    "best_score": best["score"],
                    "sample_time": sample_time,
                    "evaluate_time": evaluate_time,
                    "function": str(function),
                })) + "\n")

        start_time = time.time()
        # The initial skeleton seeds every island.
        initial = template.get_function(function_to_evolve).body
        register(initial, None, None, sample_order=None, sample_time=None)

        sample_order = 1
        while sample_order < self.max_samples:
            if best["score"] == 0:
                break
            prompt = buffer.get_prompt()
            prompt_text = build_llm_prompt(prompt.code).build()
            batch_start = time.time()
            samples = self.generator.draw_samples(prompt_text)
            sample_time = (time.time() - batch_start) / max(len(samples), 1)
            for sample in samples:
                sample_order += 1
                register(sample, prompt.island_id, prompt.version_generated, sample_order, sample_time)
            print(f"[{problem.name}] samples={sample_order - 1}/{self.max_samples} "
                  f"valid={stats['valid']} invalid={stats['invalid']} best_score={best['score']:.6g}")
        search_time = time.time() - start_time

        result = self._evaluate_best(problem, best, train_data, search_time)
        result["search_statistics"] = {**stats, **buffer.get_statistics()}

        with open(problem_dir / "result.json", "w") as f:
            json.dump(_to_jsonable(result), f, indent=2)
        with open(Path(output_dir) / "results.jsonl", "a") as f:
            f.write(json.dumps(_to_jsonable(result)) + "\n")

        self.results[problem.name] = result
        return result

    def _evaluate_best(self, problem: EquationData, best: Dict[str, Any],
                       train_data: Dict[str, Any], search_time: float) -> Dict[str, Any]:
        """Re-fit the best skeleton on the training data and evaluate on ID / OOD test data."""
        best_program = str(best["function"]).strip("\n") if best["function"] is not None else None
        result = {
            "equation_id": problem.name,
            "gt_equation": problem.expression,
            "model": self.model_name,
            "num_datapoints": len(problem.train_data),
            "num_eval_datapoints": len(problem.test_data) if problem.test_data is not None else 0,
            "search_time": search_time,
            "discovered_program": best_program,
            "best_program_sample_order": best["sample_order"],
            "best_program_score": best["score"],
            "optimized_params": None,
            "id_metrics": None,
            "ood_metrics": None,
        }
        if best_program is None:
            return result

        # Body = docstring + code of the best function (drop the `def` line).
        best_body = "\n".join(best_program.split("\n")[1:])
        params = self.oracle.fit_parameters(best_body, train_data)
        if params is None:
            print(f"[{problem.name}] Parameter fitting of the best program failed.")
            return result
        result["optimized_params"] = params
        result["id_metrics"] = self.oracle.evaluate_on_data(best_body, params, problem.test_data)
        result["ood_metrics"] = self.oracle.evaluate_on_data(best_body, params, problem.ood_test_data)
        print(f"[{problem.name}] ID metrics: {result['id_metrics']}")
        print(f"[{problem.name}] OOD metrics: {result['ood_metrics']}")
        return result

    def discover_all_equations(self,
                               output_dir: str = "outputs",
                               problem_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run LLM-SR on all (or filtered) problems; problems already in results.jsonl are skipped."""
        if self.dataset_loader is None:
            raise ValueError("Dataset not loaded. Call setup_dataset() first.")

        problems = self.dataset_loader.problems
        if problem_filter:
            problems = [p for p in problems if p.name in problem_filter]

        done = set()
        results_path = Path(output_dir) / "results.jsonl"
        if results_path.exists():
            with open(results_path) as f:
                done = {json.loads(line)["equation_id"] for line in f if line.strip()}

        print(f"Starting LLM-SR for {len(problems)} problems ({len(done)} already done)")
        all_results = {}
        for i, problem in enumerate(problems):
            if problem.name in done:
                print(f"Skipping {problem.name} (already in {results_path})")
                continue
            print(f"\nProcessing problem {i + 1}/{len(problems)}: {problem.name}")
            try:
                all_results[problem.name] = self.discover_equation(problem, output_dir)
            except Exception as e:
                print(f"Error processing problem {problem.name}: {e}")
                all_results[problem.name] = {"error": str(e)}
        return all_results

    def print_summary(self):
        """Print the best equation and test metrics of each problem."""
        if not self.results:
            print("No results available.")
            return
        print("\n" + "=" * 80)
        print("LLM-SR EQUATION DISCOVERY SUMMARY")
        print("=" * 80)
        for problem_name, result in self.results.items():
            print(f"\nProblem: {problem_name}")
            print(f"Best training score (-MSE): {result['best_program_score']}")
            for split in ("id_metrics", "ood_metrics"):
                metrics = result.get(split)
                if metrics:
                    print(f"{split}: NMSE={metrics['nmse']:.6g}  R2={metrics['r2']:.6g}  "
                          f"Acc_0.1={metrics['acc_0.1']}")
            print(f"Program:\n{result['discovered_program']}")
        print("\n" + "=" * 80)


# Backwards-compatible alias.
LLMSRWorkflow = LLMSREvolutionaryWorkflow
