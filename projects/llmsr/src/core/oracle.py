"""Evaluation of LLM-SR equation skeletons.

Two kinds of evaluation are provided, both matching the original LLM-SR
implementation used for the paper results:

1. Search-time fitness (`EquationOracle.analyse`): the sampled function body is
   inserted into the specification program and executed in a separate process
   with a timeout. Parameters are fitted with BFGS (initialized at 1.0) and the
   score is the negative training MSE.
2. Final evaluation (`EquationOracle.fit_and_predict` + `compute`): the best
   skeleton is re-fitted on the training data and evaluated on held-out ID /
   OOD test data with NMSE, R2, Kendall tau, MAPE and Acc_tau metrics.
"""

from __future__ import annotations

import ast
import copy
import ctypes
import multiprocessing
import time
from functools import reduce
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau

from sde_harness.core import Oracle

from . import code_manipulation
from .prompt import OPTIMIZATION_SPEC, EXECUTION_SPEC


class _FunctionLineVisitor(ast.NodeVisitor):
    """Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:
        """Collect the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """Extract the body of the generated function, trimming anything after it.

    Indentation is REQUIRED: the body must be indented as a function body.
    Lines from the first syntax error onwards are dropped.
    """
    if not generated_code:
        return ''

    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            if e.lineno is None:  # Nothing could be saved when syntaxError
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'


def sample_to_program(
        generated_code: str,
        version_generated: Optional[int],
        template: code_manipulation.Program,
        function_to_evolve: str,
) -> Tuple[code_manipulation.Function, str]:
    """Return the compiled generated function and the full runnable program."""
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    return evolved_function, str(program)


def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """Return whether the generated function is calling an earlier version."""
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False


class LocalSandbox:
    """Executes generated programs in a separate process with a timeout."""

    def run(self, program: str, function_to_run: str, inputs: Any, test_input: str,
            timeout_seconds: int, get_result_sleep_time: float = 0.1,
            result_array_shape: Optional[Tuple[int, ...]] = None) -> Tuple[Any, bool]:
        """Return `function_to_run(inputs[test_input])` and whether execution succeeded."""
        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue()

        if result_array_shape is not None:
            array_size = reduce(lambda x, y: x * y, result_array_shape, 1)
            shared_array = multiprocessing.RawArray(ctypes.c_double, array_size)
            shared_array_np = np.ndarray(result_array_shape, dtype=np.float64, buffer=shared_array)
        else:
            shared_array_np = None

        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, dataset, result_queue, shared_array_np)
        )
        process.start()
        process.join(timeout=timeout_seconds)

        # Terminate if the process did not finish in time.
        if process.is_alive():
            process.terminate()
            process.join()
            results = None, False
        else:
            results = self._get_results(result_queue, sleep_time=get_result_sleep_time)

        if shared_array_np is not None and results[1]:
            results = np.array(shared_array_np), True
        return results

    @staticmethod
    def _get_results(queue, sleep_time=0.1):
        for _ in range(5):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(sleep_time)
        return None, False

    @staticmethod
    def _compile_and_run_function(program, function_to_run, dataset, result_queue, shared_array_np=None):
        try:
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            results = function_to_run(dataset)

            if not isinstance(results, (int, float, list, tuple, np.ndarray)):
                result_queue.put((None, False))
                return

            if isinstance(results, list) and shared_array_np is not None:
                for i in range(len(results)):
                    shared_array_np[i] = results[i]
            # Arrays are returned through shared memory; only send small results.
            result_queue.put((None if shared_array_np is not None else results, True))
        except Exception as e:
            print(f"Execution Error: {e}")
            result_queue.put((None, False))


def acc_threshold(y_true: np.ndarray, y_pred: np.ndarray, rtol: float = 0.1,
                  atol: float = 0.0, tolerance_point: float = 0.95) -> float:
    """1.0 if at least `tolerance_point` of the points satisfy |y_pred - y| <= rtol * |y|."""
    if y_pred is None or len(y_pred) == 0 or np.isnan(np.min(y_pred)):
        return np.nan
    try:
        is_close = np.isclose(y_pred, y_true, atol=atol, rtol=rtol)
        return float(is_close.mean() >= tolerance_point)
    except Exception:
        return np.nan


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (same definition as scikit-learn)."""
    eps = np.finfo(np.float64).eps
    return float(np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps)))


class EquationOracle(Oracle):
    """Oracle for LLM-SR equation skeletons.

    Search-time scores are the negative training MSE after BFGS parameter
    fitting. Registered metrics (`mse`, `nmse`, `r2`, `kdt`, `mape`, `acc_0.1`,
    `acc_0.01`, `acc_0.001`) take `prediction=y_pred`, `reference=y_true`.
    """

    def __init__(
        self,
        template: code_manipulation.Program,
        function_to_evolve: str,
        function_to_run: str,
        equation_specification: str,
        inputs: Dict[str, Dict[str, Any]],
        timeout_seconds: int = 30,
    ):
        super().__init__()
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._equation_specification = equation_specification
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = LocalSandbox()

        self.register_metric("mse", lambda p, r, **kw: float(np.mean((r - p) ** 2)))
        self.register_metric("nmse", lambda p, r, **kw: float(np.mean((r - p) ** 2) / np.var(r)))
        self.register_metric("r2", lambda p, r, **kw: float(1 - np.sum((r - p) ** 2) / np.sum((r - r.mean()) ** 2)))
        self.register_metric("kdt", lambda p, r, **kw: float(kendalltau(r, p)[0]))
        self.register_metric("mape", lambda p, r, **kw: _mape(r, p))
        for tau in (0.1, 0.01, 0.001):
            self.register_metric(f"acc_{tau}", lambda p, r, tau=tau, **kw: acc_threshold(r, p, rtol=tau))

    # ------------------------------------------------------------------ search
    def analyse(self, sample: str, version_generated: Optional[int]
                ) -> Tuple[code_manipulation.Function, Optional[Dict[str, float]], float]:
        """Compile a sampled function body and score it on the training data.

        Returns (function, scores per input set or None if it failed, evaluation time).
        """
        new_function, program = sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)
        scores_per_test = {}
        start = time.time()
        for current_input in self._inputs:
            test_output, runs_ok = self._sandbox.run(
                program, self._function_to_run, self._inputs, current_input, self._timeout_seconds)
            if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and test_output is not None:
                if isinstance(test_output, (int, float)):
                    scores_per_test[current_input] = test_output
        evaluate_time = time.time() - start
        if not scores_per_test:
            return new_function, None, evaluate_time
        return new_function, scores_per_test, evaluate_time

    # ------------------------------------------------------- final evaluation
    def _program_with_spec(self, spec: str, function_body: str) -> str:
        template = code_manipulation.text_to_program(spec + self._equation_specification)
        _, program = sample_to_program(function_body, None, template, self._function_to_evolve)
        return program

    def fit_parameters(self, function_body: str, train_data: Dict[str, Any]) -> Optional[list]:
        """Fit the skeleton parameters with BFGS on `train_data`."""
        program = self._program_with_spec(OPTIMIZATION_SPEC, function_body)
        params, ok = self._sandbox.run(program, "optimize", {"data": train_data}, "data",
                                       self._timeout_seconds)
        return params if ok else None

    def predict(self, function_body: str, params: list, X: np.ndarray) -> Optional[np.ndarray]:
        """Evaluate the skeleton with fixed `params` on inputs `X` (n_points, n_inputs)."""
        program = self._program_with_spec(EXECUTION_SPEC.format(params=params), function_body)
        y_pred, ok = self._sandbox.run(
            program, "execute", {"data": [X[:, i] for i in range(X.shape[1])]}, "data",
            self._timeout_seconds, result_array_shape=(X.shape[0],))
        return y_pred if ok else None

    def evaluate_on_data(self, function_body: str, params: list, data: np.ndarray) -> Optional[Dict[str, float]]:
        """Paper metrics on `data` (first column = output). NaN predictions are dropped."""
        if data is None:
            return None
        y_true = data[:, 0]
        y_pred = self.predict(function_body, params, data[:, 1:])
        if y_pred is None:
            return None
        valid = ~np.isnan(y_pred)
        y_true, y_pred = y_true[valid], y_pred[valid]
        metrics = self.compute(y_pred, y_true)
        metrics["num_valid_points"] = int(valid.sum())
        return metrics
