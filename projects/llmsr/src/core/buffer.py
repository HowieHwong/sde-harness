# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-island experience buffer implementing the LLM-SR evolutionary search.

Direct port of the experience buffer used in the original LLM-SR code (and in
the LLM-SRBench / SDE-Harness paper runs):

* Programs are scored by the negative training MSE (higher is better).
* Inside an island, programs with identical scores are grouped in clusters.
* Clusters are sampled with a softmax whose temperature is annealed with the
  number of programs registered on the island.
* Inside a cluster, shorter programs are preferred.
* Every `reset_period` seconds the weaker half of the islands is reset and
  re-seeded with the best program of a randomly chosen surviving island.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Tuple

import numpy as np
import scipy.special

from . import code_manipulation


Signature = Tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Experience buffer hyper-parameters (defaults match the paper runs).

    Args:
        functions_per_prompt: Number of previous hypotheses included in a prompt.
        num_islands: Number of islands kept for diversity.
        reset_period: Seconds between resets of the weakest islands.
        cluster_sampling_temperature_init: Initial cluster softmax temperature.
        cluster_sampling_temperature_period: Period of the temperature decay.
    """
    functions_per_prompt: int = 2
    num_islands: int = 10
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """Represents test scores as a canonical signature."""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the experience buffer.

    Args:
        code: The prompt, ending with the header of the function to be completed.
        version_generated: The function to be completed is `_v{version_generated}`.
        island_id: Island that produced the samples in the prompt. New samples
            are registered back into the same island.
    """
    code: str
    version_generated: int
    island_id: int


class ExperienceBuffer:
    """A collection of programs, organized as islands."""

    def __init__(
            self,
            config: ExperienceBufferConfig,
            template: code_manipulation.Program,
            function_to_evolve: str,
    ) -> None:
        self._config = config
        self._template = template
        self._function_to_evolve = function_to_evolve

        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(template, function_to_evolve, config.functions_per_prompt,
                       config.cluster_sampling_temperature_init,
                       config.cluster_sampling_temperature_period))
        self._best_score_per_island: list[float] = [-float('inf')] * config.num_islands
        self._best_program_per_island: list[Optional[code_manipulation.Function]] = (
            [None] * config.num_islands)
        self._best_scores_per_test_per_island: list[Optional[ScoresPerTest]] = (
            [None] * config.num_islands)

        self._last_reset_time: float = time.time()

    def get_prompt(self) -> Prompt:
        """Returns a prompt containing samples from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)

    def _register_program_in_island(
            self,
            program: code_manipulation.Function,
            island_id: int,
            scores_per_test: ScoresPerTest,
    ) -> None:
        """Registers `program` in the specified island."""
        self._islands[island_id].register_program(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)

    def register_program(
            self,
            program: code_manipulation.Function,
            island_id: Optional[int],
            scores_per_test: ScoresPerTest,
    ) -> None:
        """Registers a new program; `island_id=None` registers it in every island."""
        if island_id is None:
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, scores_per_test)
        else:
            self._register_program_in_island(program, island_id, scores_per_test)

        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()

    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # Sort best scores after adding minor noise to break ties.
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period)
            self._best_score_per_island[island_id] = -float('inf')
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id, founder_scores)

    def get_statistics(self) -> dict:
        """Summary statistics of the buffer (for logging)."""
        return {
            "num_islands": len(self._islands),
            "best_score_per_island": list(self._best_score_per_island),
            "programs_per_island": [island.num_programs for island in self._islands],
            "clusters_per_island": [island.num_clusters for island in self._islands],
        }


class Island:
    """A sub-population of the program skeleton experience buffer."""

    def __init__(
            self,
            template: code_manipulation.Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
    ) -> None:
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._functions_per_prompt = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0

    @property
    def num_programs(self) -> int:
        return self._num_programs

    @property
    def num_clusters(self) -> int:
        return len(self._clusters)

    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = _get_signature(scores_per_test)
        if signature not in self._clusters:
            score = _reduce_score(scores_per_test)
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1

    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing equation program skeletons from this island."""
        signatures = list(self._clusters.keys())
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures])

        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        probabilities = _softmax(cluster_scores, temperature)

        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities)
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(
            self,
            implementations: Sequence[code_manipulation.Function]) -> str:
        """Create a prompt containing a sequence of function `implementations`."""
        implementations = copy.deepcopy(implementations)

        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            # Update the docstring for all subsequent functions after `_v0`.
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of `{self._function_to_evolve}_v{i - 1}`.')
            # If the function is recursive, replace calls to itself with its new name.
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(
                code_manipulation.text_to_function(implementation))

        # Create header of new function to be completed
        next_version = len(implementations)
        new_function_name = f'{self._function_to_evolve}_v{next_version}'
        header = dataclasses.replace(
            implementations[-1],
            name=new_function_name,
            body='',
            docstring=('Improved version of '
                       f'`{self._function_to_evolve}_v{next_version - 1}`.'),
        )
        versioned_functions.append(header)

        # Replace functions in the template with the list constructed here.
        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        return str(prompt)


class Cluster:
    """A cluster of programs on the same island and with the same Signature."""

    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        return self._score

    def register_program(self, program: code_manipulation.Function) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorter programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)
