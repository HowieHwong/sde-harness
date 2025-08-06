from __future__ import annotations

from typing import List, Any, Dict, Sequence
import random, numpy as np

from .protein_optimizer import ProteinOptimizer
from .pareto import non_dominated_sort


class ParetoOptimizer(ProteinOptimizer):
    """Genetic algorithm that keeps non-dominated front (NSGA-II simplified)."""

    def __init__(self, objectives: Sequence[object], *args, **kwargs):
        # objectives: list of oracle-like objects each having evaluate_molecule
        self._objectives = objectives
        super().__init__(oracle=None, *args, **kwargs)  # oracle None, we'll override eval

    # Override initialize_population and evaluation logic
    def _eval_sequence(self, seq: str) -> List[float]:
        return [oracle.evaluate_protein(seq) for oracle in self._objectives]

    def _initialize_population(self, starting_sequences: List[str]):
        self.population = []
        self.scores = []  # list of list
        self.all_results = {}
        for seq in starting_sequences:
            if seq not in self.all_results:
                sc = self._eval_sequence(seq)
                self.population.append(seq)
                self.scores.append(sc)
                self.all_results[seq] = sc
        while len(self.population) < self.population_size:
            parent = random.choice(self.population)
            mutant = self._random_mutate(parent)
            if mutant not in self.all_results:
                sc = self._eval_sequence(mutant)
                self.population.append(mutant)
                self.scores.append(sc)
                self.all_results[mutant] = sc

    def _evolve_one_generation(self):
        shifted = np.random.permutation(len(self.population))
        probs = np.full(len(self.population), 1 / len(self.population))
        offspring, offspring_scores = [], []
        for _ in range(self.offspring_size // 2):
            p1, p2 = np.random.choice(len(self.population), 2, replace=True, p=probs)
            c1, c2 = self._crossover(self.population[p1], self.population[p2])
            c1, c2 = self._random_mutate(c1), self._random_mutate(c2)
            for c in [c1, c2]:
                if c not in self.all_results:
                    sc = self._eval_sequence(c)
                    offspring.append(c)
                    offspring_scores.append(sc)
                    self.all_results[c] = sc
        # combine
        all_seq = self.population + offspring
        all_scores = self.scores + offspring_scores
        fronts = non_dominated_sort(all_scores)
        # keep non-dominated first; if fewer than pop size, fill random.
        new_pop, new_scores = [], []
        for idx in fronts:
            new_pop.append(all_seq[idx])
            new_scores.append(all_scores[idx])
            if len(new_pop) >= self.population_size:
                break
        # fill remaining randomly
        while len(new_pop) < self.population_size:
            r = random.choice(range(len(all_seq)))
            if all_seq[r] not in new_pop:
                new_pop.append(all_seq[r])
                new_scores.append(all_scores[r])
        self.population, self.scores = new_pop, new_scores 

    def optimize(self, starting_sequences: List[str], num_generations: int = 20) -> Dict[str, Any]:
        """Run the GA optimisation loop for Pareto optimization."""
        if not starting_sequences:
            raise ValueError("`starting_sequences` must contain at least one sequence.")

        self._initialize_population(starting_sequences)

        for gen in range(num_generations):
            self._evolve_one_generation()
            print(f"Generation {gen+1}: Population size = {len(self.population)}")

        # Final non-dominated sort to return only the Pareto front
        final_front_indices = non_dominated_sort(self.scores)
        pareto_front = [
            (self.population[i], self.scores[i]) for i in final_front_indices
        ]
        
        return {
            "pareto_front": pareto_front,
            "final_population": list(zip(self.population, self.scores)),
            "all_results": self.all_results,
        } 