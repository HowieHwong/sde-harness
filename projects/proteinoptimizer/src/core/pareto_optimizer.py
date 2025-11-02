from __future__ import annotations

from typing import List, Any, Dict, Sequence
import random, numpy as np

from .protein_optimizer import ProteinOptimizer
from .pareto import non_dominated_sort

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


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
        for _ in range(self.offspring_size):
            p1, p2 = np.random.choice(len(self.population), 2, replace=True, p=probs)
            child1 = self._llm_mutate(p1, p2) if self.use_llm_mutations else self._random_mutate(seq=self.population[p1])
            if child1 not in self.all_results:
                sc = self._eval_sequence(seq=child1)
                offspring.append(child1)
                offspring_scores.append(sc)
                self.all_results[child1] = sc
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
    def _llm_mutate(self, idx_a: int, idx_b: int, k: int = 5) -> str:
        """Use LLM to suggest a mutant; fall back to random if fails."""
        seq = self.population[idx_a]
        if not self.use_llm_mutations:
            return self._random_mutate(seq)

        prompt = (
            None  # will be built via Prompt below
        )
        # Build prompt
        from sde_harness.core.prompt import Prompt
        # Choose two random parents from population if available
        import random, statistics
        if len(self.population) >= 2:
            # idx_a, idx_b = random.sample(range(len(self.population)), 2)
            parent_a = self.population[idx_a]
            parent_b = self.population[idx_b]
            score_a = self.scores[idx_a]
            score_b = self.scores[idx_b]
        else:
            parent_a = parent_b = seq
            score_a = score_b = self.oracle.evaluate_molecule(seq)

        prompt_obj = Prompt(template_name="protein_opt", default_vars=dict(
            task="maximise fitness",
            seq_len=len(seq),
            score1=score_a,
            score2=score_b,
            protein_seq_1=parent_a,
            protein_seq_2=parent_b,
            mean=0.0,
            std=0.0
        ))

        prompt = prompt_obj.build()

        try:
            retry_count = 0
            max_retries = 5
            generated_sequence = None

            while retry_count < max_retries:
                response = self.generator.generate(
                    prompt=prompt,
                    model_name=self.model_name,
                    temperature=0.8,
                    max_tokens=len(seq) + 50,  # More room for box etc.
                )
                text = response["text"].strip()

                # 1. Look for \box{SEQUENCE}
                import re
                match = re.search(r'\\box\{(.*?)\}', text)
                if match:
                    candidate = match.group(1).replace(' ', '')
                    if len(candidate) == len(seq) and all(c in AMINO_ACIDS for c in candidate):
                        generated_sequence = candidate
                        break

                # 2. Fallback to parsing raw lines
                for line in text.split('\n'):
                    candidate = line.strip().upper()
                    if len(candidate) == len(seq) and all(c in AMINO_ACIDS for c in candidate):
                        generated_sequence = candidate
                        break
                if generated_sequence:
                    break
                retry_count += 1

            if generated_sequence:
                return generated_sequence

        except Exception:
            pass
        # fallback
        # If all LLM attempts fail, use crossover + mutation
        parent_a, parent_b = prompt_obj.default_vars["protein_seq_1"], prompt_obj.default_vars["protein_seq_2"]
        child = self._crossover(parent_a, parent_b)
        return self._random_mutate(child) 