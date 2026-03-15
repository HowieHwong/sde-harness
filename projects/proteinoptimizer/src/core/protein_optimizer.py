"""ProteinOptimizer: Simple GA for protein sequence optimization

This optimizer is tailored for fixed-length protein sequences such as those
found in the Syn-3bfo dataset. It supports basic evolutionary operations
(selection, crossover, mutation) and can optionally leverage an LLM-guided
mutation function if provided in the future.
"""

from __future__ import annotations

import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# 20 canonical amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


class ProteinOptimizer:
    """Genetic algorithm optimizer for protein sequences."""

    def __init__(
        self,
        oracle: ProteinOracle,
        population_size: int = 100,
        offspring_size: int = 200,
        mutation_rate: float = 0.3,
        random_seed: Optional[int] = None,
        model_name: Optional[str] = None,
        use_llm_mutations: bool = False,
    ) -> None:
        """Initialize the optimizer.

        Args:
            oracle: An oracle with an `evaluate_protein(sequence: str) -> float` method.
            population_size: Size of the population to maintain.
            offspring_size: Number of offspring produced per generation.
            mutation_rate: Per-position mutation rate.
            random_seed: Optional RNG seed for reproducibility.
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.oracle = oracle
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.use_llm_mutations = use_llm_mutations and model_name is not None
        self.model_name = model_name

        # Internal state
        self.population: List[str] = []
        self.scores: List[float] = []
        self.all_results: Dict[str, float] = {}
        self.generation_count: int = 0

        # LLM generator (lazy import)
        if self.use_llm_mutations:
            from sde_harness.core import Generation  # imported only when needed
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            self.generator = Generation(
                models_file=os.path.join(project_root, "models.yaml"),
                credentials_file=os.path.join(project_root, "credentials.yaml"),
            )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def optimize(
        self, 
        starting_sequences: List[str], 
        num_generations: int = 20,
        initial_population_with_scores: Optional[List[Tuple[str, float]]] = None,
        start_generation: int = 0
    ) -> Dict[str, Any]:
        """Run the GA optimisation loop.

        Args:
            starting_sequences: Initial population (used if initial_population_with_scores is None).
            num_generations: Number of generations to evolve.
            initial_population_with_scores: Optional pre-evaluated population as [(sequence, score), ...].
            start_generation: Starting generation number (for display).
        """
        # if starting_sequences != None:
        #     raise ValueError("`starting_sequences` must contain at least one sequence.")

        # Reset state
        self.generation_count = start_generation
        
        # Use pre-evaluated population if provided (for resume)
        if initial_population_with_scores is not None:
            self._load_population_from_scores(initial_population_with_scores)
        else:
            # Normal initialization
            # Ensure all sequences have the same length
            seq_len = len(starting_sequences[0])
            if any(len(s) != seq_len for s in starting_sequences):
                raise ValueError("All sequences must have the same length.")
            if hasattr(self.oracle, 'get_initial_population'):
                starting_sequences = self.oracle.get_initial_population(self.population_size)
            self._initialize_population(starting_sequences)
        

        best_scores: List[float] = []
        best_sequences: List[str] = []

        for gen in range(num_generations):
            self._evolve_one_generation()
            best_idx = int(np.argmax(self.scores))
            best_scores.append(self.scores[best_idx])
            best_sequences.append(self.population[best_idx])
            print(
                f"Generation {self.generation_count}: Best score = {best_scores[-1]:.4f}, "
                f"Oracle calls = {self.oracle.call_count}"
            )

        return {
            "best_sequence": best_sequences[-1],
            "best_score": best_scores[-1],
            "best_scores_history": best_scores,
            "best_sequences_history": best_sequences,
            "final_population": list(zip(self.population, self.scores)),
            "oracle_calls": self.oracle.call_count,
            "all_results": self.all_results,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialize_population(self, starting_sequences: List[str]):
        """Populate internal state from starting sequences, padding as needed."""
        seq_len = len(starting_sequences[0])
        self.population = []
        self.scores = []
        self.all_results = {}
        # Add starting sequences (deduplicated)
        for seq in starting_sequences:
            if seq not in self.all_results:
                score = self.oracle.evaluate_protein(seq)
                self.population.append(seq)
                self.scores.append(score)
                self.all_results[seq] = score

        # Fill the rest of the population with random mutations of starting sequences
        while len(self.population) < self.population_size:
            parent_seq = random.choice(self.population)
            mutant = self._random_mutate(parent_seq)
            if mutant not in self.all_results:
                score = self.oracle.evaluate_protein(mutant)
                self.population.append(mutant)
                self.scores.append(score)
                self.all_results[mutant] = score
    
    def _load_population_from_scores(self, population_with_scores: List[Tuple[str, float]]):
        """Load population from pre-evaluated sequences with scores (for resume).
        
        Args:
            population_with_scores: List of (sequence, score) tuples.
        """
        self.population = []
        self.scores = []
        self.all_results = {}
        
        # Load sequences and scores
        for seq, score in population_with_scores:
            self.population.append(seq)
            self.scores.append(score)
            self.all_results[seq] = score
        
        # Ensure we have enough sequences
        if len(self.population) < self.population_size:
            # Fill with random mutations if needed
            while len(self.population) < self.population_size:
                parent_seq = random.choice(self.population)
                mutant = self._random_mutate(parent_seq)
                if mutant not in self.all_results:
                    score = self.oracle.evaluate_protein(mutant)
                    self.population.append(mutant)
                    self.scores.append(score)
                    self.all_results[mutant] = score
                else:
                    # Use cached score
                    score = self.all_results[mutant]
                    self.population.append(mutant)
                    self.scores.append(score)
        
        # Trim to population_size if we have more
        if len(self.population) > self.population_size:
            # Sort by score and keep top N
            sorted_indices = np.argsort(self.scores)[::-1][:self.population_size]
            self.population = [self.population[i] for i in sorted_indices]
            self.scores = [self.scores[i] for i in sorted_indices]

    def _evolve_one_generation(self):
        self.generation_count += 1
        # Create mating pool via fitness-proportional (roulette wheel) selection
        shifted_scores = np.array(self.scores)
        min_score = float(shifted_scores.min())
        if min_score < 0:
            shifted_scores = shifted_scores - min_score + 1e-6
        else:
            shifted_scores = shifted_scores + 1e-6
        probs = shifted_scores / shifted_scores.sum()

        offspring: List[str] = []
        offspring_scores: List[float] = []
        step = 0

        while len(offspring) < self.offspring_size:
            if step >= 2*self.offspring_size:
                break
            step += 1
            # Select parents
            parents_idx = np.random.choice(len(self.population), size=2, p=probs, replace=True)
            parent_a, parent_b = parents_idx[0], parents_idx[1]
            # Crossover and mutation
            child = self._crossover(self.population[parent_a], self.population[parent_b])
            child = self._llm_mutate(parent_a, parent_b) if self.use_llm_mutations else self._random_mutate(child)
            # Evaluate and store
            if child not in self.all_results:
                score = self.oracle.evaluate_protein(child)
                offspring.append(child)
                offspring_scores.append(score)
                self.all_results[child] = score

        # Combine and select best individuals for next generation
        all_sequences = self.population + offspring
        all_scores = self.scores + offspring_scores
        top_indices = np.argsort(all_scores)[::-1][: self.population_size]
        self.population = [all_sequences[i] for i in top_indices]
        self.scores = [all_scores[i] for i in top_indices]

    # --------------- Evolutionary operators ---------------------------------
    def _crossover(self, seq1: str, seq2: str) -> tuple[str, str]:
        if len(seq1) != len(seq2):
            raise ValueError("Sequences must be of the same length for crossover.")
        
        # Randomly choose a crossover point
        point = random.randint(1, len(seq1) - 1)
        
        # Perform crossover
        new_seq1 = seq1[:point] + seq2[point:]
        # new_seq2 = seq2[:point] + seq1[point:]
        # new_seq1 = seq1
        # new_seq1 = seq1[:point] + seq2[point] + seq1[point+1:]
        return new_seq1


    def _random_mutate(self, sequence: str) -> str:
        """Random point mutations applied across the sequence length."""
        
        valid_chars = 'ACDEFGHIKLMNPQRSTVWY'  
        # print(mutation_rate)
        if random.random() < self.mutation_rate:
            return sequence
        sequence_list = list(sequence)
        char = random.choice(valid_chars)
        index = random.choice(range(len(sequence_list)))
        # for i in range(len(sequence_list)):
        #     sequence_list[i] = random.choice(valid_chars)
        sequence_list[index] = char
        return ''.join(sequence_list)

    # ------------------------------------------------------------------
    def _llm_mutate(self, idx_a: int , idx_b: int, k: int = 5) -> str:
        """Use LLM to suggest a mutant; fall back to random if fails."""
        seq = self.population[idx_a]
        if not self.use_llm_mutations:
            return self._random_mutate(self.population[idx_a])

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
            score_a = score_b = self.oracle.evaluate_protein(seq)

        pop_mean = statistics.mean(self.scores) if self.scores else 0.0
        pop_std = statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0

        prompt_obj = Prompt(template_name="protein_opt", default_vars=dict(
            task="maximise fitness",
            seq_len=len(seq),
            score1=score_a,
            score2=score_b,
            protein_seq_1=parent_a,
            protein_seq_2=parent_b,
            mean=pop_mean,
            std=pop_std,
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
                    # temperature=0.8,
                    # max_completion_tokens=len(seq) + 500,  # More room for box etc.
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
                print(retry_count)

            if generated_sequence:
                return generated_sequence

        except Exception:
            pass
        # fallback
        # If all LLM attempts fail, use crossover + mutation
        parent_a, parent_b = prompt_obj.default_vars["protein_seq_1"], prompt_obj.default_vars["protein_seq_2"]
        child = self._crossover(parent_a, parent_b)
        return self._random_mutate(child) 