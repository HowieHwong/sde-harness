"""Evolutionary Buffer for LLMSR - Multi-island storage with Boltzmann sampling."""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import math


@dataclass
class EquationProgram:
    """Represents an equation program with its metadata."""
    code: str
    nmse_score: float
    iteration: int
    island_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure NMSE score is finite
        if not np.isfinite(self.nmse_score):
            self.nmse_score = float('inf')


class Island:
    """An island in the evolutionary buffer containing equation programs."""
    
    def __init__(self, island_id: int, max_programs: int = 50):
        self.island_id = island_id
        self.max_programs = max_programs
        self.programs: List[EquationProgram] = []
        self.best_nmse = float('inf')
        self.best_program: Optional[EquationProgram] = None
    
    def add_program(self, program: EquationProgram) -> bool:
        """
        Add a program to this island.
        
        Args:
            program: The equation program to add
            
        Returns:
            True if program was added, False if it didn't improve the best score
        """
        # Check if this program improves the best NMSE
        improved = program.nmse_score < self.best_nmse
        
        if improved:
            self.best_nmse = program.nmse_score
            self.best_program = program
        
        # Add program to the list
        self.programs.append(program)
        
        # Keep only the best programs if we exceed max_programs
        if len(self.programs) > self.max_programs:
            # Sort by NMSE score (lower is better)
            self.programs.sort(key=lambda p: p.nmse_score)
            # Keep the best programs
            self.programs = self.programs[:self.max_programs]
        
        return improved
    
    def sample_programs(self, num_samples: int, temperature: float = 1.0) -> List[EquationProgram]:
        """
        Sample unique programs using Boltzmann distribution.

        Args:
            num_samples: Number of programs to sample
            temperature: Temperature for Boltzmann sampling (lower = more greedy)

        Returns:
            List of sampled unique programs
        """
        if not self.programs:
            return []

        # Remove duplicate programs based on code (or another unique identifier)
        # We'll use the 'code' attribute for uniqueness
        unique_programs_dict = {}
        for p in self.programs:
            # Use code as the unique key; you could use a hash or another identifier if needed
            unique_programs_dict[p.code] = p
        unique_programs = list(unique_programs_dict.values())

        if not unique_programs:
            return []

        # Convert NMSE scores to logits (lower NMSE = higher logit)
        nmse_scores = np.array([p.nmse_score for p in unique_programs])

        # Handle infinite scores
        finite_mask = np.isfinite(nmse_scores)
        if not np.any(finite_mask):
            return []

        # Use only finite scores for sampling
        finite_programs = [p for p, finite in zip(unique_programs, finite_mask) if finite]
        finite_scores = nmse_scores[finite_mask]

        # Convert to logits (negative NMSE so better scores have higher probability)
        logits = -finite_scores / temperature

        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        probabilities = exp_logits / np.sum(exp_logits)

        # Sample unique programs
        num_to_sample = min(num_samples, len(finite_programs))
        if num_to_sample == 0:
            return []

        sampled_indices = np.random.choice(
            len(finite_programs),
            size=num_to_sample,
            replace=False,
            p=probabilities
        )

        return [finite_programs[i] for i in sampled_indices]
    
    def get_best_programs(self, num_programs: int) -> List[EquationProgram]:
        """Get the best programs from this island."""
        sorted_programs = sorted(self.programs, key=lambda p: p.nmse_score)
        return sorted_programs[:num_programs]


class EvolutionaryBuffer:
    """Multi-island evolutionary buffer for equation programs."""
    
    def __init__(self, 
                 num_islands: int = 4,
                 max_programs_per_island: int = 50,
                 reset_period: int = 100,
                 reset_fraction: float = 0.5):
        """
        Initialize the evolutionary buffer.
        
        Args:
            num_islands: Number of islands to maintain
            max_programs_per_island: Maximum programs per island
            reset_period: How often to reset weaker islands
            reset_fraction: Fraction of islands to reset
        """
        self.num_islands = num_islands
        self.max_programs_per_island = max_programs_per_island
        self.reset_period = reset_period
        self.reset_fraction = reset_fraction
        
        # Initialize islands
        self.islands = [Island(i, max_programs_per_island) for i in range(num_islands)]
        
        # Global tracking
        self.global_best_nmse = float('inf')
        self.global_best_program: Optional[EquationProgram] = None
        self.total_programs_added = 0
        self.programs_improved = 0
        
        # Reset tracking
        self.last_reset_iteration = 0
    
    def add_program(self, 
                   code: str, 
                   nmse_score: float, 
                   iteration: int,
                   island_id: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a program to the buffer.
        
        Args:
            code: The equation code
            nmse_score: NMSE score (lower is better)
            iteration: Current iteration number
            island_id: Specific island to add to (None for random)
            metadata: Additional metadata
            
        Returns:
            True if this program improved the global best score
        """
        if not np.isfinite(nmse_score):
            return False
        
        # Choose island
        if island_id is None:
            island_id = random.randint(0, self.num_islands - 1)
        
        # Create program object
        program = EquationProgram(
            code=code,
            nmse_score=nmse_score,
            iteration=iteration,
            island_id=island_id,
            metadata=metadata or {}
        )
        
        # Add to island
        improved = self.islands[island_id].add_program(program)
        
        # Update global tracking
        self.total_programs_added += 1
        
        # Check if this improves global best
        if nmse_score < self.global_best_nmse:
            self.global_best_nmse = nmse_score
            self.global_best_program = program
            self.programs_improved += 1
            improved = True
        
        # Check if we should reset islands
        if iteration - self.last_reset_iteration >= self.reset_period:
            self._reset_weaker_islands()
            self.last_reset_iteration = iteration
        
        return improved
    
    def sample_examples(self, 
                       num_examples: int = 3, 
                       temperature: float = 1.0,
                       strategy: str = "boltzmann") -> List[EquationProgram]:
        """
        Sample example programs for inclusion in prompts.
        
        Args:
            num_examples: Number of examples to sample
            temperature: Temperature for Boltzmann sampling
            strategy: Sampling strategy ("boltzmann", "best", "random")
            
        Returns:
            List of sampled programs
        """
        if strategy == "best":
            return self._sample_best_examples(num_examples)
        elif strategy == "random":
            return self._sample_random_examples(num_examples)
        else:  # boltzmann
            return self._sample_boltzmann_examples(num_examples, temperature)
    
    def _sample_boltzmann_examples(self, num_examples: int, temperature: float) -> List[EquationProgram]:
        """Sample using Boltzmann distribution across all islands."""
        all_programs = []
        for island in self.islands:
            all_programs.extend(island.programs)
        
        if not all_programs:
            return []
        
        # Convert to numpy arrays for efficient computation
        nmse_scores = np.array([p.nmse_score for p in all_programs])
        
        # Handle infinite scores
        finite_mask = np.isfinite(nmse_scores)
        if not np.any(finite_mask):
            return []
        
        finite_programs = [p for p, finite in zip(all_programs, finite_mask) if finite]
        finite_scores = nmse_scores[finite_mask]
        
        # Convert to logits (negative NMSE so better scores have higher probability)
        logits = -finite_scores / temperature
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Sample programs
        num_to_sample = min(num_examples, len(finite_programs))
        if num_to_sample == 0:
            return []
        
        sampled_indices = np.random.choice(
            len(finite_programs),
            size=num_to_sample,
            replace=False,
            p=probabilities
        )
        
        return [finite_programs[i] for i in sampled_indices]
    
    def _sample_best_examples(self, num_examples: int) -> List[EquationProgram]:
        """Sample the best programs across all islands."""
        all_programs = []
        for island in self.islands:
            all_programs.extend(island.programs)
        
        # Sort by NMSE score and return best
        sorted_programs = sorted(all_programs, key=lambda p: p.nmse_score)
        return sorted_programs[:num_examples]
    
    def _sample_random_examples(self, num_examples: int) -> List[EquationProgram]:
        """Sample random programs across all islands."""
        all_programs = []
        for island in self.islands:
            all_programs.extend(island.programs)
        
        if not all_programs:
            return []
        
        num_to_sample = min(num_examples, len(all_programs))
        return random.sample(all_programs, num_to_sample)
    
    def _reset_weaker_islands(self):
        """Reset the weaker half of islands."""
        # Sort islands by their best NMSE score
        island_scores = [(i, island.best_nmse) for i, island in enumerate(self.islands)]
        island_scores.sort(key=lambda x: x[1])  # Sort by NMSE (lower is better)
        
        # Determine how many islands to reset
        num_to_reset = max(1, int(self.num_islands * self.reset_fraction))
        
        # Get the best islands to keep
        keep_islands = island_scores[:self.num_islands - num_to_reset]
        reset_islands = island_scores[self.num_islands - num_to_reset:]
        
        # Reset weaker islands
        for island_idx, _ in reset_islands:
            # Choose a founder from the best islands
            founder_island_idx = random.choice(keep_islands)[0]
            founder_island = self.islands[founder_island_idx]
            
            # Create new island with founder's best program
            new_island = Island(island_idx, self.max_programs_per_island)
            if founder_island.best_program:
                new_island.add_program(founder_island.best_program)
            
            self.islands[island_idx] = new_island
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the buffer."""
        total_programs = sum(len(island.programs) for island in self.islands)
        non_empty_islands = sum(1 for island in self.islands if island.programs)
        
        return {
            "total_programs": total_programs,
            "non_empty_islands": non_empty_islands,
            "global_best_nmse": self.global_best_nmse,
            "total_programs_added": self.total_programs_added,
            "programs_improved": self.programs_improved,
            "improvement_rate": self.programs_improved / max(1, self.total_programs_added),
            "island_stats": [
                {
                    "island_id": island.island_id,
                    "num_programs": len(island.programs),
                    "best_nmse": island.best_nmse
                }
                for island in self.islands
            ]
        }
    
    def clear(self):
        """Clear all programs from the buffer."""
        for island in self.islands:
            island.programs.clear()
            island.best_nmse = float('inf')
            island.best_program = None
        
        self.global_best_nmse = float('inf')
        self.global_best_program = None
        self.total_programs_added = 0
        self.programs_improved = 0
