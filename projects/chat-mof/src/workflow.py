"""
MOF Optimization Workflow - orchestrates iterative MOF generation and evaluation
"""

import sys
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd

# Add sde_harness to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from sde_harness.core import Workflow
import weave

from .generation import MOFGeneration
from .prompt import MOFPrompt
from .oracle import MOFOracle


class MOFWorkflow(Workflow):
    """
    MOF optimization workflow using genetic algorithm principles.
    Iteratively generates and evaluates MOF candidates to find high surface area materials.
    """
    
    def __init__(
        self,
        generator: MOFGeneration,
        oracle: MOFOracle,
        prompt: MOFPrompt,
        # GA-inspired parameters
        population_size: int = 20,
        n_generations: int = 10,
        elitism_rate: float = 0.1,
        target_surface_area: float = 1000.0,
        # Workflow parameters
        model_name: Optional[str] = None,
        stop_early: bool = True,
        min_success_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize MOF optimization workflow.
        
        Args:
            generator: MOFGeneration instance
            oracle: MOFOracle instance  
            prompt: MOFPrompt instance
            population_size: Number of MOFs to generate per iteration
            n_generations: Maximum number of iterations
            elitism_rate: Fraction of best MOFs to keep for next iteration
            target_surface_area: Surface area threshold (m²/g)
            model_name: LLM model to use
            stop_early: Whether to stop if target is consistently met
            min_success_rate: Minimum success rate to consider stopping
            **kwargs: Additional workflow parameters
        """
        super().__init__(
            generator=generator.generator,  # Pass the underlying sde_harness generator
            oracle=oracle,
            max_iterations=n_generations,
            **kwargs
        )
        
        self.mof_generator = generator
        self.mof_oracle = oracle
        self.mof_prompt = prompt
        
        # GA parameters
        self.population_size = population_size
        self.n_generations = n_generations
        self.elitism_rate = elitism_rate
        self.n_elite = max(1, int(population_size * elitism_rate))
        self.target_surface_area = target_surface_area
        
        # Workflow parameters
        self.model_name = model_name
        self.stop_early = stop_early
        self.min_success_rate = min_success_rate
        
        # History tracking
        self.all_tested_mofs = []
        self.best_mofs = []
        self.generation_stats = []
        
    def _extract_mof_names_from_response(self, response: str) -> List[str]:
        """
        Extract MOF names from LLM response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            List of extracted MOF names
        """
        lines = response.strip().split('\n')
        mof_names = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove numbering, bullets, dashes
            line = re.sub(r'^\d+\.\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            
            # Skip obvious non-MOF lines
            if any(skip_word in line.lower() for skip_word in ['explanation', 'based on', 'analysis', 'note:', 'the']):
                continue
            
            if line and len(line) < 100:  # Reasonable MOF name length
                mof_names.append(line)
                
        return mof_names[:self.population_size]  # Limit to population size
    
    def _format_best_mofs_for_prompt(self, best_mofs: List[Dict[str, Any]], max_examples: int = 10) -> str:
        """
        Format best MOFs for inclusion in prompts.
        
        Args:
            best_mofs: List of best MOF results
            max_examples: Maximum number of examples to include
            
        Returns:
            Formatted string of best MOFs
        """
        if not best_mofs:
            return "No successful MOFs found yet."
            
        formatted = []
        for i, mof in enumerate(best_mofs[:max_examples]):
            surface_area = mof.get('surface_area', 'Unknown')
            name = mof.get('mof_name', 'Unknown')
            formatted.append(f"- {name}: {surface_area} m²/g")
            
        return "\n".join(formatted)
    
    def _format_recent_history_for_prompt(self, recent_mofs: List[Dict[str, Any]], max_recent: int = 20) -> str:
        """
        Format recent MOF history for prompts.
        
        Args:
            recent_mofs: Recent MOF evaluation results
            max_recent: Maximum number of recent MOFs to show
            
        Returns:
            Formatted string of recent history
        """
        if not recent_mofs:
            return "No recent history available."
            
        formatted = []
        for mof in recent_mofs[-max_recent:]:
            name = mof.get('mof_name', 'Unknown')
            found = mof.get('found', False)
            surface_area = mof.get('surface_area', None)
            
            if found and surface_area is not None:
                formatted.append(f"- {name}: {surface_area} m²/g ✓")
            else:
                formatted.append(f"- {name}: Not found in database ✗")
                
        return "\n".join(formatted)
    
    def _should_stop_early(self, generation: int) -> bool:
        """
        Determine if we should stop early based on success rate.
        
        Args:
            generation: Current generation number
            
        Returns:
            True if should stop early
        """
        if not self.stop_early or generation < 3:  # Need at least 3 generations
            return False
            
        # Check recent success rate
        recent_stats = self.generation_stats[-3:]  # Last 3 generations
        if len(recent_stats) < 3:
            return False
            
        avg_success_rate = sum(stats['success_rate'] for stats in recent_stats) / len(recent_stats)
        avg_best_surface_area = sum(stats['best_surface_area'] for stats in recent_stats if stats['best_surface_area'] > 0) / max(1, len([s for s in recent_stats if s['best_surface_area'] > 0]))
        
        # Stop if consistently finding good MOFs
        return (avg_success_rate >= self.min_success_rate and 
                avg_best_surface_area >= self.target_surface_area)
    
    @weave.op()
    def run_optimization(self) -> Dict[str, Any]:
        """
        Run the complete MOF optimization workflow.
        
        Returns:
            Dict containing optimization results and statistics
        """
        print(f"🚀 Starting MOF optimization for surface area > {self.target_surface_area} m²/g")
        print(f"Parameters: {self.population_size} MOFs/generation, {self.n_generations} generations max")
        
        # Initialize with some high surface area examples from database
        initial_examples = self.mof_oracle.get_high_surface_area_mofs(
            threshold=self.target_surface_area * 0.8,  # Slightly lower threshold for examples
            top_n=5
        )
        
        for generation in range(self.n_generations):
            print(f"\n--- Generation {generation + 1}/{self.n_generations} ---")
            
            # Build prompt based on current state
            if generation == 0:
                # Initial generation - use examples from database
                examples_text = self.mof_prompt.format_mof_examples([
                    {'name': mof.get(self.mof_oracle.mof_name_column, 'Unknown'), 
                     'surface_area': mof.get(self.mof_oracle.surface_area_column, 'Unknown')}
                    for mof in initial_examples
                ])
                
                prompt_text = self.mof_prompt.build_generation_prompt(
                    target_surface_area=self.target_surface_area,
                    num_samples=self.population_size,
                    examples=examples_text
                )
            else:
                # Iterative generation - use best MOFs found so far
                best_mofs_text = self._format_best_mofs_for_prompt(self.best_mofs)
                recent_history_text = self._format_recent_history_for_prompt(self.all_tested_mofs)
                
                prompt_text = self.mof_prompt.build_iterative_prompt(
                    current_iteration=generation + 1,
                    max_iterations=self.n_generations,
                    target_surface_area=self.target_surface_area,
                    num_samples=self.population_size,
                    best_mofs=best_mofs_text,
                    recent_history=recent_history_text
                )
            
            # Generate MOF candidates
            print(f"Generating {self.population_size} MOF candidates...")
            response = self.mof_generator.generate_mof_names(
                prompt=prompt_text,
                model_name=self.model_name,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract MOF names from response
            candidate_names = self._extract_mof_names_from_response(response['text'])
            print(f"Extracted {len(candidate_names)} MOF names: {candidate_names[:3]}...")
            
            # Evaluate candidates using Oracle
            print("Evaluating candidates...")
            evaluation_results = self.mof_oracle.evaluate_mof_candidates(
                candidate_names,
                threshold=self.target_surface_area
            )
            
            # Update history
            self.all_tested_mofs.extend(evaluation_results)
            
            # Find successful MOFs in this generation
            successful_mofs = [mof for mof in evaluation_results if mof['above_threshold']]
            found_mofs = [mof for mof in evaluation_results if mof['found']]
            
            # Update best MOFs list
            for mof in successful_mofs:
                self.best_mofs.append(mof)
            
            # Sort and keep only top performers
            self.best_mofs = sorted(
                self.best_mofs, 
                key=lambda x: x.get('surface_area', 0), 
                reverse=True
            )[:self.n_elite * 3]  # Keep more than just elite for diversity
            
            # Calculate statistics
            success_rate = len(successful_mofs) / len(evaluation_results)
            found_rate = len(found_mofs) / len(evaluation_results)
            
            # Handle None values in surface area calculations
            surface_areas = [mof.get('surface_area') for mof in evaluation_results if mof.get('surface_area') is not None]
            best_surface_area = max(surface_areas) if surface_areas else 0.0
            
            found_surface_areas = [mof.get('surface_area') for mof in found_mofs if mof.get('surface_area') is not None]
            avg_surface_area = sum(found_surface_areas) / len(found_surface_areas) if found_surface_areas else 0.0
            
            generation_stats = {
                'generation': generation + 1,
                'candidates_generated': len(candidate_names),
                'found_in_db': len(found_mofs),
                'above_threshold': len(successful_mofs),
                'success_rate': success_rate,
                'found_rate': found_rate,
                'best_surface_area': best_surface_area,
                'avg_surface_area': avg_surface_area
            }
            self.generation_stats.append(generation_stats)
            
            # Print generation summary
            print(f"Results: {len(found_mofs)}/{len(candidate_names)} found in DB, "
                  f"{len(successful_mofs)} above threshold")
            if successful_mofs:
                best_this_gen = max(successful_mofs, key=lambda x: x.get('surface_area', 0))
                print(f"Best this generation: {best_this_gen['mof_name']} "
                      f"({best_this_gen['surface_area']} m²/g)")
            
            # Check stopping criteria
            if self._should_stop_early(generation):
                print(f"🎯 Early stopping at generation {generation + 1} - consistent success!")
                break
        
        # Compile final results
        total_tested = len(self.all_tested_mofs)
        total_found = len([mof for mof in self.all_tested_mofs if mof['found']])
        total_successful = len([mof for mof in self.all_tested_mofs if mof['above_threshold']])
        
        final_results = {
            'optimization_complete': True,
            'generations_run': len(self.generation_stats),
            'total_mofs_tested': total_tested,
            'total_found_in_db': total_found,
            'total_above_threshold': total_successful,
            'overall_success_rate': total_successful / max(1, total_tested),
            'overall_found_rate': total_found / max(1, total_tested),
            'best_mofs': self.best_mofs[:10],  # Top 10
            'generation_stats': self.generation_stats,
            'target_surface_area': self.target_surface_area,
            'parameters': {
                'population_size': self.population_size,
                'n_generations': self.n_generations,
                'elitism_rate': self.elitism_rate,
                'model_name': self.model_name
            }
        }
        
        # Print final summary
        print(f"\n🏁 Optimization Complete!")
        print(f"Tested {total_tested} MOFs over {len(self.generation_stats)} generations")
        print(f"Found {total_successful} MOFs above {self.target_surface_area} m²/g threshold")
        print(f"Success rate: {final_results['overall_success_rate']:.1%}")
        
        if self.best_mofs:
            best_overall = self.best_mofs[0]
            print(f"Best MOF: {best_overall['mof_name']} ({best_overall['surface_area']} m²/g)")
        
        return final_results