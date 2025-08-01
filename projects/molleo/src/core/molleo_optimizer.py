"""Main MolLEO optimizer using SDE harness framework"""

import random
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rdkit import Chem
import sys
import os

# Add SDE harness to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from sde_harness.core import Generation, Workflow
from ..utils import (
    mol_to_smiles, 
    smiles_to_mol,
    make_mating_pool, 
    reproduce,
    get_best_mol
)
from ..oracles import MolecularOracle
from .prompts import MolecularPrompts


class MolLEOOptimizer:
    """
    MolLEO: LLM-augmented evolutionary algorithm for molecular discovery
    Integrated with SDE harness framework
    """
    
    def __init__(self, 
                 oracle: MolecularOracle,
                 population_size: int = 100,
                 offspring_size: int = 200,
                 mutation_rate: float = 0.01,
                 n_jobs: int = -1,
                 model_name: str = "openai/gpt-4o-2024-08-06",
                 use_llm_mutations: bool = True):
        """
        Initialize MolLEO optimizer
        
        Args:
            oracle: Molecular property oracle
            population_size: Size of population to maintain
            offspring_size: Number of offspring per generation
            mutation_rate: Probability of mutation
            n_jobs: Number of parallel jobs
            model_name: LLM model to use for mutations
            use_llm_mutations: Whether to use LLM for guided mutations
        """
        self.oracle = oracle
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.n_jobs = n_jobs
        self.model_name = model_name
        self.use_llm_mutations = use_llm_mutations
        
        # Initialize generation model
        self.generator = Generation(
            models_file=os.path.join(project_root, "models.yaml"),
            credentials_file=os.path.join(project_root, "credentials.yaml")
        )
        
        # Initialize population
        self.population_mol = []
        self.population_scores = []
        self.generation_count = 0
        self.all_results = {}
        
    def initialize_population(self, starting_smiles: List[str]):
        """Initialize population from starting molecules"""
        self.population_mol = []
        self.population_scores = []
        
        for smiles in starting_smiles:
            mol = smiles_to_mol(smiles)
            if mol is not None:
                score = self.oracle.evaluate_molecule(smiles)
                self.population_mol.append(mol)
                self.population_scores.append(score)
                self.all_results[smiles] = score
                
        # Fill remaining population with random mutations
        while len(self.population_mol) < self.population_size:
            parent_idx = random.randint(0, len(self.population_mol) - 1)
            parent_mol = self.population_mol[parent_idx]
            
            # Try mutation
            if self.use_llm_mutations:
                mutant = self._llm_mutate(parent_mol)
            else:
                mutant = self._random_mutate(parent_mol)
                
            if mutant is not None:
                mutant_smiles = mol_to_smiles(mutant)
                if mutant_smiles not in self.all_results:
                    score = self.oracle.evaluate_molecule(mutant_smiles)
                    self.population_mol.append(mutant)
                    self.population_scores.append(score)
                    self.all_results[mutant_smiles] = score
                    
    def _llm_mutate(self, parent_mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Use LLM to generate molecular mutations with context"""
        parent_smiles = mol_to_smiles(parent_mol)
        
        # If population is not yet established, use simple mutation prompt
        if not self.population_scores:
            prompt = MolecularPrompts.get_mutation_prompt(
                parent_smiles=parent_smiles,
                num_mutations=5
            )
        else:
            # Get top molecules from current population for context
            top_indices = np.argsort(self.population_scores)[-5:][::-1]
            context_molecules = []
            for idx in top_indices:
                mol = self.population_mol[idx]
                score = self.population_scores[idx]
                smiles = mol_to_smiles(mol)
                context_molecules.append(f"[{smiles}, {score:.4f}]")
            
            # Create context-aware prompt
            molecule_context = "\n".join(context_molecules)
            
            # Use optimization prompt for better context
            prompt = MolecularPrompts.get_optimization_prompt(
                molecule_data=molecule_context,
                target_property=self.oracle.property_name,
                best_score=max(self.population_scores),
                num_molecules=5
            )
        
        try:
            # Generate mutations using SDE harness Generation
            response = self.generator.generate(
                prompt=prompt.build(),
                model_name=self.model_name,
                temperature=0.8,
                max_tokens=300
            )
            
            # Parse response - handle various formats
            text = response['text'].strip()
            
            # Try to extract SMILES from various formats
            mutation_smiles = []
            
            # Check for boxed format (like original GPT4 implementation)
            boxed_matches = re.findall(r'\\box\{(.*?)\}', text)
            if boxed_matches:
                mutation_smiles.extend(boxed_matches)
            
            # Also check for plain SMILES on separate lines
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                # Basic SMILES pattern check
                if line and not line.startswith('#') and not ':' in line:
                    # Try to validate it's a SMILES
                    if smiles_to_mol(line) is not None:
                        mutation_smiles.append(line)
            
            # Try each mutation
            for smiles in mutation_smiles:
                smiles = smiles.strip()
                if smiles:
                    mol = smiles_to_mol(smiles)
                    if mol is not None:
                        return mol
                        
        except Exception as e:
            print(f"LLM mutation failed: {e}")
            
        # Fallback to random mutation
        return self._random_mutate(parent_mol)
        
    def _random_mutate(self, parent_mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Random molecular mutation using original MolLEO operations"""
        from src.ga import mutations as mu
        # Use original random mutation
        return mu.mutate(parent_mol, self.mutation_rate, mol_lm=None)
        
    def evolve_one_generation(self):
        """Evolve population for one generation"""
        self.generation_count += 1
        
        # Create mating pool
        mating_pool = make_mating_pool(
            self.population_mol,
            self.population_scores,
            self.offspring_size
        )
        
        # Generate offspring
        offspring_mol = []
        offspring_scores = []
        
        successful_reproductions = 0
        
        for i in range(self.offspring_size // 2):
            
            # Reproduce - always pass self as mol_lm, it will handle LLM vs random
            child1, child2 = reproduce(
                mating_pool, 
                self.mutation_rate,
                mol_lm=self if self.use_llm_mutations else None
            )
            
            # Evaluate offspring
            for child in [child1, child2]:
                if child is not None:
                    successful_reproductions += 1
                    child_smiles = mol_to_smiles(child)
                    
                    # Check if already evaluated
                    if child_smiles in self.all_results:
                        score = self.all_results[child_smiles]
                    else:
                        score = self.oracle.evaluate_molecule(child_smiles)
                        self.all_results[child_smiles] = score
                        
                    offspring_mol.append(child)
                    offspring_scores.append(score)
                else:
                    print(f"DEBUG: Child is None")
        
        print(f"DEBUG: Successful reproductions: {successful_reproductions}")
        print(f"DEBUG: Offspring generated: {len(offspring_mol)}")
        
        # Combine population and offspring
        all_mol = self.population_mol + offspring_mol
        all_scores = self.population_scores + offspring_scores
        
        # Select top molecules for next generation
        sorted_indices = np.argsort(all_scores)[::-1][:self.population_size]
        self.population_mol = [all_mol[i] for i in sorted_indices]
        self.population_scores = [all_scores[i] for i in sorted_indices]
        
    def optimize(self, 
                 starting_smiles: List[str],
                 num_generations: int = 20) -> Dict[str, Any]:
        """
        Run optimization
        
        Args:
            starting_smiles: Initial molecules
            num_generations: Number of generations to run
            
        Returns:
            Optimization results
        """
        # Initialize population
        self.initialize_population(starting_smiles)
        
        # Track best molecules
        best_scores = []
        best_molecules = []
        
        # Evolution loop
        for gen in range(num_generations):
            # Evolve
            self.evolve_one_generation()
            
            # Track best
            best_idx = np.argmax(self.population_scores)
            best_score = self.population_scores[best_idx]
            best_mol = self.population_mol[best_idx]
            best_smiles = mol_to_smiles(best_mol)
            
            best_scores.append(best_score)
            best_molecules.append(best_smiles)
            
            print(f"Generation {gen+1}: Best score = {best_score:.4f}, "
                  f"Oracle calls = {self.oracle.call_count}")
            
        # Get final results
        final_population = [
            (mol_to_smiles(mol), score) 
            for mol, score in zip(self.population_mol, self.population_scores)
        ]
        
        return {
            "best_molecule": best_molecules[-1],
            "best_score": best_scores[-1],
            "best_scores_history": best_scores,
            "best_molecules_history": best_molecules,
            "final_population": final_population,
            "oracle_calls": self.oracle.call_count,
            "all_results": self.all_results,
        }
    
    # Compatibility method for mutation
    def mutate(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Mutate molecule (for compatibility with original code)"""
        if self.use_llm_mutations:
            return self._llm_mutate(mol)
        else:
            return self._random_mutate(mol)