"""Prompts for molecular generation and optimization"""

import sys
import os

# Add SDE harness to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from sde_harness.core import Prompt


class MolecularPrompts:
    """Collection of prompts for molecular tasks"""
    
    # Mutation prompt template
    MUTATION_PROMPT = """You are a molecular designer tasked with creating chemical analogs.

Given the following molecule:
SMILES: {parent_smiles}

Please generate {num_mutations} chemical analogs by making small structural modifications.
Consider the following types of modifications:
- Add/remove functional groups
- Replace atoms (e.g., C->N, O->S)
- Change ring sizes
- Add/remove rings
- Modify side chains

Requirements:
1. Each analog should be chemically valid
2. Maintain similar molecular weight (±50 Da)
3. Keep modifications reasonable and drug-like

Output format:
Return only the SMILES strings, one per line, without any additional text or explanations.
"""

    # Property optimization prompt
    OPTIMIZATION_PROMPT = """You are an expert medicinal chemist optimizing molecules for specific properties.

Current best molecules and their scores:
{molecule_data}

Target property: {target_property}
Current best score: {best_score}

Based on the structure-activity relationships observed, generate {num_molecules} new molecules that might have improved {target_property}.

Consider:
1. What structural features correlate with high scores?
2. How can you combine or modify these features?
3. What novel modifications might improve the property?

Requirements:
- The new molecules should be drug-like and synthetically accessible
- Each molecule should be structurally related to the parent molecules
- Focus on improving the {target_property} score

Output format:
Return only valid SMILES strings, one per line, without explanations.
If you want to highlight a particularly promising molecule, you can use: \box{{SMILES_HERE}}
"""

    # Multi-objective optimization prompt
    MULTI_OBJECTIVE_PROMPT = """You are optimizing molecules for multiple properties simultaneously.

Current population:
{population_data}

Target properties and their importance:
{objectives}

Generate {num_molecules} new molecules that balance all objectives.
Consider trade-offs between properties and aim for Pareto-optimal solutions.

Output format:
Return only valid SMILES strings, one per line.
"""

    # Analog generation prompt
    ANALOG_GENERATION_PROMPT = """You are an expert medicinal chemist. Generate {num_analogs} structural analogs of the following molecule that are optimized for {objective}.

Input molecule: {molecule}

Guidelines:
- Generate molecules with similar core structure but varied functional groups
- Ensure all molecules are valid SMILES strings
- Focus on modifications that would improve the objective
- Each analog should be chemically reasonable and synthetically accessible
- Output ONLY the SMILES strings, one per line, nothing else

Analogs:"""

    @staticmethod
    def get_mutation_prompt(parent_smiles: str, num_mutations: int = 5) -> Prompt:
        """Create mutation prompt"""
        return Prompt(
            custom_template=MolecularPrompts.MUTATION_PROMPT,
            default_vars={
                "parent_smiles": parent_smiles,
                "num_mutations": num_mutations
            }
        )
    
    @staticmethod
    def get_optimization_prompt(molecule_data: str, 
                              target_property: str,
                              best_score: float,
                              num_molecules: int = 10) -> Prompt:
        """Create optimization prompt"""
        return Prompt(
            custom_template=MolecularPrompts.OPTIMIZATION_PROMPT,
            default_vars={
                "molecule_data": molecule_data,
                "target_property": target_property,
                "best_score": best_score,
                "num_molecules": num_molecules
            }
        )
    
    @staticmethod
    def get_multi_objective_prompt(population_data: str,
                                  objectives: str,
                                  num_molecules: int = 10) -> Prompt:
        """Create multi-objective optimization prompt"""
        return Prompt(
            custom_template=MolecularPrompts.MULTI_OBJECTIVE_PROMPT,
            default_vars={
                "population_data": population_data,
                "objectives": objectives,
                "num_molecules": num_molecules
            }
        )