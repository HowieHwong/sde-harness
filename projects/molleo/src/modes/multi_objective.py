"""Multi-objective optimization mode for MolLEO"""

import sys
import os
from typing import Dict, Any, List, Tuple
import numpy as np
import weave

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from ..core import MolLEOOptimizer, MolecularPrompts
from ..oracles import TDCOracle, MolecularOracle


class MultiObjectiveOracle(MolecularOracle):
    """Oracle for multi-objective optimization"""
    
    def __init__(self, 
                 oracles: List[MolecularOracle],
                 weights: List[float] = None,
                 mode: str = "weighted_sum"):
        """
        Initialize multi-objective oracle
        
        Args:
            oracles: List of individual oracles
            weights: Weights for each oracle (for weighted sum)
            mode: Aggregation mode ('weighted_sum' or 'pareto')
        """
        super().__init__("multi_objective")
        self.oracles = oracles
        self.weights = weights or [1.0] * len(oracles)
        self.mode = mode
        
    def _evaluate_molecule_impl(self, smiles: str) -> float:
        """Evaluate molecule on all objectives"""
        scores = []
        for oracle in self.oracles:
            score = oracle.evaluate_molecule(smiles)
            scores.append(score)
            
        if self.mode == "weighted_sum":
            # Weighted sum of objectives
            return sum(s * w for s, w in zip(scores, self.weights))
        else:
            # For Pareto mode, return average (actual Pareto selection handled elsewhere)
            return np.mean(scores)
            
    def evaluate_multi(self, smiles: str) -> List[float]:
        """Get individual scores for all objectives"""
        return [oracle.evaluate_molecule(smiles) for oracle in self.oracles]


def calculate_pareto_front(population: List[Tuple[str, List[float]]]) -> List[int]:
    """
    Calculate Pareto front indices
    
    Args:
        population: List of (smiles, scores) tuples
        
    Returns:
        Indices of Pareto optimal solutions
    """
    n = len(population)
    pareto_indices = []
    
    for i in range(n):
        is_pareto = True
        scores_i = population[i][1]
        
        for j in range(n):
            if i == j:
                continue
                
            scores_j = population[j][1]
            
            # Check if j dominates i
            dominates = all(s_j >= s_i for s_j, s_i in zip(scores_j, scores_i))
            strictly_better = any(s_j > s_i for s_j, s_i in zip(scores_j, scores_i))
            
            if dominates and strictly_better:
                is_pareto = False
                break
                
        if is_pareto:
            pareto_indices.append(i)
            
    return pareto_indices


def run_multi_objective(args) -> Dict[str, Any]:
    """
    Run multi-objective optimization
    
    Args:
        args: Command line arguments with:
            - max_objectives: List of objectives to maximize
            - min_objectives: List of objectives to minimize
            - mode: 'weighted_sum' or 'pareto'
            - weights: Weights for objectives (for weighted_sum mode)
            - Other arguments same as single objective
            
    Returns:
        Optimization results
    """
    print(f"🚀 Running multi-objective optimization...")
    print(f"Maximizing: {args.max_objectives}")
    print(f"Minimizing: {args.min_objectives}")
    
    # Initialize Weave
    obj_str = "_".join(args.max_objectives + args.min_objectives)
    weave.init(f"molleo_multi_{obj_str}")
    
    # Create individual oracles
    oracles = []
    oracle_names = []
    oracle_directions = []  # 1 for maximize, -1 for minimize
    
    for obj in args.max_objectives:
        oracles.append(TDCOracle(obj))
        oracle_names.append(obj)
        oracle_directions.append(1)
        
    for obj in args.min_objectives:
        oracles.append(TDCOracle(obj))
        oracle_names.append(obj)
        oracle_directions.append(-1)
    
    # Adjust weights for min objectives
    weights = args.weights if hasattr(args, 'weights') and args.weights is not None else [1.0] * len(oracles)
    adjusted_weights = [w * d for w, d in zip(weights, oracle_directions)]
    
    # Create multi-objective oracle
    multi_oracle = MultiObjectiveOracle(
        oracles=oracles,
        weights=adjusted_weights,
        mode=args.mode if hasattr(args, 'mode') else 'weighted_sum'
    )
    
    # Get initial molecules
    print("Using default initial molecules...")
    initial_smiles = []
    
    # Use some common drug molecules as starting points
    default_molecules = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC(=O)OC1=CC=CC=C1C(=O)O",        # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # Caffeine
        "CC(C)NCC(COC1=CC=CC=C1)O",        # Propranolol
        "CN1CCC(CC1)C2=C(OC3=CC=CC=C23)C4=CC=CC=C4",  # Tamoxifen
        "CC(=O)NC1=CC=C(C=C1)O",           # Paracetamol
        "CC1=C(C=C(C=C1)C(C)C)C(C)C",      # p-Cymene
        "O=C(O)C1=CC=CC=C1O",               # Salicylic acid
        "CCC(C)C1CCC(CC1)C(C)CC",          # Menthol derivative
        "CC1=CC=C(C=C1)C(C)(C)C",          # tert-Butyltoluene
        "COC1=CC=CC=C1CCNC(C)C",           # Methoxyphenethylamine derivative
        "CC1=CC(=O)CC(C)(C)C1",            # Isophorone
        "CCOC(=O)C1=CC=CC=C1",             # Ethyl benzoate
        "CC(C)(C)C1=CC=C(O)C=C1",          # 4-tert-Butylphenol
        "COC1=CC=C(C=C1)C=O",              # p-Anisaldehyde
    ]
    
    # If TDC is available, try to get some molecules from it
    try:
        from tdc.generation import MolGen
        # Use a valid dataset like 'zinc' or 'moses'
        data = MolGen(name='zinc')
        # Get some molecules
        df = data.get_data()
        if len(df) > 0:
            # Add some molecules from TDC
            default_molecules = df.sample(n=min(10, len(df)))['smiles'].tolist()
            print(f"Successfully loaded {len(default_molecules)} molecules from TDC")
    except Exception as e:
        print(f"TDC sampling failed: {e}")
        print("Using default molecules...")
        
    initial_smiles = default_molecules[:args.initial_size]
        
    print(f"Starting with {len(initial_smiles)} initial molecules")
    
    # Create optimizer
    optimizer = MolLEOOptimizer(
        oracle=multi_oracle,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        mutation_rate=args.mutation_rate,
        model_name=args.model,
        use_llm_mutations=bool(args.model) if hasattr(args, 'model') else False
    )
    
    # Override prompt for multi-objective
    if optimizer.use_llm_mutations:
        # Custom multi-objective mutation
        original_llm_mutate = optimizer._llm_mutate
        
        def multi_obj_llm_mutate(parent_mol):
            # Get current scores
            from ..utils import mol_to_smiles
            parent_smiles = mol_to_smiles(parent_mol)
            parent_scores = multi_oracle.evaluate_multi(parent_smiles)
            
            # Create context about objectives
            obj_info = []
            for name, score, direction in zip(oracle_names, parent_scores, oracle_directions):
                dir_str = "maximize" if direction > 0 else "minimize"
                obj_info.append(f"{name} ({dir_str}): {score:.3f}")
                
            objectives_str = "\n".join(obj_info)
            
            # Create population data
            pop_data = f"Parent molecule:\nSMILES: {parent_smiles}\nObjectives:\n{objectives_str}"
            
            # Use multi-objective prompt
            prompt = MolecularPrompts.get_multi_objective_prompt(
                population_data=pop_data,
                objectives=objectives_str,
                num_molecules=5
            )
            
            # Rest of mutation logic
            try:
                response = optimizer.generator.generate(
                    prompt=prompt.build(),
                    model_name=optimizer.model_name,
                    temperature=0.8,
                    max_tokens=200
                )
                
                mutation_smiles = response['text'].strip().split('\n')
                
                for smiles in mutation_smiles:
                    smiles = smiles.strip()
                    if smiles:
                        from ..utils import smiles_to_mol
                        mol = smiles_to_mol(smiles)
                        if mol is not None:
                            return mol
                            
            except Exception as e:
                print(f"Multi-objective LLM mutation failed: {e}")
                
            return original_llm_mutate(parent_mol)
            
        optimizer._llm_mutate = multi_obj_llm_mutate
    
    # Run optimization
    results = optimizer.optimize(
        starting_smiles=initial_smiles,
        num_generations=args.generations
    )
    
    # Calculate final Pareto front if in Pareto mode
    if hasattr(args, 'mode') and args.mode == 'pareto':
        final_population_multi = []
        for smi, _ in results['final_population']:
            scores = multi_oracle.evaluate_multi(smi)
            # Adjust for minimization objectives
            adjusted_scores = [s * d for s, d in zip(scores, oracle_directions)]
            final_population_multi.append((smi, adjusted_scores))
            
        pareto_indices = calculate_pareto_front(final_population_multi)
        pareto_solutions = [final_population_multi[i] for i in pareto_indices]
        
        results['pareto_front'] = pareto_solutions
        print(f"\nFound {len(pareto_solutions)} Pareto optimal solutions")
    
    # Print summary
    print("\n📊 Optimization Results:")
    print(f"Best weighted score: {results['best_score']:.4f}")
    print(f"Total oracle calls: {results['oracle_calls']}")
    
    # Evaluate best molecule on all objectives
    best_smiles = results['best_molecule']
    best_scores = multi_oracle.evaluate_multi(best_smiles)
    print(f"\nBest molecule: {best_smiles}")
    for name, score, direction in zip(oracle_names, best_scores, oracle_directions):
        dir_str = "↑" if direction > 0 else "↓"
        print(f"  {name} {dir_str}: {score:.4f}")
    
    return results