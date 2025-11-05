"""Analysis mode for evaluating generated structures and computing comprehensive metrics"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from pymatgen.core.structure import Structure

from ..utils.evaluate_structures import StructureEvaluator
from ..utils.data_loader import load_training_structures_from_cif_csv, load_seed_structures
from .csg import MatLLMSearchCSG


def run_analyze(args) -> Dict[str, Any]:
    """
    Analysis mode: Evaluate generated structures and compute all metrics.
    
    Can either:
    1. Read existing results from CSV file
    2. Run new evaluation (if input structures provided)
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with evaluation results
    """
    print("="*80)
    print("ANALYSIS MODE: Structure Evaluation")
    print("="*80)
    
    # Check if we should generate structures via API using CSG workflow
    generate_via_api = getattr(args, 'generate', False)
    
    if generate_via_api:
        # Generate structures using CSG workflow
        print("Generating structures using CSG workflow...")
        structures = _generate_structures_via_csg(args)
        
        if not structures:
            print("Error: No structures were generated via CSG workflow")
            sys.exit(1)
        
        print(f"Generated and deduplicated {len(structures)} unique structures via CSG workflow")
    else:
        # Read structures from file
        # Determine input file - support both old format (results-path) and new format (input)
        if hasattr(args, 'input') and args.input:
            input_file = Path(args.input)
        elif hasattr(args, 'results_path') and args.results_path:
            # Old format: look for generations.csv in results_path
            results_path = Path(args.results_path)
            input_file = results_path / "generations.csv"
            if not input_file.exists():
                print(f"Error: generations.csv not found in {results_path}")
                sys.exit(1)
        else:
            # Default: data/llama_test.csv
            project_root = Path(__file__).parent.parent.parent
            input_file = project_root / "data" / "llama_test.csv"
            if not input_file.exists():
                print(f"Error: Default input file not found: {input_file}")
                print("Please specify --input, --results-path, or --generate to use API")
                sys.exit(1)
        
        print(f"Input file: {input_file}")
        
        # Load structures from input file
        print(f"\nLoading structures from {input_file}...")
        evaluator_temp = StructureEvaluator()
        structures = evaluator_temp.load_structures_from_file(str(input_file), fmt='auto')
        
        if not structures:
            print(f"Error: No valid structures found in {input_file}")
            sys.exit(1)
        
        print(f"Loaded {len(structures)} structures")
    
    # Determine novelty reference set (SUN score reference)
    # Priority:
    # 1) If --training-data provided, use it
    # 2) Otherwise, use the reference pool (seed structures) from --data-path
    training_structures = []
    if hasattr(args, 'training_data') and args.training_data:
        training_file = Path(args.training_data)
    else:
        training_file = None
    
    # Determine output file
    if hasattr(args, 'output') and args.output:
        output_file = Path(args.output)
    elif hasattr(args, 'results_path') and args.results_path:
        # save to results_path
        results_path = Path(args.results_path)
        output_file = results_path / f"{getattr(args, 'experiment_name', 'experiment')}_evaluation.json"
    else:
        project_root = Path(__file__).parent.parent.parent
        output_file = project_root / "evaluation_results.json"
    
    print(f"Output file: {output_file}")
    print()
    
    # Initialize evaluator
    evaluator = StructureEvaluator(
        mlip=getattr(args, 'mlip', 'chgnet'),
        ppd_path=getattr(args, 'ppd_path', 'data/2023-02-07-ppd-mp.pkl.gz'),
        device=getattr(args, 'device', 'cuda')
    )
    
    # Load novelty reference structures
    if training_file and training_file.exists():
        print(f"Loading novelty reference structures from {training_file}...")
        if training_file.suffix == '.csv':
            training_structures = load_training_structures_from_cif_csv(str(training_file))
        else:
            training_structures = evaluator.load_structures_from_file(str(training_file), fmt='json')
    else:
        # Fallback to reference pool (seed structures) from data_path
        data_path = getattr(args, 'data_path', 'data/band_gap_processed_5000.csv')
        print(f"Loading novelty reference structures from reference pool: {data_path}...")
        try:
            training_structures = load_seed_structures(
                data_path=data_path,
                task="csg",
                random_seed=getattr(args, 'seed', 42)
            )
        except Exception as e:
            print(f"Warning: Failed to load reference pool from {data_path}: {e}")
            training_structures = []

    if training_structures:
        evaluator.training_structures = training_structures
        evaluator.training_formulas = {}
        for struct in training_structures:
            try:
                formula = struct.composition.reduced_formula
                if formula not in evaluator.training_formulas:
                    evaluator.training_formulas[formula] = []
                evaluator.training_formulas[formula].append(struct)
            except Exception as e:
                print(f"Error loading training structures: {e}")
                continue
        print(f"Loaded {len(training_structures)} novelty reference structures")
    else:
        print("No novelty reference structures available - SUN score calculation will be limited")
    
    # Limit structures if requested (for faster testing)
    if hasattr(args, 'limit') and args.limit and args.limit > 0:
        if len(structures) > args.limit:
            print(f"Limiting to first {args.limit} structures for faster evaluation")
            structures = structures[:args.limit]
    
    # Evaluate structures
    print("\n" + "="*80)
    print("Starting evaluation...")
    print("="*80)
    
    # Always calculate stability
    results = evaluator.evaluate(structures, calculate_stability=True)
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        def convert_to_serializable(obj):
            """Convert numpy types and other non-serializable objects to native Python types"""
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, '__float__'):
                return float(obj)
            elif hasattr(obj, '__int__'):
                return int(obj)
            else:
                return str(obj)
        
        json.dump(convert_to_serializable(results), f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    print_summary(results)
    print("\n" + "="*80)
    print(f"Full results saved to: {output_file}")
    print("="*80)
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print summary of evaluation results"""
    print(f"\nTotal structures: {results.get('total_structures', 0)}")
    
    # Validity metrics
    if 'structural_validity' in results:
        print(f"\nStructural Validity: {results['structural_validity']:.4f}")
    if 'composition_validity' in results:
        print(f"Composition Validity: {results['composition_validity']:.4f}")
    
    # Diversity metrics
    if 'composition_diversity' in results:
        print(f"\nComposition Diversity: {results['composition_diversity']:.4f}")
        if 'composition_diversity_details' in results:
            cd = results['composition_diversity_details']
            print(f"  Unique compositions: {cd.get('unique_compositions', 0)}")
    
    if 'structural_diversity' in results:
        print(f"Structural Diversity: {results['structural_diversity']:.4f}")
        if 'structural_diversity_details' in results:
            sd = results['structural_diversity_details']
            print(f"  Unique structures: {sd.get('unique_structures', 0)}")
    
    # Novelty metrics
    if 'overall_novelty' in results:
        print(f"\nOverall Novelty (SUN score): {results['overall_novelty']:.4f}")
    
    if 'composition_novelty' in results:
        print(f"Composition Novelty: {results['composition_novelty']:.4f}")
        if 'novelty' in results and 'composition_novelty' in results['novelty']:
            cn = results['novelty']['composition_novelty']
            print(f"  Novel compositions: {cn.get('novel_compositions', 0)}")
    
    if 'structural_novelty' in results:
        print(f"Structural Novelty: {results['structural_novelty']:.4f}")
        if 'novelty' in results and 'structural_novelty' in results['novelty']:
            sn = results['novelty']['structural_novelty']
            print(f"  Novel structures: {sn.get('novel_structures', 0)}")
    
    # Success rate and stability metrics
    if 'success_rate' in results:
        sr = results['success_rate']
        if 'validity_rate' in sr:
            print(f"\nValidity Rate: {sr.get('validity_rate', 0):.4f}")
            print(f"Success Rate (<0.1 eV): {sr.get('success_rate', 0):.4f}")
            print(f"Stability Rate (<0.03 eV): {sr.get('stability_rate_0.03', 0):.4f}")
            print(f"Stability Rate (<0.10 eV): {sr.get('stability_rate_0.10', 0):.4f}")
    
    if 'm3gnet_metastability' in results:
        print(f"M3GNet Metastability: {results['m3gnet_metastability']:.4f}")
            
            # Stability statistics
    if 'stability_stats' in results:
        stats = results['stability_stats']
        print(f"\nStability Statistics:")
        print(f"  Min E-hull: {stats.get('min_e_hull', 0):.6f} eV/atom")
        print(f"  Mean E-hull: {stats.get('mean_e_hull', 0):.6f} eV/atom")
        print(f"  Median E-hull: {stats.get('median_e_hull', 0):.6f} eV/atom")


def _generate_structures_via_csg(args) -> list:
    """
    Generate structures using the CSG workflow (MatLLMSearchCSG).
    
    This uses the existing CSG implementation which properly uses StructureGenerator
    and the evolutionary workflow.
    
    Args:
        args: Command line arguments with generation parameters
        
    Returns:
        List of generated structures
    """
    # Set up CSG args - ensure required attributes are set
    # Set defaults for CSG-specific args if not provided
    # Note: population_size controls per-iteration generation, not total limit
    if not hasattr(args, 'population_size') or args.population_size is None:
        # Use a reasonable default for population size (will generate more structures across iterations)
        args.population_size = 10
    
    if not hasattr(args, 'reproduction_size'):
        args.reproduction_size = 5
    
    if not hasattr(args, 'parent_size'):
        args.parent_size = 2
    
    if not hasattr(args, 'max_iter'):
        args.max_iter = 1
    
    if not hasattr(args, 'opt_goal'):
        args.opt_goal = 'e_hull_distance'
    
    if not hasattr(args, 'log_dir'):
        args.log_dir = 'logs'
    
    if not hasattr(args, 'save_label'):
        args.save_label = 'analyze_generation'
    
    # Initialize and run CSG workflow
    csg_project = MatLLMSearchCSG(args=args)
    csg_results = csg_project.run()
    
    # Extract structures from CSG results
    # CSG saves results to generations.csv in the output path
    output_path = Path(csg_results.get('output_path', 'logs/analyze_generation'))
    generations_file = output_path / "generations.csv"
    
    structures = []
    if generations_file.exists():
        # Load structures from the generations.csv file
        df = pd.read_csv(generations_file)
        
        for _, row in df.iterrows():
            try:
                # Parse structure from JSON
                struct_dict = json.loads(row['Structure'])
                structure = Structure.from_dict(struct_dict)
                structures.append(structure)
            except Exception as e:
                print(f"Warning: Could not parse structure from row {_}: {e}")
                continue
        
        print(f"Loaded {len(structures)} structures from CSG results")
    else:
        print(f"Warning: Generations file not found at {generations_file}")
        return []
    
    # Deduplicate structures after all iterations
    # Note: CSG workflow saves all structures from all iterations without deduplication,
    # so we need to deduplicate here before evaluation
    print(f"\nDeduplicating {len(structures)} structures...")
    structures = _deduplicate_structures(structures)
    print(f"After deduplication: {len(structures)} unique structures")
    
    return structures


def _deduplicate_structures(structures: list) -> list:
    """
    Deduplicate structures using StructureMatcher.
    
    Args:
        structures: List of structures to deduplicate
        
    Returns:
        List of unique structures
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    
    if not structures:
        return []
    
    matcher = StructureMatcher()
    unique_structures = []
    
    for struct in structures:
        if struct is None:
            continue
        
        is_unique = True
        for unique_struct in unique_structures:
            try:
                if matcher.fit(struct, unique_struct):
                    is_unique = False
                    break
            except Exception:
                # If matching fails, consider it unique
                continue
        
        if is_unique:
            unique_structures.append(struct)
    
    return unique_structures

