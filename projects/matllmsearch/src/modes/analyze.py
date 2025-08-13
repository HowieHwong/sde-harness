"""Analysis mode for MatLLMSearch results"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json
import matplotlib.pyplot as plt
import seaborn as sns

from pymatgen.core.structure import Structure


def run_analyze(args) -> Dict[str, Any]:
    """Analyze MatLLMSearch experimental results"""
    
    results_path = Path(args.results_path)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results path {results_path} not found")
    
    print(f"Analyzing results in {results_path}")
    
    # Load generation data
    generations_file = results_path / "generations.csv"
    metrics_file = results_path / "metrics.csv"
    
    analysis_results = {}
    
    if generations_file.exists():
        print("Analyzing generation data...")
        generation_analysis = analyze_generations(generations_file)
        analysis_results.update(generation_analysis)
    
    if metrics_file.exists():
        print("Analyzing metrics data...")
        metrics_analysis = analyze_metrics(metrics_file)
        analysis_results.update(metrics_analysis)
    
    # Generate summary report
    generate_analysis_report(analysis_results, results_path, args.experiment_name)
    
    print("Analysis completed!")
    return analysis_results


def analyze_generations(generations_file: Path) -> Dict[str, Any]:
    """Analyze generation data"""
    
    df = pd.read_csv(generations_file)
    
    # Basic statistics
    total_structures = len(df)
    unique_iterations = df['Iteration'].nunique()
    
    # Parse compositions
    compositions = []
    for _, row in df.iterrows():
        try:
            if pd.notna(row['Composition']):
                comp_str = str(row['Composition'])
                compositions.append(comp_str)
        except:
            continue
    
    unique_compositions = len(set(compositions))
    
    # Stability analysis
    stability_stats = {}
    if 'EHullDistance' in df.columns:
        ehull_values = df['EHullDistance'].dropna()
        if len(ehull_values) > 0:
            stability_stats = {
                'min_ehull': float(ehull_values.min()),
                'mean_ehull': float(ehull_values.mean()),
                'median_ehull': float(ehull_values.median()),
                'stable_count_003': int(sum(ehull_values <= 0.03)),
                'stable_count_01': int(sum(ehull_values <= 0.1)),
                'stable_rate_003': float(sum(ehull_values <= 0.03) / len(ehull_values)),
                'stable_rate_01': float(sum(ehull_values <= 0.1) / len(ehull_values))
            }
    
    # Energy analysis
    energy_stats = {}
    if 'EnergyRelaxed' in df.columns:
        energy_values = df['EnergyRelaxed'].dropna()
        if len(energy_values) > 0:
            energy_stats = {
                'min_energy': float(energy_values.min()),
                'mean_energy': float(energy_values.mean()),
                'median_energy': float(energy_values.median())
            }
    
    return {
        'generation_analysis': {
            'total_structures': total_structures,
            'unique_iterations': unique_iterations,
            'unique_compositions': unique_compositions,
            'diversity_rate': unique_compositions / total_structures if total_structures > 0 else 0,
            'stability_stats': stability_stats,
            'energy_stats': energy_stats
        }
    }


def analyze_metrics(metrics_file: Path) -> Dict[str, Any]:
    """Analyze metrics data"""
    
    df = pd.read_csv(metrics_file)
    
    # Progress over iterations
    if 'iteration' in df.columns:
        iterations = df['iteration'].values
        
        # Stability metrics over time
        stability_progress = {}
        for threshold in ['0.00', '0.03', '0.10']:
            stable_rate_col = f'stable_rate_{threshold}'
            stable_num_col = f'stable_num_{threshold}'
            
            if stable_rate_col in df.columns:
                stability_progress[f'stable_rate_{threshold}'] = df[stable_rate_col].tolist()
            if stable_num_col in df.columns:
                stability_progress[f'stable_num_{threshold}'] = df[stable_num_col].tolist()
        
        # Validity metrics over time
        validity_progress = {}
        for col in ['valid', 'comp_valid', 'struct_valid']:
            if col in df.columns:
                validity_progress[col] = df[col].tolist()
        
        # Diversity metrics over time
        diversity_progress = {}
        if 'comp_div' in df.columns:
            diversity_progress['composition_diversity'] = df['comp_div'].tolist()
        
        return {
            'metrics_analysis': {
                'iterations': iterations.tolist(),
                'stability_progress': stability_progress,
                'validity_progress': validity_progress,
                'diversity_progress': diversity_progress
            }
        }
    
    return {'metrics_analysis': {}}


def generate_analysis_report(analysis_results: Dict[str, Any], 
                           output_path: Path, experiment_name: str):
    """Generate comprehensive analysis report"""
    
    report_path = output_path / f"{experiment_name}_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"MatLLMSearch Analysis Report: {experiment_name}\\n")
        f.write("=" * 60 + "\\n\\n")
        
        # Generation analysis
        if 'generation_analysis' in analysis_results:
            gen_analysis = analysis_results['generation_analysis']
            f.write("GENERATION ANALYSIS\\n")
            f.write("-" * 30 + "\\n")
            f.write(f"Total structures generated: {gen_analysis['total_structures']}\\n")
            f.write(f"Unique iterations: {gen_analysis['unique_iterations']}\\n")
            f.write(f"Unique compositions: {gen_analysis['unique_compositions']}\\n")
            f.write(f"Composition diversity rate: {gen_analysis['diversity_rate']:.3f}\\n\\n")
            
            # Stability statistics
            if gen_analysis['stability_stats']:
                stats = gen_analysis['stability_stats']
                f.write("STABILITY STATISTICS\\n")
                f.write("-" * 30 + "\\n")
                f.write(f"Minimum E_hull distance: {stats['min_ehull']:.6f} eV/atom\\n")
                f.write(f"Mean E_hull distance: {stats['mean_ehull']:.6f} eV/atom\\n")
                f.write(f"Stable structures (≤0.03 eV/atom): {stats['stable_count_003']} ({stats['stable_rate_003']:.1%})\\n")
                f.write(f"Metastable structures (≤0.1 eV/atom): {stats['stable_count_01']} ({stats['stable_rate_01']:.1%})\\n\\n")
            
            # Energy statistics
            if gen_analysis['energy_stats']:
                stats = gen_analysis['energy_stats']
                f.write("ENERGY STATISTICS\\n")
                f.write("-" * 30 + "\\n")
                f.write(f"Minimum energy: {stats['min_energy']:.6f} eV/atom\\n")
                f.write(f"Mean energy: {stats['mean_energy']:.6f} eV/atom\\n\\n")
        
        # Metrics analysis
        if 'metrics_analysis' in analysis_results:
            metrics_analysis = analysis_results['metrics_analysis']
            f.write("OPTIMIZATION PROGRESS\\n")
            f.write("-" * 30 + "\\n")
            
            if 'stability_progress' in metrics_analysis:
                stability_prog = metrics_analysis['stability_progress']
                if 'stable_rate_0.03' in stability_prog:
                    final_rate = stability_prog['stable_rate_0.03'][-1] if stability_prog['stable_rate_0.03'] else 0
                    f.write(f"Final stability rate (≤0.03 eV/atom): {final_rate:.1%}\\n")
                
                if 'stable_num_0.03' in stability_prog:
                    final_count = stability_prog['stable_num_0.03'][-1] if stability_prog['stable_num_0.03'] else 0
                    f.write(f"Final stable structure count: {final_count}\\n")
            
            if 'validity_progress' in metrics_analysis:
                validity_prog = metrics_analysis['validity_progress']
                if 'valid' in validity_prog:
                    final_validity = validity_prog['valid'][-1] if validity_prog['valid'] else 0
                    f.write(f"Final structure validity rate: {final_validity:.1%}\\n")
    
    print(f"Analysis report saved to {report_path}")
