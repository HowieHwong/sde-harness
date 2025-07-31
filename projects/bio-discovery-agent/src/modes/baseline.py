"""Baseline sampling mode."""
import os
import numpy as np
import random
from typing import Dict, Any, List
from ..utils.gene_utils import *
from ..evaluators.bio_metrics import BioEvaluator


def baseline_sample(dataset: str, sample_size: int, pathways: List[str]) -> List[str]:
    """
    Baseline sampling of genes.
    
    Args:
        dataset: Dataset name
        sample_size: Number of genes to sample
        pathways: List of pathways to sample from
        
    Returns:
        List of sampled gene names
    """
    # Get all genes in pathways
    pathway_genes = []
    for pathway in pathways:
        pathway_genes.extend(get_genes_in_pathway(pathway))
    
    # Remove duplicates
    pathway_genes = list(set(pathway_genes))
    
    # Sample genes
    if len(pathway_genes) >= sample_size:
        sampled = random.sample(pathway_genes, sample_size)
    else:
        # If not enough genes, sample with replacement
        sampled = random.choices(pathway_genes, k=sample_size)
    
    return sampled


def run_baseline(args) -> Dict[str, Any]:
    """Run baseline sampling."""
    print(f"Running baseline sampling for {args.data_name}")
    print(f"Sample size: {args.sample_size}")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize evaluator
    evaluator = BioEvaluator(args.data_name)
    
    # Sample genes
    if args.pathways:
        print(f"Sampling from pathways: {args.pathways}")
        sampled_genes = baseline_sample(args.data_name, args.sample_size, args.pathways)
    else:
        # Random sampling from all genes
        all_genes = get_all_genes()  # This function should be implemented in gene_utils
        sampled_genes = random.sample(all_genes, min(args.sample_size, len(all_genes)))
    
    # Evaluate
    results = evaluator.evaluate(sampled_genes)
    
    # Save results
    log_dir = os.path.join(args.log_dir, f"baseline_{args.data_name}")
    os.makedirs(log_dir, exist_ok=True)
    
    np.save(
        os.path.join(log_dir, "sampled_genes.npy"),
        np.array(sampled_genes)
    )
    
    print(f"\\nBaseline Results:")
    print(f"Hit rate: {results['hit_rate']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1 Score: {results['f1_score']:.3f}")
    
    return results