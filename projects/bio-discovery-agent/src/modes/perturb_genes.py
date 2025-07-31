"""Gene perturbation discovery mode with fixes including gene scores."""
import os
import sys
import json
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Set
import weave

# Add sde-harness to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from sde_harness.core import Generation, Oracle, Workflow, Prompt
from ..utils.data_loader import load_dataset, format_gene_list
from ..utils.prompts import get_prompt_template
from ..utils.tools import BioDiscoveryTools
from ..utils.llm_interface import BioLLMInterface
from ..evaluators.bio_metrics import BioEvaluator


def parse_gene_solution(response: str, is_pairs: bool = False) -> List[str]:
    """Parse gene names from LLM response."""
    # Look for "Solution:" section
    solution_match = re.search(r'Solution:(.*?)(?:\n\n|$)', response, re.DOTALL)
    if not solution_match:
        solution_match = re.search(r'[234]\. Solution:(.*?)(?:\n\n|$)', response, re.DOTALL)
    
    if solution_match:
        solution_text = solution_match.group(1).strip()
        
        if is_pairs:
            # Extract gene pairs
            pair_pattern = r'\d+\.\s*([A-Z][A-Z0-9\-_]+)\s*\+\s*([A-Z][A-Z0-9\-_]+)'
            pairs = re.findall(pair_pattern, solution_text)
            # Return pairs as tuples sorted alphabetically
            return [tuple(sorted([gene1, gene2])) for gene1, gene2 in pairs]
        else:
            # Extract single gene names
            gene_pattern = r'\d+\.\s*([A-Z][A-Z0-9\-_]+)'
            genes = re.findall(gene_pattern, solution_text)
            return genes
    
    return []


def parse_tool_request(response: str, tool_name: str) -> str:
    """Parse tool request from LLM response."""
    # Look for tool section
    patterns = [
        rf'{tool_name}:\s*([A-Z][A-Z0-9\-_]+)',
        rf'3\.\s*{tool_name}:\s*([A-Z][A-Z0-9\-_]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ""


def load_ground_truth_with_scores(data_name: str, is_pairs: bool = False) -> pd.DataFrame:
    """Load ground truth data with scores."""
    try:
        if not is_pairs:
            ground_truth = pd.read_csv(f'./datasets/ground_truth_{data_name}.csv', index_col=0)
        else:
            import ast
            ground_truth = pd.read_csv(f'./datasets/ground_truth_{data_name}.csv')
            ground_truth['Gene_pairs'] = ground_truth['Gene_pairs'].apply(ast.literal_eval)
            ground_truth.set_index('Gene_pairs', inplace=True)
        return ground_truth
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return pd.DataFrame()


def format_gene_scores(genes: List[str], ground_truth: pd.DataFrame, 
                      max_display: int = 500) -> str:
    """Format genes with their scores for display."""
    if len(genes) == 0:
        return "No genes tested yet."
    
    # Get scores for all genes (they should all be valid now)
    gene_scores = []
    for gene in genes[:max_display]:
        if gene in ground_truth.index:
            score = ground_truth.loc[gene].values[0] if hasattr(ground_truth.loc[gene], 'values') else ground_truth.loc[gene]
            gene_scores.append((gene, score))
    
    # Format as simple line-by-line list
    result = ""
    for gene, score in gene_scores:
        if isinstance(score, float):
            result += f"{gene}: {score:.4f}\n"
        else:
            result += f"{gene}: {score}\n"
    
    if len(genes) > max_display:
        result += f"... and {len(genes) - max_display} more genes"
    
    return result.strip()


def run_discovery_step(llm_interface: BioLLMInterface,
                      prompt: str,
                      args: Any,
                      all_tested_genes: Set[str],
                      ground_truth: pd.DataFrame,
                      enable_tools: bool = False,
                      is_pairs: bool = False) -> Tuple[str, List[str]]:
    """Run a single discovery step - generate genes and keep only valid ones."""
    measured_genes = set(ground_truth.index.values)
    
    # Generate response
    response = llm_interface.complete_text(
        prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Handle tool usage if enabled
    if enable_tools:
        # Check for gene search request
        if args.gene_search:
            gene_to_search = parse_tool_request(response, "Gene Search")
            if gene_to_search:
                similar_genes = BioDiscoveryTools.gene_search(
                    gene_to_search, args.csv_path, k=10
                )
                if similar_genes:
                    tool_info = f"\nSimilar genes to {gene_to_search}: {', '.join(similar_genes)}"
                    response += tool_info
        
        # Check for pathway search
        if args.reactome:
            gene_for_pathways = parse_tool_request(response, "Reactome Pathways")
            if gene_for_pathways:
                pathways = BioDiscoveryTools.get_reactome_pathways(gene_for_pathways)
                if pathways:
                    tool_info = f"\nPathways for {gene_for_pathways}: {', '.join(pathways)}"
                    response += tool_info
    
    # Parse genes from response
    try:
        predicted_genes = parse_gene_solution(response, is_pairs=is_pairs)
        original_count = len(predicted_genes)
    except:
        print('Failed to parse genes from response')
        predicted_genes = []
        original_count = 0
    
    # Remove duplicates within this prediction
    if is_pairs:
        # For pairs, use set of tuples to remove duplicates
        unique_predicted = []
        seen_pairs = set()
        for pair in predicted_genes:
            if isinstance(pair, tuple) and len(pair) == 2:
                # Sort to ensure (A,B) and (B,A) are treated as same
                sorted_pair = tuple(sorted(pair))
                if sorted_pair not in seen_pairs:
                    seen_pairs.add(sorted_pair)
                    unique_predicted.append(pair)
        predicted_genes = unique_predicted
    else:
        # For single genes, convert to list of unique genes preserving order
        seen = set()
        unique_predicted = []
        for gene in predicted_genes:
            if gene not in seen:
                seen.add(gene)
                unique_predicted.append(gene)
        predicted_genes = unique_predicted
    
    if original_count > len(predicted_genes):
        print(f"LLM generated {original_count} genes total, {len(predicted_genes)} unique (removed {original_count - len(predicted_genes)} duplicates)")
    else:
        print(f"LLM generated {len(predicted_genes)} unique genes (no duplicates)")
    
    # Keep only valid genes that exist in ground truth and haven't been tested
    valid_genes = []
    invalid_count = 0
    already_tested_count = 0
    
    if is_pairs:
        # For pairs, validate both genes exist
        for pair in predicted_genes:
            if isinstance(pair, tuple) and len(pair) == 2:
                if pair[0] in measured_genes and pair[1] in measured_genes:
                    pair_str = f"{pair[0]}_{pair[1]}"
                    if pair_str not in all_tested_genes:
                        valid_genes.append(pair_str)
                    else:
                        already_tested_count += 1
                else:
                    invalid_count += 1
    else:
        # For single genes, keep those in ground truth and not tested
        for gene in predicted_genes:
            if gene in measured_genes:
                if gene not in all_tested_genes:
                    valid_genes.append(gene)
                else:
                    already_tested_count += 1
            else:
                invalid_count += 1
    
    print(f"Dropped {invalid_count} invalid genes and {already_tested_count} already tested genes")
    print(f"Final valid genes for this step: {len(valid_genes)}")
    
    return response, valid_genes


def get_task_config(task_variant: str, data_name: str, task_description: str, 
                   measurement: str, num_genes: int) -> Dict[str, str]:
    """Get the research problem and instructions based on task variant."""
    
    if task_variant == "brief":
        research_problem = (f"I'm planning to run a genome-wide CRISPR screen "
                          f"to {task_description}. There are 18,939 possible genes to perturb and I can only "
                          f"perturb {num_genes} genes at a time. For each "
                          f"perturbation, I'm able to measure out {measurement} which "
                          f"will be referred to as the score. I can "
                          f"only do a few rounds of experimentation.")
        
        instructions = (f"\n Based on these results and "
                       f"prior knowledge of biology, make the best "
                       f"possible prediction of the "
                       f"first {num_genes} genes that I should test to maximize "
                       f"the score. Use HGNC gene naming convention."
                       f"DO NOT PREDICT GENES THAT HAVE ALREADY BEEN TESTED")
    
    elif task_variant == "brief-NormanGI":
        research_problem = (f"I'm planning to run a genome-wide CRISPR screen "
                          f"to {task_description}. There are 92 possible genes to perturb and I can only "
                          f"perturb {num_genes} gene pairs at a time. For each "
                          f"perturbation, I'm able to measure out {measurement} which "
                          f"will be referred to as the score. I can "
                          f"only do a few rounds of experimentation.")
        
        instructions = (f"\n Based on these results and "
                       f"prior knowledge of biology, make the best "
                       f"possible prediction of the "
                       f"first {num_genes} genes that I should test to maximize "
                       f"the score. Use HGNC gene naming convention."
                       f"DO NOT PREDICT GENES THAT HAVE ALREADY BEEN TESTED")
    
    elif task_variant == "brief-Horlbeck":
        research_problem = (f"I am interested in {task_description}. There are 450 genes from which pairs of "
                          f"genes must be chosen. I can only perturb {num_genes} gene pairs at a "
                          f"time. For each perturbation, I'm able to measure out {measurement} which will "
                          f"be referred to as the score.")
        
        instructions = (f"\n Based on these results and using your prior knowledge of biology,"
                       f"can you suggest {num_genes} other combinations that may also show a synergistic"
                       f"effect upon perturbation. DO NOT PREDICT GENE PAIRS THAT HAVE ALREADY"
                       f"BEEN TESTED. Hint: genetic interactions are often found between"
                       f"functionally related genes")
    
    else:  # "full" variant
        research_problem = (f"You are running a series of experiments to "
                          f"identify genes whose perturbation would "
                          f"most impact Interferon-gamma production. Given a "
                          f"list of experimental outcomes following the perturbation of some set of genes, "
                          f"the goal is to predict a set of new genes "
                          f"to perturb that will have a significant impact on the biological process. "
                          f"For each gene perturbation, you are able to measure out "
                          f"{measurement} which will be referred to as the score. "
                          f"The goal is to identify the set of {num_genes} functionally related genes.")
        
        instructions = (f"\n Based on these results and your knowledge of biology, "
                       f"predict the next {num_genes} genes I should experimentally "
                       f"test, i.e. genes that show a strong log "
                       f"fold change in INF-gamma (whether strongly positive or strongly negative) "
                       f"upon being knocked out. IFN-gamma is a cytokine produced "
                       f"by CD4+ and CD8+ T cells that induces additional T "
                       f"cells. It might be worth exploring co-essential "
                       f"genes. Use HGNC gene naming convention.  "
                       f"DO NOT PREDICT GENES THAT HAVE ALREADY BEEN TESTED ")
    
    return {
        "research_problem": research_problem,
        "instructions": instructions
    }


def run_perturb_genes(args) -> Dict[str, Any]:
    """Main gene perturbation discovery workflow with fixes including scores."""
    # Initialize Weave for experiment tracking
    weave.init(f"bio-discovery-{args.data_name}")
    
    # Load dataset
    dataset = load_dataset(args.data_name)
    task_description = dataset.get("task_description", "")
    measurement = dataset.get("measurement", "")
    
    # Get task configuration based on variant
    task_variant = getattr(args, 'task_variant', 'brief')
    task_config = get_task_config(
        task_variant, args.data_name, task_description, measurement, args.num_genes
    )
    research_problem = task_config["research_problem"]
    instructions = task_config["instructions"]
    
    # Initialize components
    llm_interface = BioLLMInterface(model=args.model)
    evaluator = BioEvaluator(args.data_name)
    
    # Determine if using pairs
    is_pairs = task_variant in ["brief-NormanGI", "brief-Horlbeck"]
    
    # Load ground truth with scores
    ground_truth = load_ground_truth_with_scores(args.data_name, is_pairs)
    
    # Setup logging
    log_dir = os.path.join(args.log_dir, f"{args.model}_{args.data_name}", args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Track all predictions and hits
    all_tested_genes = []  # List to maintain order and scores
    all_tested_genes_set = set()  # Set for fast lookup
    all_hits = []  # Track hit genes with scores
    hits_history = []  # Track hits progression
    round_results = []
    
    # Determine which prompt template to use
    if task_variant == "brief-NormanGI":
        initial_prompt_type = "perturb_genes_pairs_norman"
        follow_up_prompt_type = "follow_up"
        tool_section = None
        tool_description = None
    elif task_variant == "brief-Horlbeck":
        initial_prompt_type = "perturb_genes_pairs_horlbeck"
        follow_up_prompt_type = "follow_up"
        tool_section = None
        tool_description = None
    elif args.gene_search:
        initial_prompt_type = "perturb_genes_gene_search"
        follow_up_prompt_type = "follow_up_with_tool"
        tool_section = "Gene Search"
        tool_description = "10 most similar genes based on features"
    elif args.reactome:
        initial_prompt_type = "perturb_genes_pathways"
        follow_up_prompt_type = "follow_up_with_tool"
        tool_section = "Reactome Pathways"
        tool_description = "the associated biological pathways"
    else:
        initial_prompt_type = "perturb_genes"
        follow_up_prompt_type = "follow_up"
        tool_section = None
        tool_description = None
    
    # Run discovery steps
    for step in range(args.steps):
        print(f"\n=== Step {step + 1}/{args.steps} ===")
        
        if step == 0:
            # Initial prompt
            prompt_template = get_prompt_template(
                initial_prompt_type,
                research_problem=research_problem
            )
            prompt = prompt_template.build()
        else:
            # Follow-up prompt with previous results INCLUDING SCORES
            
            # Format tested genes with their scores
            non_hit_genes = [g for g in all_tested_genes if g not in [h[0] for h in all_hits]]
            hit_genes = [h[0] for h in all_hits]
            
            # Create detailed results section
            result = f"This is not your first round. All tested genes and their measured {measurement} are:\n\n"
            
            # Show non-hit genes with scores (only those with valid scores)
            if non_hit_genes:
                formatted_non_hits = format_gene_scores(non_hit_genes, ground_truth)
                if formatted_non_hits:
                    result += "NON-HIT GENES (lower impact):\n"
                    result += formatted_non_hits
                    result += "\n\n"
            
            # Show hit genes with scores (emphasized)
            if all_hits:
                result += f"HIT GENES (high impact) - You have successfully identified {len(all_hits)} hits so far:\n"
                for gene, score in all_hits:
                    result += f"{gene}: {score:.4f}\n"
                result += "\n"
                result += "These hit genes are particularly important as they show strong effects.\n"
            
            # Add progression information
            result += f"\nProgression of cumulative hits across rounds: {', '.join(map(str, hits_history))}"
            if len(hits_history) > 2 and hits_history[-1] - hits_history[-3] < 5:
                result += "\nNote: The number of hits has not increased much recently. Consider exploring different types of genes or pathways."
            
            # Create observed section (just list of all tested genes for reference)
            if is_pairs:
                tested_display = [g.replace('_', ' + ') for g in all_tested_genes]
            else:
                tested_display = all_tested_genes
            observed = f"For reference, ALL genes tested so far across all rounds: {format_gene_list(tested_display)}"
            
            if tool_section:
                prompt_template = get_prompt_template(
                    follow_up_prompt_type,
                    research_problem=research_problem,
                    observed=observed,
                    measurement=measurement,
                    result=result,
                    instructions=instructions,
                    tool_section=tool_section,
                    tool_description=tool_description
                )
            else:
                prompt_template = get_prompt_template(
                    follow_up_prompt_type,
                    research_problem=research_problem,
                    observed=observed,
                    measurement=measurement,
                    result=result,
                    instructions=instructions
                )
            prompt = prompt_template.build()
        
        # Run discovery step
        response, predicted_genes = run_discovery_step(
            llm_interface,
            prompt,
            args,
            all_tested_genes_set,
            ground_truth,
            enable_tools=(args.gene_search or args.reactome),
            is_pairs=is_pairs
        )
        
        # Add critique if enabled
        if args.critique and predicted_genes:
            # Include scores in critique
            critique_context = format_gene_list(predicted_genes)
            if all_hits:
                critique_context += f"\n\nFor context, successful hits so far: {[h[0] for h in all_hits]}"
            
            critique = BioDiscoveryTools.critique_solution(
                critique_context,
                research_problem,
                args.model
            )
            response += f"\n\nCritique:\n{critique}"
        
        # Add literature review if enabled
        if args.lit_review and predicted_genes and len(predicted_genes) > 0:
            lit_review = BioDiscoveryTools.literature_search(
                predicted_genes[0],  # Review first gene as example
                research_problem,
                args.model
            )
            response += f"\n\nLiterature Review for {predicted_genes[0]}:\n{lit_review}"
        
        # Save response
        with open(os.path.join(log_dir, f"step_{step + 1}_response.txt"), "w") as f:
            f.write(response)
        
        # Update tracking
        all_tested_genes.extend(predicted_genes)
        all_tested_genes_set.update(predicted_genes)
        
        # Evaluate this round
        round_eval = evaluator.evaluate(predicted_genes, step + 1)
        round_results.append(round_eval)
        
        # Get hits with their scores
        new_hits = evaluator.get_hits(predicted_genes)
        for hit_gene in new_hits:
            if hit_gene in ground_truth.index:
                score = ground_truth.loc[hit_gene].values[0] if hasattr(ground_truth.loc[hit_gene], 'values') else ground_truth.loc[hit_gene]
                all_hits.append((hit_gene, score))
        
        hits_history.append(len(all_hits))
        
        print(f"Predicted {len(predicted_genes)} genes")
        print(f"Hit rate: {round_eval['hit_rate']:.3f}")
        print(f"Total hits so far: {len(all_hits)}")
        if new_hits:
            print(f"New hits this round: {new_hits}")
        
        # Save predictions for this step
        np.save(
            os.path.join(log_dir, f"sampled_genes_{step + 1}.npy"),
            np.array(predicted_genes)
        )
    
    # Final evaluation
    final_results = evaluator.evaluate_multiple_rounds(
        [r["predicted_genes"] for r in round_results]
    )
    
    # Add hits tracking to results
    final_results["hits_progression"] = hits_history
    final_results["total_hits"] = len(all_hits)
    final_results["hit_genes_with_scores"] = [{"gene": gene, "score": float(score)} for gene, score in all_hits]
    
    # Save final results
    with open(os.path.join(log_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n=== Final Results ===")
    print(f"Mean hit rate: {final_results['aggregate']['mean_hit_rate']:.3f}")
    print(f"Total unique genes: {final_results['aggregate']['total_unique_genes']}")
    print(f"Total hits: {len(all_hits)}")
    
    return final_results