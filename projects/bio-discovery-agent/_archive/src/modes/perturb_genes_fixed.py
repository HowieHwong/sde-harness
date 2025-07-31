"""Gene perturbation discovery mode with fixes for historical tracking, hits communication, and duplicate removal."""
import os
import sys
import json
import re
import numpy as np
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


def run_discovery_step_with_retry(llm_interface: BioLLMInterface,
                                prompt: str,
                                args: Any,
                                all_tested_genes: Set[str],
                                num_genes_needed: int,
                                enable_tools: bool = False,
                                is_pairs: bool = False,
                                max_retries: int = 20) -> Tuple[str, List[str]]:
    """Run a single discovery step with retry logic for duplicate removal."""
    collected_genes = []
    all_responses = []
    
    # Try up to max_retries times to get enough unique genes
    for retry in range(max_retries):
        # Generate response
        response = llm_interface.complete_text(
            prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        all_responses.append(response)
        
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
                        all_responses[-1] = response
            
            # Check for pathway search
            if args.reactome:
                gene_for_pathways = parse_tool_request(response, "Reactome Pathways")
                if gene_for_pathways:
                    pathways = BioDiscoveryTools.get_reactome_pathways(gene_for_pathways)
                    if pathways:
                        tool_info = f"\nPathways for {gene_for_pathways}: {', '.join(pathways)}"
                        response += tool_info
                        all_responses[-1] = response
        
        # Parse genes from response
        predicted_genes = parse_gene_solution(response, is_pairs=is_pairs)
        
        # Filter out already tested genes
        if is_pairs:
            # For pairs, convert to string format for comparison
            new_genes = [g for g in predicted_genes if f"{g[0]}_{g[1]}" not in all_tested_genes]
        else:
            new_genes = [g for g in predicted_genes if g not in all_tested_genes]
        
        # Add new unique genes to collection
        for gene in new_genes:
            if gene not in collected_genes:
                collected_genes.append(gene)
                if len(collected_genes) >= num_genes_needed:
                    break
        
        # Check if we have enough genes
        if len(collected_genes) >= num_genes_needed:
            break
        
        # Update prompt for next iteration
        genes_still_needed = num_genes_needed - len(collected_genes)
        if is_pairs:
            collected_str = [f"{g[0]} + {g[1]}" for g in collected_genes]
        else:
            collected_str = collected_genes
            
        prompt = f"""{prompt}

You have so far predicted {len(collected_genes)} out of the required {num_genes_needed} genes for this round. These were:
{', '.join(str(g) for g in collected_str)}

Please add {genes_still_needed} more genes to this list. Remember not to include previously tested genes.
DO NOT repeat any of the genes you just predicted above."""
    
    # Convert pairs back to string format if needed
    if is_pairs:
        collected_genes = [f"{g[0]}_{g[1]}" for g in collected_genes[:num_genes_needed]]
    else:
        collected_genes = collected_genes[:num_genes_needed]
    
    # Combine all responses for logging
    full_response = "\n\n--- Retry ---\n\n".join(all_responses)
    
    return full_response, collected_genes


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
    """Main gene perturbation discovery workflow with fixes."""
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
    
    # Setup logging
    log_dir = os.path.join(args.log_dir, f"{args.model}_{args.data_name}", args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Track all predictions and hits
    all_tested_genes = set()  # FIX 1: Track all historical genes
    all_hits = []  # FIX 2: Track hit genes
    hits_history = []  # Track hits progression
    round_results = []
    
    # Determine which prompt template to use and if pairs
    is_pairs = False
    if task_variant == "brief-NormanGI":
        initial_prompt_type = "perturb_genes_pairs_norman"
        follow_up_prompt_type = "follow_up"
        tool_section = None
        tool_description = None
        is_pairs = True
    elif task_variant == "brief-Horlbeck":
        initial_prompt_type = "perturb_genes_pairs_horlbeck"
        follow_up_prompt_type = "follow_up"
        tool_section = None
        tool_description = None
        is_pairs = True
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
            # Follow-up prompt with previous results
            # FIX 1: Include ALL previously tested genes
            all_tested_list = list(all_tested_genes)
            if is_pairs:
                # Convert pair format for display
                tested_display = [g.replace('_', ' + ') for g in all_tested_list]
            else:
                tested_display = all_tested_list
                
            observed = f"ALL genes tested so far across all rounds: {format_gene_list(tested_display)}"
            
            # Get results from last round
            last_round_genes = round_results[-1]["predicted_genes"]
            last_round_eval = round_results[-1]
            
            # Simulate results with hit information
            result = f"Results for the last round of {len(last_round_genes)} genes:\n"
            result += f"- Hit rate: {last_round_eval['hit_rate']:.3f}\n"
            result += f"- Precision: {last_round_eval['precision']:.3f}\n"
            
            # FIX 2: Explicitly communicate hit genes
            if all_hits:
                result += f"\nIMPORTANT: The following genes have been HITS (successful) so far: {', '.join(all_hits)}\n"
                result += f"These hit genes are particularly important as they show strong effects.\n"
            
            # Add progression information
            result += f"\nProgression of cumulative hits across rounds: {', '.join(map(str, hits_history))}"
            if len(hits_history) > 2 and hits_history[-1] - hits_history[-3] < 5:
                result += "\nNote: The number of hits has not increased much recently. Consider exploring different types of genes or pathways."
            
            result += f"\nYou have successfully identified {len(all_hits)} hits so far over all experiment cycles!"
            
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
        
        # FIX 3: Run discovery step with retry logic for duplicate removal
        response, predicted_genes = run_discovery_step_with_retry(
            llm_interface,
            prompt,
            args,
            all_tested_genes,
            args.num_genes,
            enable_tools=(args.gene_search or args.reactome),
            is_pairs=is_pairs
        )
        
        # Add critique if enabled
        if args.critique and predicted_genes:
            critique = BioDiscoveryTools.critique_solution(
                format_gene_list(predicted_genes),
                research_problem,
                args.model
            )
            response += f"\n\nCritique:\n{critique}"
        
        # Add literature review if enabled
        if args.lit_review and predicted_genes and len(predicted_genes) > 0:
            lit_review = BioDiscoveryTools.literature_search(
                predicted_genes[0],  # Review first gene as example
                research_problem
            )
            response += f"\n\nLiterature Review for {predicted_genes[0]}:\n{lit_review}"
        
        # Save response
        with open(os.path.join(log_dir, f"step_{step + 1}_response.txt"), "w") as f:
            f.write(response)
        
        # Update tracking
        all_tested_genes.update(predicted_genes)
        
        # Evaluate this round
        round_eval = evaluator.evaluate(predicted_genes, step + 1)
        round_results.append(round_eval)
        
        # Update hits (in real implementation, this would come from actual evaluation)
        # For now, simulate by checking ground truth
        new_hits = evaluator.get_hits(predicted_genes)
        all_hits.extend(new_hits)
        hits_history.append(len(all_hits))
        
        print(f"Predicted {len(predicted_genes)} genes")
        print(f"Hit rate: {round_eval['hit_rate']:.3f}")
        print(f"Total hits so far: {len(all_hits)}")
        
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
    final_results["hit_genes"] = list(all_hits)
    
    # Save final results
    with open(os.path.join(log_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n=== Final Results ===")
    print(f"Mean hit rate: {final_results['aggregate']['mean_hit_rate']:.3f}")
    print(f"Total unique genes: {final_results['aggregate']['total_unique_genes']}")
    print(f"Total hits: {len(all_hits)}")
    
    return final_results