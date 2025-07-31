#!/usr/bin/env python3
"""
Backward compatibility wrapper for research_assistant.py
Maps old command line arguments to new CLI structure.
"""
import sys
import argparse


def main():
    """Main compatibility wrapper."""
    # Parse original arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="perturb-genes-brief-Horlbeck",
                        help="task name")
    parser.add_argument("--log_dir", type=str, default="logs", help="script")
    parser.add_argument("--folder_name", type=str, default="temp", help="temp folder name")
    parser.add_argument("--run_name", type=str, default="exp", help="script name")
    parser.add_argument("--data_name", type=str, default='Horlbeck',
                        help="dataset name")
    parser.add_argument("--steps", type=int, default=6, help="number of steps")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--model", type=str, default='claude-v1',
                        help="LLM choice")
    parser.add_argument("--python", type=str, default="/lfs/turing2/0/qhwang/miniconda3/envs/llm/bin/python", 
                        help="python command")
    parser.add_argument("--continue_research", type=str, default=None, help="continue from a previous run")
    parser.add_argument("--interactive_interval", type=int, default=None, help="interactive interval")
    parser.add_argument("--enable_help", type=bool, default=False, help="enable help")
    parser.add_argument("--use_gpt4", type=bool, default=False, help="use gpt4")
    parser.add_argument("--num_genes", type=int, default=128, help="number of predicted genes")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--lit_review", type=bool, default=False, help="enable lit review")
    parser.add_argument("--critique", type=bool, default=False, help="enable critique")
    parser.add_argument("--reactome", type=bool, default=False, help="enable reactome")
    parser.add_argument("--gene_search", type=bool, default=False, help="enable gene search")
    parser.add_argument("--csv_path", type=str, default="./", help="path to csv files")
    
    args = parser.parse_args()
    
    # Map old model names to new format
    model_mapping = {
        'claude-v1': 'anthropic/claude-3-5-sonnet-20240620',
        'claude-3-5-sonnet-20240620': 'anthropic/claude-3-5-sonnet-20240620',
        'gpt-4o': 'openai/gpt-4o-2024-08-06',
        'gpt-4': 'openai/gpt-4',
        'gpt-3.5-turbo': 'openai/gpt-3.5-turbo'
    }
    
    mapped_model = model_mapping.get(args.model, args.model)
    
    # Parse task to extract variant and dataset
    task_parts = args.task.split('-')
    if args.task.startswith("perturb-genes"):
        mode = "perturb-genes"
        
        # Determine task variant
        if "brief-NormanGI" in args.task:
            task_variant = "brief-NormanGI"
        elif "brief-Horlbeck" in args.task:
            task_variant = "brief-Horlbeck"
        elif "brief" in args.task:
            task_variant = "brief"
        else:
            task_variant = "full"
        
        # Extract dataset name from task if not provided
        if args.data_name == 'Horlbeck' and '-' in args.task:
            # Try to extract dataset from task name
            task_suffix = args.task.split('-')[-1]
            if task_suffix not in ['genes', 'brief', 'NormanGI']:
                args.data_name = task_suffix
    else:
        # For other tasks, just use perturb-genes with default variant
        mode = "perturb-genes"
        task_variant = "brief"
    
    # Build new CLI command
    new_argv = [
        "cli.py",
        mode,
        "--task-variant", task_variant,
        "--data-name", args.data_name,
        "--steps", str(args.steps),
        "--num-genes", str(args.num_genes),
        "--run-name", args.run_name,
        "--folder-name", args.folder_name,
        "--log-dir", args.log_dir,
        "--model", mapped_model,
        "--temperature", str(args.temperature),
        "--csv-path", args.csv_path
    ]
    
    # Add tool flags
    if args.lit_review:
        new_argv.append("--lit-review")
    if args.critique:
        new_argv.append("--critique")
    if args.reactome:
        new_argv.append("--reactome")
    if args.gene_search:
        new_argv.append("--gene-search")
    
    # Replace sys.argv and import cli
    sys.argv = new_argv
    
    # Import and run CLI
    from cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()