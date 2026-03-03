"""
Classification mode for X-ray transient classification.

Implements the main workflow using SDE-Harness components.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from sde_harness.core import Generation, Oracle, Workflow, Prompt

from ..utils.data_loader import (
    load_transient,
    format_observation_for_prompt,
    get_ground_truth,
)
from ..utils.physics_utils import generate_feedback
from ..oracles.classification_oracle import ClassificationOracle
from ..prompts.templates import (
    CLASSIFY_TEMPLATE,
    ITERATIVE_TEMPLATE,
    PromptBuilder,
)


def run_classify(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the classification workflow on a single transient.
    
    Args:
        args: Command line arguments containing:
            - input: Path to transient JSON file
            - model: LLM model to use
            - iterations: Number of refinement iterations
            - temperature: Sampling temperature
            - max_tokens: Maximum tokens in response
            - output: Optional output file path
            - verbose: Whether to print verbose output
            
    Returns:
        Dictionary containing classification results and metrics
    """
    observation = load_transient(args.input)
    ground_truth = get_ground_truth(observation)
    
    if args.verbose:
        print(f"Loaded transient: {observation.get('transient_id', 'Unknown')}")
        print(f"Ground truth: {ground_truth.get('classification', 'Not provided')}")
    
    project_root = Path(__file__).parent.parent.parent
    generator = Generation(
        models_file=str(project_root / "models.yaml"),
        credentials_file=str(project_root / "credentials.yaml"),
        model_name=args.model,
    )
    
    custom_oracle = ClassificationOracle(observation)
    
    def wrap_metric(metric_func: Callable[[str], float]) -> Callable:
        """Wrap a metric function to match Oracle's expected signature."""
        def wrapped(prediction: str, reference: str, **kwargs) -> float:
            return metric_func(prediction)
        return wrapped
    
    oracle = Oracle()
    single_round_metrics = custom_oracle.get_single_round_metrics()
    for name, func in single_round_metrics.items():
        oracle.register_metric(name, wrap_metric(func))
    
    workflow = Workflow(
        generator=generator,
        oracle=oracle,
        max_iterations=args.iterations,
    )
    
    observation_text = format_observation_for_prompt(observation)
    prompt_builder = PromptBuilder(observation)
    
    def escape_braces(text: str) -> str:
        """Escape braces so Prompt.build() doesn't try to format them."""
        return text.replace("{", "{{").replace("}", "}}")
    
    initial_prompt_text = escape_braces(prompt_builder.classify())
    initial_prompt = Prompt(custom_template=initial_prompt_text)
    
    def dynamic_prompt(iteration: int, history: Dict[str, List[Any]]) -> Prompt:
        """
        Generate prompt for each iteration.
        
        First iteration uses the classification template.
        Subsequent iterations use the iterative template with feedback.
        """
        if iteration == 1:
            return initial_prompt
        
        outputs = history.get('outputs', [])
        if not outputs:
            return initial_prompt
        
        prev_response = outputs[-1]
        prev_scores = custom_oracle.evaluate(prev_response)
        feedback = generate_feedback(prev_scores)
        
        iterative_text = escape_braces(prompt_builder.iterate(prev_response, feedback))
        return Prompt(custom_template=iterative_text)
    
    if args.verbose:
        print(f"\nRunning {args.iterations} iteration(s) with model {args.model}...")
    
    results = workflow.run_sync(
        prompt=dynamic_prompt,
        reference=json.dumps(ground_truth),
        gen_args={
            "model_name": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
    )
    
    output = {
        'transient_id': observation.get('transient_id'),
        'model': args.model,
        'iterations': args.iterations,
        'ground_truth': ground_truth,
        'results': results,
    }
    
    history = results.get('history', {})
    outputs = history.get('outputs', [])
    scores_history = history.get('scores', [])
    
    if outputs:
        final_response = outputs[-1]
        final_scores = custom_oracle.evaluate(final_response)
        
        output['final_response'] = final_response
        output['final_scores'] = final_scores
        
        if args.verbose:
            print("\n=== Final Response ===")
            print(final_response[:500] + "..." if len(final_response) > 500 else final_response)
            print("\n=== Final Scores ===")
            print(f"Top-1 Correct: {final_scores.get('top1_correct', 0):.2f}")
            print(f"Alternatives Identified: {final_scores.get('alternatives_identified', 0):.2f}")
            print(f"Luminosity (Power Law): {final_scores.get('luminosity_powerlaw', 0):.2f}")
            print(f"Luminosity (Blackbody): {final_scores.get('luminosity_blackbody', 0):.2f}")
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        if args.verbose:
            print(f"\nResults saved to: {args.output}")
    
    return output


def add_classify_args(parser: argparse.ArgumentParser) -> None:
    """Add classify subcommand arguments to parser."""
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to transient observation JSON file'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='openai/gpt-4o',
        help='LLM model to use (default: openai/gpt-4o)'
    )
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=2,
        help='Number of refinement iterations (default: 2)'
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2000,
        help='Maximum tokens in response (default: 2000)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for results JSON'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output'
    )
