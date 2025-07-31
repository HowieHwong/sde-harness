#!/usr/bin/env python3
"""
CLI Integration Example

This example demonstrates how to create command-line interfaces using
the SDE-Harness CLIBase class for consistent CLI patterns.

Usage:
    python 01_cli_integration.py --help
    python 01_cli_integration.py --task summarize --input "Your text here"
    python 01_cli_integration.py --task optimize --target 42 --iterations 5
"""

import sys
import os
from typing import Dict, Any
import json

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from sde_harness.base import CLIBase, ProjectBase
from sde_harness.core import Workflow, Generation, Oracle, Prompt


class ScientificAssistantProject(ProjectBase):
    """Example project implementing a scientific assistant"""

    def _setup_project(self, **kwargs):
        """Setup project-specific components"""
        self.task_type = kwargs.get("task_type", "summarize")
        self.input_text = kwargs.get("input_text", "")
        self.target_value = kwargs.get("target_value", 42)
        self.max_iterations = kwargs.get("max_iterations", 3)

        # Setup task-specific metrics
        self.setup_task_metrics()

    def setup_task_metrics(self):
        """Setup metrics based on task type"""
        if self.task_type == "summarize":

            def summary_quality(prediction: str, reference: str, **kwargs) -> float:
                """Evaluate summary quality"""
                if not prediction.strip():
                    return 0.0

                # Simple quality indicators
                word_count = len(prediction.split())
                has_key_points = any(
                    word in prediction.lower()
                    for word in ["key", "main", "important", "summary"]
                )
                appropriate_length = 50 <= word_count <= 200

                score = 0.0
                if has_key_points:
                    score += 0.4
                if appropriate_length:
                    score += 0.4
                if word_count > 10:  # Not too short
                    score += 0.2

                return score

            self.oracle.register_metric("summary_quality", summary_quality)

        elif self.task_type == "optimize":

            def optimization_progress(
                prediction: str, reference: str, **kwargs
            ) -> float:
                """Evaluate optimization progress"""
                try:
                    # Extract numbers from prediction
                    import re

                    numbers = re.findall(r"\d+(?:\.\d+)?", prediction)
                    if not numbers:
                        return 0.0

                    closest_value = min(
                        numbers, key=lambda x: abs(float(x) - self.target_value)
                    )
                    error = abs(float(closest_value) - self.target_value)

                    # Normalize error (closer to target = higher score)
                    max_error = self.target_value + 50
                    return max(0.0, 1.0 - (error / max_error))

                except:
                    return 0.0

            self.oracle.register_metric("optimization_progress", optimization_progress)

    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the project workflow"""
        if self.task_type == "summarize":
            return self.run_summarization()
        elif self.task_type == "optimize":
            return self.run_optimization()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def run_summarization(self) -> Dict[str, Any]:
        """Run text summarization workflow"""
        prompt = Prompt(
            template_name="summarize", default_vars={"max_length": "150 words"}
        )

        built_prompt = prompt.build({"input_text": self.input_text})

        workflow = Workflow(
            generator=self.generator,
            oracle=self.oracle,
            max_iterations=self.max_iterations,
        )

        return workflow.run_sync(
            prompt=prompt,  # Pass the Prompt object, not the built string
            reference=self.input_text,
            gen_args={
                "model_name": "openai/gpt-4o-mini",
                "max_tokens": 200,
                "temperature": 0.3,
            },
        )

    def run_optimization(self) -> Dict[str, Any]:
        """Run numerical optimization workflow"""

        def create_optimization_prompt(iteration: int, history: Dict) -> str:
            if iteration == 1:
                return f"""Find mathematical expressions that equal or are very close to {self.target_value}.
                
Provide 3 different mathematical expressions using basic operations (+, -, *, /).
Each expression should evaluate to a number close to {self.target_value}.

Examples of good expressions: "20 + 22", "84 / 2", "7 * 6"

Your expressions:"""
            else:
                prev_output = history["outputs"][-1] if history.get("outputs") else ""
                prev_score = (
                    history["scores"][-1].get("optimization_progress", 0.0)
                    if history.get("scores")
                    else 0.0
                )

                return f"""Target: {self.target_value}
Previous attempt: {prev_output[:100]}...
Previous score: {prev_score:.3f}

Improve on the previous expressions. Try to get even closer to {self.target_value}.
Provide 3 new mathematical expressions:"""

        workflow = Workflow(
            generator=self.generator,
            oracle=self.oracle,
            max_iterations=self.max_iterations,
        )

        return workflow.run_sync(
            prompt=create_optimization_prompt,
            reference=str(self.target_value),
            gen_args={
                "model_name": "openai/gpt-4o-mini",
                "max_tokens": 150,
                "temperature": 0.7,
            },
        )


class ScientificAssistantCLI(CLIBase):
    """CLI for the Scientific Assistant project"""

    def _add_project_arguments(self, parser):
        """Add project-specific CLI arguments"""
        # Task selection
        parser.add_argument(
            "--task",
            choices=["summarize", "optimize"],
            default="summarize",
            help="Task to perform (default: summarize)",
        )

        # Task-specific arguments
        parser.add_argument(
            "--input", "--input-text", help="Input text for summarization task"
        )

        parser.add_argument(
            "--target",
            "--target-value",
            type=float,
            default=42,
            help="Target value for optimization task (default: 42)",
        )

        parser.add_argument(
            "--iterations",
            "--max-iterations",
            type=int,
            default=3,
            help="Maximum number of iterations (default: 3)",
        )

        # Output options
        parser.add_argument(
            "--format",
            choices=["text", "json"],
            default="text",
            help="Output format (default: text)",
        )

        parser.add_argument("--save-output", help="Save output to file")

    def run_command(self, args) -> Dict[str, Any]:
        """Run the command with parsed arguments"""
        # Validate arguments based on task
        if args.task == "summarize" and not args.input:
            print("❌ Error: --input is required for summarization task")
            print("💡 Usage: python 01_cli_integration.py --task summarize --input 'Your text to summarize here'")
            sys.exit(1)

        # Create and run project
        # If default paths are used, prepend ../ to look in parent directory
        models_file = args.models_file
        credentials_file = args.credentials_file
        if models_file == "config/models.yaml":
            models_file = "../config/models.yaml"
        if credentials_file == "config/credentials.yaml":
            credentials_file = "../config/credentials.yaml"
            
        project = ScientificAssistantProject(
            models_file=models_file,
            credentials_file=credentials_file,
            task_type=args.task,
            input_text=args.input or "",
            target_value=args.target,
            max_iterations=args.iterations,
        )

        print(f"🚀 Running {args.task} task...")
        if args.verbose:
            print(f"Configuration:")
            print(f"  Task: {args.task}")
            if args.task == "summarize":
                print(f"  Input length: {len(args.input or '')} characters")
            elif args.task == "optimize":
                print(f"  Target value: {args.target}")
            print(f"  Max iterations: {args.iterations}")
            print()

        # Run the project
        result = project.run()

        # Format and display results
        self.display_results(result, args)

        # Save output if requested
        if args.save_output:
            self.save_results(result, args.save_output, args.format)

        return result

    def display_results(self, result: Dict[str, Any], args):
        """Display results based on format preference"""
        if args.format == "json":
            # JSON format
            print(json.dumps(result, indent=2, default=str))
        else:
            # Human-readable text format
            print("\n" + "=" * 60)
            print("RESULTS")
            print("=" * 60)

            outputs = result.get("outputs", [])
            scores = result.get("iteration_scores", [])
            final_scores = result.get("final_scores", {})

            print(f"Task: {args.task}")
            print(f"Iterations completed: {len(outputs)}")
            print()

            # Show final output
            if outputs:
                print("Final Output:")
                print("-" * 20)
                print(outputs[-1])
                print()

            # Show scores
            print("Evaluation Scores:")
            print("-" * 20)
            for metric, score in final_scores.items():
                print(f"  {metric}: {score:.3f}")

            # Show iteration progress if verbose
            if args.verbose and len(outputs) > 1:
                print("\nIteration Progress:")
                print("-" * 20)
                for i, (output, score) in enumerate(zip(outputs, scores), 1):
                    print(f"Iteration {i}:")
                    print(f"  Output: {output[:100]}...")
                    print(f"  Scores: {score}")
                    print()

    def save_results(self, result: Dict[str, Any], filename: str, format: str):
        """Save results to file"""
        try:
            with open(filename, "w") as f:
                if format == "json":
                    json.dump(result, f, indent=2, default=str)
                else:
                    f.write("SDE-Harness Results\n")
                    f.write("=" * 20 + "\n\n")

                    outputs = result.get("outputs", [])
                    final_scores = result.get("final_scores", {})

                    f.write(f"Iterations: {len(outputs)}\n")
                    f.write(f"Final Output:\n{outputs[-1] if outputs else 'None'}\n\n")
                    f.write("Scores:\n")
                    for metric, score in final_scores.items():
                        f.write(f"  {metric}: {score:.3f}\n")

            print(f"💾 Results saved to {filename}")

        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """Main function"""
    cli = ScientificAssistantCLI("Scientific Assistant")

    try:
        result = cli.main()

        # Exit with success
        print("\n✅ Task completed successfully!")
        print("\n💡 Try these examples:")
        print(
            "  python 01_cli_integration.py --task summarize --input 'Machine learning is a powerful tool for data analysis and prediction.'"
        )
        print(
            "  python 01_cli_integration.py --task optimize --target 100 --iterations 5 --verbose"
        )
        print(
            "  python 01_cli_integration.py --task summarize --input 'Your text' --format json --save-output results.json"
        )

    except KeyboardInterrupt:
        print("\n⚠️  Task interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")

        # Provide helpful error messages
        if "models_file" in str(e).lower() or "credentials" in str(e).lower():
            print("💡 Tip: Make sure you have models.yaml and credentials.yaml in the project root")
            print("   You may need to copy from the template files:")
            print("   cp models.template.yaml models.yaml")
            print("   cp credentials.template.yaml credentials.yaml")
        elif "input" in str(e).lower():
            print("💡 Tip: Use --input 'your text here' for summarization tasks")
        elif "api" in str(e).lower() or "key" in str(e).lower():
            print("💡 Tip: Check your API keys in config/credentials.yaml")
        elif "model" in str(e).lower():
            print("💡 Tip: Check your model configuration in config/models.yaml")

        sys.exit(1)


if __name__ == "__main__":
    main()
