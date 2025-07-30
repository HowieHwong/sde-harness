#!/usr/bin/env python3
"""
Simple Optimization Example

This example demonstrates iterative optimization using the SDE-Harness framework.
The goal is to find mathematical expressions that evaluate to a target value.

Usage:
    python run_optimization.py
    python run_optimization.py --target 42 --max-iterations 10
    python run_optimization.py --config custom_config.yaml
"""

import sys
import os
import json
import yaml
import argparse
import re
from typing import Dict, Any, List, Tuple
import random

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from sde_harness.core import Workflow, Generation, Oracle, Prompt


class MathExpressionOptimizer:
    """Main optimizer class for mathematical expressions"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the optimizer with configuration"""
        self.config = self.load_config(config_path)
        self.setup_components()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Config file {config_path} not found, using defaults")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "optimization": {
                "target_value": 42,
                "max_iterations": 10,
                "tolerance": 0.1,
                "initial_expressions": ["5 + 3", "10 * 2", "7 + 8"],
                "allowed_operations": ["+", "-", "*", "/"],
                "max_terms": 5,
                "max_value": 100,
            },
            "generation": {
                "model_name": "openai/gpt-4o-mini",
                "max_tokens": 200,
                "temperature": 0.7,
            },
            "workflow": {"enable_multi_round_metrics": True},
            "output": {"verbose": True, "show_intermediate_results": True},
        }

    def setup_components(self):
        """Set up SDE-Harness components"""
        # Initialize generator
        self.generator = Generation(
            models_file=f"{project_root}/models.yaml",
            credentials_file=f"{project_root}/credentials.yaml",
        )

        # Initialize oracle with metrics
        self.oracle = Oracle()
        self.setup_metrics()

        # Initialize workflow
        self.workflow = Workflow(
            generator=self.generator,
            oracle=self.oracle,
            max_iterations=self.config["optimization"]["max_iterations"],
            enable_multi_round_metrics=self.config["workflow"][
                "enable_multi_round_metrics"
            ],
        )

    def setup_metrics(self):
        """Set up evaluation metrics"""
        target = self.config["optimization"]["target_value"]
        tolerance = self.config["optimization"]["tolerance"]

        # Clear existing metrics to avoid re-registration errors
        self.oracle = Oracle()

        def error_distance(prediction: str, reference: Any, **kwargs) -> float:
            """Calculate distance from target value using the best expression found"""
            try:
                # Parse expressions from the generated text
                expressions = self.parse_generated_expressions(prediction)
                
                if not expressions:
                    return 1.0  # Maximum error if no valid expressions found
                
                # Find the best expression (closest to target)
                best_error = float('inf')
                for expr in expressions:
                    value = self.evaluate_expression(expr)
                    if value is not None:
                        error = abs(value - target)
                        if error < best_error:
                            best_error = error
                
                if best_error == float('inf'):
                    return 1.0  # No valid expressions found
                
                # Normalize error (0 = perfect, 1 = very bad)
                max_error = target + 50  # Reasonable max error
                return min(1.0, best_error / max_error)
            except:
                return 1.0

        def expression_validity(prediction: str, reference: Any, **kwargs) -> float:
            """Check if any valid expressions are found in the generated text"""
            try:
                expressions = self.parse_generated_expressions(prediction)
                if not expressions:
                    return 0.0
                
                # Check if at least one expression is valid
                valid_count = 0
                for expr in expressions:
                    if self.evaluate_expression(expr) is not None:
                        valid_count += 1
                
                return valid_count / len(expressions) if expressions else 0.0
            except:
                return 0.0

        def complexity_score(prediction: str, reference: Any, **kwargs) -> float:
            """Score based on expression complexity (simpler is better)"""
            try:
                expressions = self.parse_generated_expressions(prediction)
                if not expressions:
                    return 0.0
                
                # Use the best expression (first valid one)
                best_expr = None
                for expr in expressions:
                    if self.evaluate_expression(expr) is not None:
                        best_expr = expr
                        break
                
                if not best_expr:
                    return 0.0
                
                # Count operations and terms
                operations = sum(1 for op in ["+", "-", "*", "/"] if op in best_expr)
                terms = len(re.findall(r"\d+", best_expr))

                # Prefer moderate complexity
                ideal_ops = 2
                ideal_terms = 3

                op_penalty = abs(operations - ideal_ops) * 0.1
                term_penalty = abs(terms - ideal_terms) * 0.05

                return max(0.0, 1.0 - op_penalty - term_penalty)
            except:
                return 0.0

        def improvement_rate(
            history: Dict, reference: Any, current_iteration: int, **kwargs
        ) -> float:
            """Track improvement rate over iterations"""
            if "scores" not in history or len(history["scores"]) < 2:
                return 0.0

            # Get error distances from history
            errors = [score.get("error_distance", 1.0) for score in history["scores"]]
            if len(errors) < 2:
                return 0.0

            # Calculate improvement (reduction in error)
            initial_error = errors[0]
            current_error = errors[-1]

            if initial_error == 0:
                return 0.0

            improvement = (initial_error - current_error) / initial_error
            return max(0.0, improvement)

        def convergence_trend(
            history: Dict, reference: Any, current_iteration: int, **kwargs
        ) -> float:
            """Measure convergence trend"""
            if "scores" not in history or len(history["scores"]) < 3:
                return 0.5  # Neutral trend

            errors = [score.get("error_distance", 1.0) for score in history["scores"]]
            if len(errors) < 3:
                return 0.5

            # Check if errors are generally decreasing
            recent_errors = errors[-3:]
            is_improving = all(
                recent_errors[i] >= recent_errors[i + 1]
                for i in range(len(recent_errors) - 1)
            )

            return 1.0 if is_improving else 0.0

        # Register metrics
        self.oracle.register_metric("error_distance", error_distance)
        self.oracle.register_metric("expression_validity", expression_validity)
        self.oracle.register_metric("complexity_score", complexity_score)
        self.oracle.register_multi_round_metric("improvement_rate", improvement_rate)
        self.oracle.register_multi_round_metric("convergence_trend", convergence_trend)

    def evaluate_expression(self, expression: str) -> float:
        """Safely evaluate a mathematical expression"""
        try:
            # Clean the expression
            expression = expression.strip()

            # Basic validation
            if not re.match(r"^[0-9+\-*/().\s]+$", expression):
                return None

            # Replace common issues
            expression = expression.replace(" ", "")

            # Evaluate safely (in practice, use a proper math parser)
            # This is simplified for the example
            try:
                result = eval(expression)
                return float(result) if isinstance(result, (int, float)) else None
            except:
                return None
        except:
            return None

    def create_optimization_prompt(self, iteration: int, history: Dict) -> Prompt:
        """Create dynamic prompt based on optimization state"""
        target = self.config["optimization"]["target_value"]
        tolerance = self.config["optimization"]["tolerance"]
        operations = " ".join(self.config["optimization"]["allowed_operations"])

        if iteration == 1:
            # Initial prompt
            initial_expressions = self.config["optimization"]["initial_expressions"]
            expressions_text = ", ".join(f'"{expr}"' for expr in initial_expressions)

            template = f"""You are helping optimize mathematical expressions to reach a target value of {target}.

Current expressions: {expressions_text}

For each expression, evaluate it and see how close it gets to {target}. Then suggest 3 new mathematical expressions that might get closer to the target.

Use only these operations: {operations}
Keep expressions simple with numbers between 1-100.
Provide only the mathematical expressions, one per line.

New expressions:"""

            return Prompt(custom_template=template)

        else:
            # Iterative improvement prompt
            previous_output = history["outputs"][-1] if history.get("outputs") else ""
            previous_scores = history["scores"][-1] if history.get("scores") else {}

            error_distance = previous_scores.get("error_distance", 1.0)
            current_best_error = error_distance * (target + 50)  # Denormalize

            # Extract expressions from previous output
            previous_expressions = [
                line.strip() for line in previous_output.split("\n") if line.strip()
            ]

            feedback = ""
            if current_best_error > tolerance:
                if current_best_error > target * 0.5:
                    feedback = "The expressions are still quite far from the target. Try different approaches."
                else:
                    feedback = "Getting closer! Try small modifications to the better expressions."
            else:
                feedback = "Very close! Fine-tune the expressions."

            template = f"""You are optimizing mathematical expressions to reach a target value of {target}.

Previous expressions tried: {', '.join(previous_expressions[:3])}
Current best error: {current_best_error:.2f}

{feedback}

Suggest 3 new mathematical expressions that might get even closer to {target}.
Use only these operations: {operations}
Keep expressions simple with numbers between 1-100.

New expressions:"""

            return Prompt(custom_template=template)

    def parse_generated_expressions(self, generated_text: str) -> List[str]:
        """Parse expressions from generated text"""
        lines = generated_text.strip().split("\n")
        expressions = []

        for line in lines:
            # Clean up the line
            line = line.strip()

            # Skip empty lines and common prefixes
            if not line or line.lower().startswith(("here", "new", "expressions", "-")):
                continue

            # Remove numbering and bullet points
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[\-\*]\s*", "", line)

            # Extract mathematical expression
            # Look for pattern with numbers and operations
            match = re.search(r"([0-9+\-*/().\s]+)", line)
            if match:
                expr = match.group(1).strip()
                if expr and self.evaluate_expression(expr) is not None:
                    expressions.append(expr)

        # Fallback: if no valid expressions found, generate some simple ones
        if not expressions:
            expressions = self.generate_fallback_expressions()

        return expressions[:3]  # Return top 3

    def generate_fallback_expressions(self) -> List[str]:
        """Generate fallback expressions if parsing fails"""
        target = self.config["optimization"]["target_value"]

        # Simple expressions around the target
        fallbacks = [
            f"{target - 5} + 5",
            f"{target + 3} - 3",
            f"{target // 2} * 2" if target % 2 == 0 else f"{(target + 1) // 2} * 2 - 1",
        ]

        return fallbacks

    def run_optimization(self, custom_target: int = None) -> Dict[str, Any]:
        """Run the optimization process"""
        if custom_target:
            self.config["optimization"]["target_value"] = custom_target
            self.setup_metrics()  # Refresh metrics with new target

        target = self.config["optimization"]["target_value"]
        tolerance = self.config["optimization"]["tolerance"]

        print(f"🎯 Simple Optimization Example")
        print("=" * 50)
        print(f"Target Value: {target}")
        print(f"Tolerance: ±{tolerance}")
        print(f"Max Iterations: {self.config['optimization']['max_iterations']}")
        print()

        # Custom stop criteria
        def stop_criteria(stop_context: Dict) -> bool:
            history = stop_context.get("history", {})
            current_iteration = stop_context.get("current_iteration", 0)

            if not history.get("scores"):
                return False

            latest_scores = history["scores"][-1]
            error_distance = latest_scores.get("error_distance", 1.0)
            actual_error = error_distance * (target + 50)  # Denormalize

            # Stop if we've reached the target within tolerance
            target_achieved = actual_error <= tolerance

            if target_achieved and self.config["output"]["verbose"]:
                print(f"🎉 Target achieved! Error: {actual_error:.3f}")

            return target_achieved

        # Set up workflow with custom stop criteria
        self.workflow.stop_criteria = stop_criteria

        try:
            # Run the optimization workflow
            result = self.workflow.run_sync(
                prompt=self.create_optimization_prompt,
                reference=target,
                gen_args=self.config["generation"],
            )

            # Process and display results
            self.display_results(result)

            return result

        except Exception as e:
            print(f"❌ Error during optimization: {e}")
            return {"error": str(e)}

    def display_results(self, result: Dict[str, Any]):
        """Display optimization results"""
        print("\n" + "=" * 50)
        print("OPTIMIZATION RESULTS")
        print("=" * 50)

        target = self.config["optimization"]["target_value"]
        tolerance = self.config["optimization"]["tolerance"]

        if "error" in result:
            print(f"❌ Optimization failed: {result['error']}")
            return

        # Handle different result formats
        if 'outputs' in result:
            outputs = result.get("outputs", [])
            scores = result.get("iteration_scores", [])
        elif 'history' in result:
            outputs = result['history'].get('outputs', [])
            scores = result['history'].get('scores', [])
        else:
            outputs = []
            scores = []
            
        if 'final_scores' in result:
            final_scores = result.get("final_scores", {})
        elif 'history' in result and 'scores' in result['history'] and result['history']['scores']:
            final_scores = result['history']['scores'][-1]
        else:
            final_scores = {}

        print(f"Total Iterations: {len(outputs)}")
        print(f"Target Value: {target}")
        print()

        # Show iteration-by-iteration progress
        if self.config["output"]["show_intermediate_results"]:
            print("Iteration Progress:")
            print("-" * 30)

            for i, (output, score) in enumerate(zip(outputs, scores), 1):
                expressions = self.parse_generated_expressions(output)
                error_distance = score.get("error_distance", 1.0)
                actual_error = error_distance * (target + 50)
                validity = score.get("expression_validity", 0.0)

                print(f"Iteration {i}:")
                print(f"  Generated expressions: {expressions}")

                # Evaluate each expression
                best_expr = None
                best_value = None
                best_error = float("inf")

                for expr in expressions:
                    value = self.evaluate_expression(expr)
                    if value is not None:
                        error = abs(value - target)
                        if error < best_error:
                            best_error = error
                            best_expr = expr
                            best_value = value

                if best_expr:
                    print(f"  Best: {best_expr} = {best_value}")
                    print(f"  Error: {best_error:.3f}")
                else:
                    print(f"  No valid expressions found")

                print(f"  Scores: error={error_distance:.3f}, validity={validity:.3f}")
                print()

        # Final summary
        print("Final Summary:")
        print("-" * 20)

        final_error = final_scores.get("error_distance", 1.0) * (target + 50)
        improvement = final_scores.get("improvement_rate", 0.0)
        convergence = final_scores.get("convergence_trend", 0.0)

        success = final_error <= tolerance

        print(f"Success: {'✅ Yes' if success else '❌ No'}")
        print(f"Final Error: {final_error:.3f}")
        print(f"Improvement Rate: {improvement:.3f}")
        print(f"Convergence Trend: {convergence:.3f}")

        if success:
            print(
                f"\n🎉 Successfully found expressions within {tolerance} of target {target}!"
            )
        else:
            print(f"\n🔄 Try running again or adjusting parameters for better results.")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Simple Optimization Example")
    parser.add_argument("--target", type=int, help="Target value to optimize for")
    parser.add_argument(
        "--max-iterations", type=int, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Configuration file path"
    )
    parser.add_argument("--operations", help="Allowed operations (e.g., '+-*/')")
    parser.add_argument("--tolerance", type=float, help="Success tolerance")

    args = parser.parse_args()

    # Initialize optimizer
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"⚠️  Config file {config_path} not found in current directory")
        print("Using default configuration...")

    optimizer = MathExpressionOptimizer(config_path)

    # Apply CLI overrides
    if args.max_iterations:
        optimizer.config["optimization"]["max_iterations"] = args.max_iterations
    if args.operations:
        optimizer.config["optimization"]["allowed_operations"] = list(args.operations)
    if args.tolerance:
        optimizer.config["optimization"]["tolerance"] = args.tolerance

    # Run optimization
    result = optimizer.run_optimization(custom_target=args.target)

    # Exit with appropriate code
    if "error" in result:
        sys.exit(1)
    else:
        final_error = result.get("final_scores", {}).get("error_distance", 1.0)
        target = optimizer.config["optimization"]["target_value"]
        tolerance = optimizer.config["optimization"]["tolerance"]
        actual_error = final_error * (target + 50)

        if actual_error <= tolerance:
            print("\n💡 Next Steps:")
            print("- Try a different target value")
            print("- Experiment with different operations")
            print("- Modify the prompt templates")
            print("- Add new evaluation metrics")
            sys.exit(0)
        else:
            print("\n💡 Suggestions for improvement:")
            print("- Increase max_iterations in config")
            print("- Adjust temperature for more creative solutions")
            print("- Modify the prompt to be more specific")
            print("- Add more sophisticated evaluation metrics")
            sys.exit(1)


if __name__ == "__main__":
    main()
