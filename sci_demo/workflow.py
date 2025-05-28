import asyncio
from typing import Any, Dict, List, Optional, Callable, Union

from generation import Generation
from prompt import Prompt
from oracle import Oracle


class Workflow:
    """
    A pipeline to orchestrate multi-stage science discovery powered by LLMs.
    Supports iterative generate-evaluate-feedback loops with dynamic prompts and metrics.
    Enhanced with history support for prompts and multi-round metrics.
    """
    def __init__(
        self,
        generator: Generation,
        oracle: Oracle,
        max_iterations: int = 3,
        stop_criteria: Optional[Callable[[Dict[str, Any]], bool]] = None,
        enable_history_in_prompts: bool = True,
        enable_multi_round_metrics: bool = True,
    ):
        """
        Args:
            generator: Instance of Generation class for inference.
            oracle: Instance of Oracle class for evaluation.
            max_iterations: Maximum number of optimization loops.
            stop_criteria: Function that takes last results dict and returns True to stop early.
            enable_history_in_prompts: Whether to automatically add history to prompts.
            enable_multi_round_metrics: Whether to use multi-round metrics when available.
        """
        self.generator = generator
        self.oracle = oracle
        self.max_iterations = max_iterations
        self.stop_criteria = stop_criteria
        self.enable_history_in_prompts = enable_history_in_prompts
        self.enable_multi_round_metrics = enable_multi_round_metrics

    async def run(
        self,
        prompt: Union[Prompt, Callable[[int, Dict[str, List[Any]]], Prompt]],
        reference: Any,
        gen_args: Optional[Dict[str, Any]] = None,
        metrics: Union[List[str], Callable[[int, Dict[str, List[Any]]], List[str]]] = None,
        history_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the iterative workflow with optional dynamic prompt and metrics.
        Enhanced with history support.

        Args:
            prompt: Either a Prompt instance or a function taking (iteration, history) and returning a Prompt.
            reference: Ground truth for evaluation.
            gen_args: Arguments for generation (max_tokens, temperature...).
            metrics: Either a list of metric names or a function taking (iteration, history) and returning list of metrics.
            history_context: Additional context to include in history (e.g., task_description).

        Returns:
            A dict containing history of prompts, outputs, scores, and metadata.
        """
        history: Dict[str, List[Any]] = {
            "prompts": [],
            "outputs": [],
            "scores": [],
            "raw_outputs": [],  # Store full generation outputs
            "iterations": [],
        }
        
        gen_args = gen_args or {}
        history_context = history_context or {}
        
        # metrics default: all registered (including multi-round if enabled)
        if self.enable_multi_round_metrics:
            default_metrics = self.oracle.list_metrics()
        else:
            default_metrics = self.oracle.list_single_round_metrics()

        for iteration in range(1, self.max_iterations + 1):
            print(f"Starting iteration {iteration}/{self.max_iterations}")
            
            # Determine prompt for this iteration
            if callable(prompt):
                current_prompt = prompt(iteration, history)
            else:
                current_prompt = prompt
            
            # Add history to prompt if enabled and prompt supports it
            if self.enable_history_in_prompts and hasattr(current_prompt, 'add_history'):
                # Create a copy to avoid modifying the original
                current_prompt = self._copy_prompt(current_prompt)
                
                # Add task description from context if available
                if history_context.get("task_description"):
                    current_prompt.add_vars(task_description=history_context["task_description"])
                
                # Add history information
                current_prompt.add_history(history, iteration)
            
            built_prompt = current_prompt.build()
            history["prompts"].append(built_prompt)
            history["iterations"].append(iteration)

            # Generate output
            print(f"Generating response for iteration {iteration}...")
            output = await self.generator.generate_async(
                built_prompt,
                **gen_args
            )
            
            # Store raw output and extract text
            history["raw_outputs"].append(output)
            text = output.get("text") if isinstance(output, dict) else output
            history["outputs"].append(text)
            print(f"Generated {len(text)} characters")

            # Determine metrics for this iteration
            if callable(metrics):
                current_metrics = metrics(iteration, history)
            else:
                current_metrics = metrics or default_metrics

            # Evaluate with appropriate method
            print(f"Evaluating with metrics: {current_metrics}")
            if self.enable_multi_round_metrics and iteration > 1:
                # Use multi-round evaluation for iterations after the first
                scores = self.oracle.compute_with_history(
                    text, reference, history, iteration, current_metrics
                )
            else:
                # Use single-round evaluation for first iteration or when multi-round is disabled
                single_round_metrics = [m for m in current_metrics if m in self.oracle.list_single_round_metrics()]
                scores = self.oracle.compute(text, reference, single_round_metrics)
            
            history["scores"].append(scores)
            print(f"Scores: {scores}")

            # Check stopping criteria
            context = {
                "iteration": iteration,
                "scores": scores,
                "output": text,
                "history": history,
                "raw_output": output,
            }
            if self.stop_criteria and self.stop_criteria(context):
                print(f"Stopping criteria met at iteration {iteration}")
                break

            # Feedback: add last score to prompt vars if dynamic prompt
            if hasattr(current_prompt, 'add_vars'):
                try:
                    current_prompt.add_vars(
                        last_score=scores,
                        last_output=text,
                        iteration_number=iteration
                    )
                except Exception as e:
                    print(f"Warning: Could not add feedback to prompt: {e}")

        # Add summary statistics
        final_result = {
            "history": history,
            "total_iterations": len(history["outputs"]),
            "final_scores": history["scores"][-1] if history["scores"] else {},
            "best_iteration": self._find_best_iteration(history),
            "trend_analysis": self._analyze_trends(history),
        }
        
        return final_result

    def _copy_prompt(self, prompt: Prompt) -> Prompt:
        """Create a copy of a prompt to avoid modifying the original."""
        try:
            # Create a new prompt with the same template and vars
            if hasattr(prompt, 'template') and hasattr(prompt, 'default_vars'):
                new_prompt = Prompt(
                    custom_template=prompt.template,
                    default_vars=prompt.default_vars.copy()
                )
                return new_prompt
        except Exception:
            pass
        return prompt

    def _find_best_iteration(self, history: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Find the iteration with the best overall performance."""
        if not history.get("scores"):
            return {"iteration": 1, "reason": "no_scores"}
        
        best_iteration = 1
        best_score = 0.0
        best_metric = "overall"
        
        for i, scores in enumerate(history["scores"], 1):
            if isinstance(scores, dict):
                # Calculate average score across all metrics
                avg_score = sum(scores.values()) / len(scores) if scores else 0.0
                if avg_score > best_score:
                    best_score = avg_score
                    best_iteration = i
        
        return {
            "iteration": best_iteration,
            "score": best_score,
            "metric": best_metric,
            "output": history["outputs"][best_iteration - 1] if history["outputs"] else ""
        }

    def _analyze_trends(self, history: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze trends across iterations."""
        if not history.get("scores") or len(history["scores"]) < 2:
            return {"trend_available": False}
        
        trends = {"trend_available": True}
        
        # Analyze each metric
        all_metrics = set()
        for score_dict in history["scores"]:
            if isinstance(score_dict, dict):
                all_metrics.update(score_dict.keys())
        
        for metric in all_metrics:
            metric_trends = self.oracle.compute_trend_metrics(history, metric)
            trends[f"{metric}_trends"] = metric_trends
        
        return trends

    def run_sync(
        self,
        prompt: Union[Prompt, Callable[[int, Dict[str, List[Any]]], Prompt]],
        reference: Any,
        gen_args: Optional[Dict[str, Any]] = None,
        metrics: Union[List[str], Callable[[int, Dict[str, List[Any]]], List[str]]] = None,
        history_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper around run().
        """
        return asyncio.run(
            self.run(
                prompt,
                reference,
                gen_args=gen_args,
                metrics=metrics,
                history_context=history_context
            )
        )

    def create_iterative_prompt(
        self, 
        task_description: str, 
        input_text: str,
        template_type: str = "iterative_with_feedback"
    ) -> Prompt:
        """
        Convenience method to create an iterative prompt with history support.
        
        Args:
            task_description: Description of the task
            input_text: The input text to process
            template_type: Type of template to use ("iterative", "iterative_with_feedback", "conversation")
            
        Returns:
            A Prompt instance configured for iterative use
        """
        return Prompt(
            template_name=template_type,
            default_vars={
                "task_description": task_description,
                "input_text": input_text,
                "additional_instructions": ""
            }
        )

    def create_dynamic_prompt_function(
        self,
        base_task: str,
        base_input: str,
        iteration_instructions: Optional[Dict[int, str]] = None
    ) -> Callable[[int, Dict[str, List[Any]]], Prompt]:
        """
        Create a dynamic prompt function that changes based on iteration and history.
        
        Args:
            base_task: Base task description
            base_input: Base input text
            iteration_instructions: Dict mapping iteration numbers to additional instructions
            
        Returns:
            A function that takes (iteration, history) and returns a Prompt
        """
        iteration_instructions = iteration_instructions or {}
        
        def prompt_function(iteration: int, history: Dict[str, List[Any]]) -> Prompt:
            # Choose template based on iteration
            if iteration == 1:
                template_type = "iterative"
            else:
                template_type = "iterative_with_feedback"
            
            # Get additional instructions for this iteration
            additional = iteration_instructions.get(iteration, "")
            if iteration > 1 and not additional:
                additional = "Please improve upon the previous attempts based on the feedback scores."
            
            prompt = Prompt(
                template_name=template_type,
                default_vars={
                    "task_description": base_task,
                    "input_text": base_input,
                    "additional_instructions": additional
                }
            )
            
            return prompt
        
        return prompt_function

# Example usage
if __name__ == "__main__":
    # Setup components
    gen = Generation(openai_api_key="YOUR_KEY")
    oracle = Oracle()

    # Register simple metrics
    def accuracy(pred, ref, **kwargs):
        return float(pred.strip().lower() == ref.strip().lower())

    def length_score(pred, ref, **kwargs):
        return min(len(pred.split()) / 50.0, 1.0)  # Normalize to 0-1

    oracle.register_metric("accuracy", accuracy)
    oracle.register_metric("length", length_score)
    
    # Register multi-round metrics
    from oracle import improvement_rate_metric, consistency_metric, convergence_metric
    oracle.register_multi_round_metric("improvement_rate", improvement_rate_metric)
    oracle.register_multi_round_metric("consistency", consistency_metric)
    oracle.register_multi_round_metric("convergence", convergence_metric)

    # Create workflow with history support
    workflow = Workflow(
        generator=gen,
        oracle=oracle,
        max_iterations=5,
        enable_history_in_prompts=True,
        enable_multi_round_metrics=True
    )

    # Method 1: Using convenience method
    prompt = workflow.create_iterative_prompt(
        task_description="Summarize scientific breakthroughs in protein folding",
        input_text="Recent advances in AI have revolutionized protein structure prediction.",
        template_type="iterative_with_feedback"
    )

    # Method 2: Using dynamic prompt function
    dynamic_prompt = workflow.create_dynamic_prompt_function(
        base_task="Write a scientific summary",
        base_input="Quantum computing and its applications in drug discovery",
        iteration_instructions={
            2: "Focus on clarity and technical accuracy",
            3: "Add more specific examples and citations",
            4: "Ensure the summary is accessible to a general audience"
        }
    )

    # Stop when accuracy reaches 0.9 or improvement rate is very low
    def stop_criteria(context):
        scores = context["scores"]
        iteration = context["iteration"]
        
        # Stop if accuracy is high enough
        if scores.get("accuracy", 0) >= 0.9:
            return True
        
        # Stop if improvement rate is very low after iteration 3
        if iteration >= 3 and scores.get("improvement_rate", 0) < 0.01:
            return True
        
        return False

    workflow.stop_criteria = stop_criteria

    # Run with history context
    result = workflow.run_sync(
        prompt=dynamic_prompt,
        reference="Quantum computing enables faster molecular simulations for drug discovery.",
        gen_args={"model": "gpt-4o", "max_tokens": 150, "temperature": 0.7},
        history_context={"task_description": "Scientific summarization task"}
    )
    
    print("=== Workflow Results ===")
    print(f"Total iterations: {result['total_iterations']}")
    print(f"Final scores: {result['final_scores']}")
    print(f"Best iteration: {result['best_iteration']}")
    print(f"Trend analysis: {result['trend_analysis']}")
    
    print("\n=== Final Output ===")
    if result['history']['outputs']:
        print(result['history']['outputs'][-1])
