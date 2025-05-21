import asyncio
from typing import Any, Dict, List, Optional, Callable, Union

from generation import Generation
from prompt import Prompt
from oracle import Oracle


class Workflow:
    """
    A pipeline to orchestrate multi-stage science discovery powered by LLMs.
    Supports iterative generate-evaluate-feedback loops with dynamic prompts and metrics.
    """
    def __init__(
        self,
        generator: Generation,
        oracle: Oracle,
        max_iterations: int = 3,
        stop_criteria: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        """
        Args:
            generator: Instance of Generation class for inference.
            oracle: Instance of Oracle class for evaluation.
            max_iterations: Maximum number of optimization loops.
            stop_criteria: Function that takes last results dict and returns True to stop early.
        """
        self.generator = generator
        self.oracle = oracle
        self.max_iterations = max_iterations
        self.stop_criteria = stop_criteria

    async def run(
        self,
        prompt: Union[Prompt, Callable[[int, Dict[str, List[Any]]], Prompt]],
        reference: Any,
        gen_args: Optional[Dict[str, Any]] = None,
        metrics: Union[List[str], Callable[[int, Dict[str, List[Any]]], List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the iterative workflow with optional dynamic prompt and metrics.

        Args:
            prompt: Either a Prompt instance or a function taking (iteration, history) and returning a Prompt.
            reference: Ground truth for evaluation.
            gen_args: Arguments for generation (max_tokens, temperature...).
            metrics: Either a list of metric names or a function taking (iteration, history) and returning list of metrics.

        Returns:
            A dict containing history of prompts, outputs, and scores.
        """
        history: Dict[str, List[Any]] = {
            "prompts": [],
            "outputs": [],
            "scores": []
        }
        gen_args = gen_args or {}
        # metrics default: all registered
        default_metrics = list(self.oracle.metrics.keys())

        for iteration in range(1, self.max_iterations + 1):
            # Determine prompt for this iteration
            if callable(prompt):
                current_prompt = prompt(iteration, history)
            else:
                current_prompt = prompt
            built_prompt = current_prompt.build()
            history["prompts"].append(built_prompt)

            # Generate output
            output = await self.generator.generate_async(
                built_prompt,
                **gen_args
            )
            text = output.get("text") if isinstance(output, dict) else output
            history["outputs"].append(text)

            # Determine metrics for this iteration
            if callable(metrics):
                current_metrics = metrics(iteration, history)
            else:
                current_metrics = metrics or default_metrics

            # Evaluate
            scores = self.oracle.compute(text, reference, current_metrics)
            history["scores"].append(scores)

            # Check stopping
            context = {
                "iteration": iteration,
                "scores": scores,
                "output": text,
                "history": history
            }
            if self.stop_criteria and self.stop_criteria(context):
                break

            # Feedback: add last score to prompt vars if dynamic prompt
            try:
                current_prompt.add_vars(last_score=scores)
            except Exception:
                pass

        return history

    def run_sync(
        self,
        prompt: Union[Prompt, Callable[[int, Dict[str, List[Any]]], Prompt]],
        reference: Any,
        gen_args: Optional[Dict[str, Any]] = None,
        metrics: Union[List[str], Callable[[int, Dict[str, List[Any]]], List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper around run().
        """
        return asyncio.run(
            self.run(
                prompt,
                reference,
                gen_args=gen_args,
                metrics=metrics
            )
        )

# Example usage
if __name__ == "__main__":
    # Setup components
    gen = Generation(api_key="YOUR_KEY", hf_model_name="meta-llama/Llama-3-7B")
    oracle = Oracle()

    # Register simple metrics
    def accuracy(pred, ref, **kwargs):
        return float(pred.strip() == ref.strip())

    def length_score(pred, ref, **kwargs):
        return len(pred.split())

    oracle.register_metric("accuracy", accuracy)
    oracle.register_metric("length", length_score)

    # Dynamic prompt function
    def prompt_fn(iteration, history):
        base_text = "Recent breakthroughs in protein folding."
        p = Prompt(template_name="summarize", default_vars={"input_text": base_text})
        # adjust instructions based on iteration
        if iteration > 1:
            p.add_vars(additional="Consider precision and details.")
            p.template += "\nAdditional instructions: {additional}\n"
        return p

    # Dynamic metrics function
    def metrics_fn(iteration, history):
        if iteration < 3:
            return ["length"]
        return ["accuracy", "length"]

    # Stop when accuracy reaches 1.0
    def stop_if_perfect(context):
        return context["scores"].get("accuracy", 0) == 1.0

    workflow = Workflow(
        generator=gen,
        oracle=oracle,
        max_iterations=5,
        stop_criteria=stop_if_perfect
    )

    result = workflow.run_sync(
        prompt=prompt_fn,
        reference="AlphaFold revolutionized protein structure prediction.",
        gen_args={"max_tokens": 50, "temperature": 0.5},
        metrics=metrics_fn
    )
    print(result)
