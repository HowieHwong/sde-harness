"""SDE-Harness Workflow implementation for Protein GA."""

from __future__ import annotations

from sde_harness.core import Workflow, Generation, Prompt
from .oracles import Syn3bfoOracle
from .core import ProteinOptimizer


class ProteinWorkflow(Workflow):
    """Workflow that runs the GA inside its generation step."""

    def __init__(self, model_name: str | None = None, **ga_kwargs):
        import os
        # Get the path to the sde-harness root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sde_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        
        models_file = os.path.join(sde_root, "models.yaml")
        credentials_file = os.path.join(sde_root, "credentials.yaml")
        gen = Generation(
            models_file=models_file,
            credentials_file=credentials_file,
            model_name=model_name or "openai/gpt-4o-2024-08-06",
        )
        oracle = Syn3bfoOracle()
        super().__init__(generator=gen, oracle=oracle)
        self.ga_kwargs = ga_kwargs  # population size, etc.

    async def _execute_generation(self, built_prompt: str, gen_args: dict, iteration: int):
        """Override generation to run the GA."""
        # We don't use the prompt directly but could parse parents from it
        initial_pop = self.oracle.get_initial_population(
            self.ga_kwargs.get("initial_size", 50)
        )

        optimizer = ProteinOptimizer(
            oracle=self.oracle,
            population_size=self.ga_kwargs.get("population_size", 100),
            offspring_size=self.ga_kwargs.get("offspring_size", 200),
            mutation_rate=self.ga_kwargs.get("mutation_rate", 0.02),
            model_name=gen_args.get("model"),
            use_llm_mutations=bool(gen_args.get("model")),
        )

        results = optimizer.optimize(
            initial_pop,
            num_generations=self.ga_kwargs.get("generations", 5)
        )

        # Return best sequence found as the "text" output of this step
        return (
            {"text": results["best_sequence"]},
            {"ga_oracle_calls": results["oracle_calls"]},
        ) 