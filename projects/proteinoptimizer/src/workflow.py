"""SDE-Harness Workflow implementation for Protein GA."""

from __future__ import annotations
from typing import cast

from sde_harness.core import Workflow, Generation, Prompt
from .oracles import (
    Syn3bfoOracle, GB1Oracle, TrpBOracle, AAVOracle, GFPOracle,
    fitness_oracles
)
from .core import ProteinOptimizer


class ProteinWorkflow(Workflow):
    """Workflow that runs the GA inside its generation step."""

    def __init__(self, oracle_name: str = "syn-3bfo", model_name: str | None = None, **ga_kwargs):
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
        # Select oracle
        if oracle_name == 'syn-3bfo':
            oracle = Syn3bfoOracle()
        elif oracle_name == 'gb1':
            oracle = GB1Oracle()
        elif oracle_name == 'trpb':
            oracle = TrpBOracle()
        elif oracle_name == 'aav':
            oracle = AAVOracle()
        elif oracle_name == 'gfp':
            oracle = GFPOracle()
        else:
            raise ValueError(f"Unknown oracle: {oracle_name}")

        super().__init__(generator=gen, oracle=oracle)
        self.ga_kwargs = ga_kwargs  # population size, etc.

    async def _execute_generation(self, built_prompt: str, gen_args: dict, iteration: int) -> tuple[dict, dict]:
        """Override generation to run the GA."""
        # We don't use the prompt directly but could parse parents from it
        
        # For ML-based oracles, we need a way to get an initial population.
        # This is a placeholder and should be improved.
        if hasattr(self.oracle, 'get_initial_population'):
            fitness_oracle = cast(fitness_oracles.FitnessOracle, self.oracle)
            initial_pop = fitness_oracle.get_initial_population(
                self.ga_kwargs.get("initial_size", 50)
            )
        else:
            # Fallback for oracles without a dataset to sample from (AAV, GFP)
            # We will start with a known reference sequence and mutate it.
            if self.oracle.name == 'aav':
                wt_sequence = "DEEEIRTTNPVATEQYGSVSTNLQRGNR"
            elif self.oracle.name == 'gfp':
                wt_sequence = "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
            else:
                raise ValueError(f"Cannot generate initial population for oracle '{self.oracle.name}'")
            initial_pop = [wt_sequence] * self.ga_kwargs.get("initial_size", 50)

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