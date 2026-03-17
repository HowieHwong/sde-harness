"""Prompt templates for X-ray spectral fitting using SDE-Harness Prompt class."""

from ..compat import Prompt


class SpectralPrompts:
    """Collection of prompts for X-ray spectral model discovery."""

    INITIAL_TEMPLATE = """You are an expert X-ray astronomer. You are given an observed X-ray spectrum and must propose spectral models to fit it.

## Observation Summary

{observation_summary}

## Your Task

Propose {num_hypotheses} different spectral models to fit this data.

The models will be fitted using Sherpa with the XSPEC model library. For each model, you may optionally provide initial parameter values and bounds, or leave them at defaults.

NOTE: Sherpa requires model expressions with instance names, e.g. "xstbabs.abs1 * xsapec.apec1" — not "xswabs*apec".

## Output Format

Respond with a JSON array of {num_hypotheses} models:

```json
[
  {{
    "model": "<Sherpa XSPEC model expression string>",
    "params": {{
      "<instance.param>": {{"val": <initial>, "min": <lower>, "max": <upper>}},
      ...
    }},
    "reasoning": "<brief physical motivation>"
  }},
  ...
]
```

IMPORTANT: This is the FIRST round. Propose DIVERSE single-component models to establish baselines.

Each model should be DIFFERENT - cover a range of emission mechanisms (thermal, non-thermal, disk, etc.) based on your physical interpretation of the source.

DO NOT propose multi-component models (no "+") in this first round.
DO NOT propose the same model type multiple times - we need variety first, refinement later.

Respond ONLY with the JSON array."""

    ITERATIVE_TEMPLATE = """You are an expert X-ray astronomer refining spectral fits.

## Observation Summary

{observation_summary}

## All Previous Results

{all_results_summary}
{errors_section}
## Current Best Models (ranked by fit quality)

{population_summary}

## Your Task

Based on all previous fit results, propose {num_hypotheses} NEW and DIFFERENT models.

NOTE: Sherpa requires model expressions with instance names, e.g. "xstbabs.abs1 * xsapec.apec1".

IMPORTANT - Propose DIVERSE models, not variations of the same thing:
1. If you haven't tried many single-component models yet, try MORE DIFFERENT ones (different emission mechanisms).
2. Only after establishing that simple models don't fit well, try adding ONE component.
3. Only refine parameters (same model, different bounds) if you've already explored diverse model types.

Look at what's been tried:
- What model TYPES haven't been tested yet? Try those first.
- Are there single-component models that haven't been tried? Try them before multi-component.
- Only if the best simple model has reduced cstat >> 1, consider adding complexity.

When refining parameters:
- If a parameter hit its boundary (see WARNINGS), widen the bounds.
- Use fitted values from good fits as starting points for new models.

BIC guidance:
- Lower BIC = better (accounts for both fit quality and complexity).
- A complex model must have SIGNIFICANTLY lower BIC to be preferred.

## Output Format

Respond with a JSON array of {num_hypotheses} models:

```json
[
  {{
    "model": "<Sherpa XSPEC model expression string>",
    "params": {{
      "<instance.param>": {{"val": <initial>, "min": <lower>, "max": <upper>}},
      ...
    }},
    "reasoning": "<physical reasoning based on previous results and source properties>"
  }},
  ...
]
```

Respond ONLY with the JSON array."""

    CLASSIFICATION_TEMPLATE = """You are an expert X-ray astronomer. Based on the spectral fitting results and observational data, classify this X-ray transient.

## Observation Summary

{observation_summary}

## Best Spectral Fit So Far

Model: {best_model}
Reduced C-stat: {reduced_cstat}
Fitted parameters: {fitted_params}

## Classification Task

Your classification MUST be physically consistent with the best-fit model and its parameters above. Choose the SINGLE best-matching source class from the list below:

1.  Stellar coronal flare (late-type star)
2.  Tidal disruption event (TDE)
3.  Low-mass X-ray binary (LMXB) outburst (accretion-powered, non-burst)
4.  Gamma-ray burst (GRB) X-ray afterglow
5.  Cataclysmic variable (CV) / classical nova outburst
6.  Active galactic nucleus (AGN) / Seyfert flare
7.  Be/X-ray binary (BeXRB) outburst
8.  Supernova shock breakout
9.  RS CVn / Algol-type binary flare
10. Thermonuclear X-ray burst
11. Blazar / BL Lac flare
12. Magnetar flare / Soft Gamma Repeater (SGR) outburst
13. Dwarf nova / AM CVn outburst
14. Magnetar giant flare
15. High-mass X-ray binary (HMXB) outburst
16. Ultraluminous X-ray source (ULX) variability
17. Supergiant fast X-ray transient (SFXT)
18. Neutron star merger (kilonova) X-ray counterpart
19. Wolf-Rayet or O-star wind variability
20. Symbiotic X-ray binary
21. Pulsar wind nebula variability
22. Unknown / unclassified transient

Respond with ONLY a JSON object:
```json
{{
  "classification": "<exact class name from the list above>",
  "reasoning": "<2-3 sentences explaining how the best-fit model and its parameters led to this classification>"
}}
```"""

    SUMMARY_TEMPLATE = """You are an expert X-ray astronomer. Based on extensive model fitting, summarize why the best model was selected.

## Observation Summary

{observation_summary}

## All Models Tried

{all_results_summary}

## Best Model Selected

Model: {best_model}
Reduced C-stat: {reduced_cstat}
BIC: {bic}
Fitted parameters: {fitted_params}

## Your Task

Provide a brief scientific summary (2-4 sentences) explaining:
1. Why this model is the best choice for this source
2. What physical emission mechanism it represents
3. Whether the fitted parameters are physically reasonable for this type of source

Respond with ONLY the summary text, no JSON or formatting."""

    @staticmethod
    def get_initial_prompt(observation_summary: str, num_hypotheses: int = 4) -> Prompt:
        """Create prompt for initial model proposals."""
        return Prompt(
            custom_template=SpectralPrompts.INITIAL_TEMPLATE,
            default_vars={
                "observation_summary": observation_summary,
                "num_hypotheses": num_hypotheses,
            }
        )

    @staticmethod
    def get_iterative_prompt(
        observation_summary: str,
        population_summary: str,
        all_results_summary: str,
        num_hypotheses: int = 4,
        errors_summary: str = "",
    ) -> Prompt:
        """Create prompt for iterative refinement."""
        if errors_summary:
            errors_section = (
                "\n## Errors from Last Round (avoid these mistakes)\n\n"
                + errors_summary
                + "\n\n"
            )
        else:
            errors_section = ""
        return Prompt(
            custom_template=SpectralPrompts.ITERATIVE_TEMPLATE,
            default_vars={
                "observation_summary": observation_summary,
                "population_summary": population_summary,
                "all_results_summary": all_results_summary,
                "num_hypotheses": num_hypotheses,
                "errors_section": errors_section,
            }
        )

    @staticmethod
    def get_classification_prompt(
        observation_summary: str,
        best_model: str,
        reduced_cstat: float,
        fitted_params: str,
    ) -> "Prompt":
        """Create prompt for source classification."""
        return Prompt(
            custom_template=SpectralPrompts.CLASSIFICATION_TEMPLATE,
            default_vars={
                "observation_summary": observation_summary,
                "best_model": best_model,
                "reduced_cstat": reduced_cstat,
                "fitted_params": fitted_params,
            }
        )

    @staticmethod
    def get_summary_prompt(
        observation_summary: str,
        all_results_summary: str,
        best_model: str,
        reduced_cstat: float,
        bic: float,
        fitted_params: str,
    ) -> Prompt:
        """Create prompt for final summary reasoning."""
        return Prompt(
            custom_template=SpectralPrompts.SUMMARY_TEMPLATE,
            default_vars={
                "observation_summary": observation_summary,
                "all_results_summary": all_results_summary,
                "best_model": best_model,
                "reduced_cstat": reduced_cstat,
                "bic": bic,
                "fitted_params": fitted_params,
            }
        )
