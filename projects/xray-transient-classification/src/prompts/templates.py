"""
Prompt templates for X-ray transient classification.
"""

from typing import Any, Dict


CLASSIFY_TEMPLATE = """You are an expert X-ray astronomer. Classify this X-ray transient based on the observational data.

## Observation Data

{observation}

## Your Task

1. Calculate the luminosity in erg/s assuming a distance of 50 kpc (LMC distance), for both spectral models (power law and blackbody)
2. Classify the source based on the observational constraints
3. List the 2 most likely alternative classifications

## Output Format

You MUST respond with valid JSON in exactly this format:

```json
{{
  "luminosity_powerlaw_erg_s": <float>,
  "luminosity_blackbody_erg_s": <float>,
  "classification": "<source type>",
  "confidence": "<high|medium|low>",
  "alternatives": ["<alternative 1>", "<alternative 2>"],
  "reasoning": "<brief explanation>"
}}
```

Valid classification values: thermonuclear_xray_burst, giant_magnetar_flare, stellar_flare, tidal_disruption_event, grb_afterglow, supergiant_fast_xray_transient, ulx_flare, gamma_ray_burst, quasi_periodic_eruption, fast_blue_optical_transient, cataclysmic_variable, supernova_shock_breakout, agn_flare

Respond ONLY with the JSON block, no other text.
"""


ITERATIVE_TEMPLATE = """You are an expert X-ray astronomer. Refine your classification based on feedback.

## Observation Data

{observation}

## Your Previous Response

{previous_reasoning}

## Feedback

{feedback}

## Your Task

Address the feedback and provide an updated classification. Use distance = 50 kpc for luminosity calculation.

## Output Format

You MUST respond with valid JSON in exactly this format:

```json
{{
  "luminosity_powerlaw_erg_s": <float>,
  "luminosity_blackbody_erg_s": <float>,
  "classification": "<source type>",
  "confidence": "<high|medium|low>",
  "alternatives": ["<alternative 1>", "<alternative 2>"],
  "reasoning": "<brief explanation>"
}}
```

Valid classification values: thermonuclear_xray_burst, giant_magnetar_flare, stellar_flare, tidal_disruption_event, grb_afterglow, supergiant_fast_xray_transient, ulx_flare, gamma_ray_burst, quasi_periodic_eruption, fast_blue_optical_transient, cataclysmic_variable, supernova_shock_breakout, agn_flare

Respond ONLY with the JSON block, no other text.
"""


def build_classify_prompt(observation_text: str) -> str:
    """Build the initial classification prompt."""
    return CLASSIFY_TEMPLATE.format(observation=observation_text)


def build_iterative_prompt(
    observation_text: str,
    previous_reasoning: str,
    feedback: str
) -> str:
    """Build an iterative refinement prompt."""
    return ITERATIVE_TEMPLATE.format(
        observation=observation_text,
        previous_reasoning=previous_reasoning,
        feedback=feedback
    )


class PromptBuilder:
    """Builder class for constructing prompts."""
    
    def __init__(self, observation: Dict[str, Any]):
        from ..utils.data_loader import format_observation_for_prompt
        self.observation = observation
        self.observation_text = format_observation_for_prompt(observation)
    
    def classify(self) -> str:
        """Build initial classification prompt."""
        return build_classify_prompt(self.observation_text)
    
    def iterate(self, previous_reasoning: str, feedback: str) -> str:
        """Build iterative refinement prompt."""
        return build_iterative_prompt(
            self.observation_text,
            previous_reasoning,
            feedback
        )
