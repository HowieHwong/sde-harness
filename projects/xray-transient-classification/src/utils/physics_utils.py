"""
Physics utilities for X-ray transient classification.
"""

from typing import Dict


SPECTRAL_HINT = (
    "Hint: At kT ~ 2 keV, the instrument bandpass (0.3-7 keV) samples primarily "
    "the peak and Rayleigh-Jeans portion of the Planck spectrum. Over this limited "
    "range, such a hot blackbody can appear spectrally hard and resemble a very "
    "flat power law (Γ ~ 0.5). Therefore, the hard power-law fit does not uniquely "
    "imply a non-thermal spectrum."
)

LUMINOSITY_POWERLAW_HINT = (
    "Hint: Check your power law luminosity calculation. Use L = 4πd²F where "
    "d = 50 kpc = 50 × 3.086 × 10^21 cm = 1.543 × 10^23 cm, and F is the peak flux "
    "(not fluence) in erg/s/cm². Use the power law peak flux from the observation data."
)

LUMINOSITY_BLACKBODY_HINT = (
    "Hint: Check your blackbody luminosity calculation. Use L = 4πd²F where "
    "d = 50 kpc = 50 × 3.086 × 10^21 cm = 1.543 × 10^23 cm, and F is the peak flux "
    "(not fluence) in erg/s/cm². Use the blackbody peak flux from the observation data."
)


def generate_feedback(scores: Dict[str, float]) -> str:
    """
    Generate feedback based on round 1 scores.
    
    - If power law luminosity is wrong, give power law hint
    - If blackbody luminosity is wrong, give blackbody hint
    - If classification is wrong, give spectral hint
    """
    hints = []
    
    pl_score = scores.get('luminosity_powerlaw', 1.0)
    bb_score = scores.get('luminosity_blackbody', 1.0)
    class_score = scores.get('top1_correct', 1.0)
    
    if pl_score < 1.0:
        hints.append(LUMINOSITY_POWERLAW_HINT)
    
    if bb_score < 1.0:
        hints.append(LUMINOSITY_BLACKBODY_HINT)
    
    if class_score < 1.0:
        hints.append(SPECTRAL_HINT)
    
    if not hints:
        hints.append("Your analysis looks correct. Verify your reasoning is complete.")
    
    return "\n\n".join(hints)
