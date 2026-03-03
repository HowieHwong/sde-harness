# X-Ray Transient Classification

A benchmark project for evaluating LLM physics reasoning through X-ray transient source classification.

## Overview

This project tests an LLM's ability to:
- Perform multi-step physical calculations (flux → luminosity → Eddington comparison)
- Recognize spectral degeneracies (hot blackbody appearing as hard power law)
- Apply conditional reasoning (constraints that depend on distance assumptions)
- Synthesize multi-wavelength constraints
- Demonstrate epistemic calibration (appropriate confidence levels)

## Installation

```bash
# From the sde-harness root directory
pip install -e .

# Install project-specific dependencies
pip install -r projects/xray-transient-classification/requirements.txt
```

## Usage

### List Available Transients

```bash
python projects/xray-transient-classification/cli.py list
```

### Classify a Transient

```bash
python projects/xray-transient-classification/cli.py classify \
    --input projects/xray-transient-classification/data/transients/lmc_burst_001.json \
    --model gpt-4 \
    --iterations 3 \
    --verbose
```

### Validate Data Files

```bash
python projects/xray-transient-classification/cli.py validate
```

## Data Format

### Transient Observations

Each transient is stored as a JSON file with the following structure:

```json
{
  "transient_id": "unique_identifier",
  "metadata": { "description": "...", "discovery_instrument": "..." },
  "detection": { "instrument": "...", "band_keV": [0.3, 7.0], ... },
  "persistent_constraints": { "pre_flare_limit_erg_s_cm2": 1e-14, ... },
  "temporal": { "spike": {...}, "tail": {...}, ... },
  "spectral": { "evolution": "...", "fits": {...}, ... },
  "energetics": { "peak_flux_erg_s_cm2": [4e-10, 6e-10], ... },
  "counterparts": { "gamma_ray": {...}, "optical": {...}, ... },
  "context": { "field": "LMC", "possible_associations": {...}, ... },
  "ground_truth": { "classification": "...", "alternatives": [...], ... }
}
```

**Key Design Principle**: Raw observables (flux) are provided, NOT derived quantities (luminosity). The LLM must perform the physics calculations.

### Source Types

The `source_types.json` file defines the taxonomy of X-ray transient types with their typical properties.

## Evaluation Metrics

### Classification Accuracy
- `top1_correct`: Primary classification matches ground truth
- `alternatives_identified`: Plausible alternatives mentioned

### Physics Reasoning Quality
- `luminosity_calculation`: Correctly derived luminosity from flux and distance
- `eddington_reasoning`: Compared to Eddington limit, connected to NS
- `bandpass_awareness`: Recognized spectral degeneracy from limited bandpass
- `distance_exploration`: Considered multiple distance scenarios
- `conditional_reasoning`: Properly scoped conditional constraints
- `counterpart_reasoning`: Used multi-wavelength constraints

### Multi-Round Metrics
- `reasoning_improvement`: Quality improvement across iterations

## Iterative Workflow

The classification uses an iterative refinement process:

1. **Initial Classification**: LLM receives observation data and classification prompt
2. **Evaluation**: Oracle scores the response on all metrics
3. **Feedback Generation**: Physics-based feedback targets specific gaps
4. **Refinement**: LLM receives feedback and refines analysis
5. **Repeat**: Steps 2-4 repeat for configured iterations

## Project Structure

```
xray-transient-classification/
├── cli.py                    # Command-line interface
├── requirements.txt          # Project dependencies
├── README.md                 # This file
├── data/
│   ├── transients/           # Observation JSON files
│   │   └── lmc_burst_001.json
│   └── source_types.json     # Source type taxonomy
├── results/                  # Output directory
└── src/
    ├── __init__.py
    ├── modes/
    │   ├── __init__.py
    │   └── classify.py       # Classification workflow
    ├── oracles/
    │   ├── __init__.py
    │   └── classification_oracle.py  # Evaluation metrics
    ├── prompts/
    │   ├── __init__.py
    │   └── templates.py      # Prompt templates
    └── utils/
        ├── __init__.py
        ├── data_loader.py    # Data loading utilities
        └── physics_utils.py  # Physics calculations
```

## Physics Background

### Key Concepts

1. **Flux to Luminosity**: L = 4πd²F
   - Requires distance assumption
   - Different distances yield vastly different luminosities

2. **Eddington Limit**: L_Edd ≈ 1.3×10³⁸ erg/s (for 1.4 M☉ NS)
   - Maximum luminosity for spherical accretion
   - Near-Eddington suggests NS surface emission

3. **Spectral Degeneracy**: At kT ~ 2 keV, instrument bandpass (0.3-7 keV) samples primarily the Rayleigh-Jeans portion of the blackbody spectrum, making it appear as a hard power law (Γ ~ 0.5).

4. **Conditional Constraints**: Some information (e.g., stellar population age) only applies under certain distance assumptions.

## Contributing

To add new transients:
1. Create a JSON file in `data/transients/`
2. Follow the schema of existing observations
3. Include ground truth for evaluation
4. Run `cli.py validate` to check file structure
