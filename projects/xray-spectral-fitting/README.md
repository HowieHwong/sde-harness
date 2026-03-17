# X-Ray Spectral Fitting Benchmark

LLM-augmented evolutionary algorithm for X-ray spectral model discovery.

## Overview

This benchmark tests an LLM's ability to discover the best-fitting spectral model for X-ray observations using an evolutionary search strategy. The LLM proposes model hypotheses, a Sherpa oracle evaluates them via spectral fitting, and the best models survive to inform the next generation.

### Key Features

- **Evolutionary Search (4→2)**: Each generation, the LLM proposes 4 model hypotheses. The top 2 (by BIC) are kept for the next round.
- **Sherpa Oracle**: Uses the Sherpa X-ray fitting package to evaluate model quality via C-statistic and BIC.
- **XSPEC Models**: Supports standard XSPEC models (tbabs, powerlaw, bbody, bremss, apec, diskbb, comptt).

## Requirements

- Python 3.8-3.10 with Sherpa and XSPEC support
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Installation

### Option 1: Setup script (Recommended)

The setup script automatically handles platform detection (including Apple Silicon):

```bash
# Run the setup script
./setup_env.sh

# Activate the environment
conda activate xray-spectral-fitting

# Set your API key
export OPENAI_API_KEY="your-key-here"
```

### Option 2: Using the environment file

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate xray-spectral-fitting

# Set your API key
export OPENAI_API_KEY="your-key-here"
```

### Option 3: Manual installation

```bash
# Create a new conda environment with Sherpa
conda create -n xray-spectral-fitting python=3.10
conda activate xray-spectral-fitting
conda install -c conda-forge -c ciao sherpa

# Install OpenAI client
pip install openai

# Set your API key
export OPENAI_API_KEY="your-key-here"
```

### Platform Notes

**Linux/Intel Mac**: Should work out of the box with any installation option.

**Apple Silicon (M1/M2/M3)**: Sherpa with XSPEC requires x86_64 architecture. The setup script handles this automatically, but for manual installation:

```bash
# Create x86_64 environment via Rosetta
CONDA_SUBDIR=osx-64 conda create -n xray-spectral-fitting python=3.10
conda activate xray-spectral-fitting
conda config --env --set subdir osx-64
conda install -c conda-forge -c ciao sherpa
pip install openai
```

### Verify Installation

```bash
# Test that Sherpa and XSPEC are working
python -c "from sherpa.astro import ui; ui.set_xsxsect('vern'); print('Sherpa+XSPEC OK')"
```

## Usage

### Run Spectral Fitting

```bash
# Basic run with GPT-4o
python cli.py fit --pha data/spectra/lmc_flare/flaresp_grp1.pha -v

# With custom settings (e.g. more hypotheses per round)
python cli.py fit --pha data/spectra/lmc_flare/flaresp_grp1.pha \
    --model gpt-4o \
    --population-size 2 \
    --offspring-size 4 \
    --generations 15 \
    -v

# Save results to file
python cli.py fit --pha data/spectra/lmc_flare/flaresp_grp1.pha -o results.json -v
```

### List Available Spectra

```bash
python cli.py list
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pha` | (required) | Path to PHA spectrum file |
| `--model` | openai/gpt-4o-2024-08-06 | LLM model to use (key from config/models.yaml) |
| `--population-size` | 2 | Number of best models to keep each generation |
| `--offspring-size` | 4 | Number of hypotheses to generate per generation |
| `--generations` | 10 | Maximum number of generations |
| `--emin` | 0.3 | Minimum energy in keV |
| `--emax` | 7.0 | Maximum energy in keV |
| `--seed` | 0 | Random seed(s) for reproducibility |
| `-o, --output` | - | Output JSON file for results |
| `-v, --verbose` | False | Verbose output |

## How It Works

1. **Initial Generation**: LLM receives observation summary (counts, hardness ratio, exposure) and proposes N model hypotheses
2. **Oracle Evaluation**: Each model is fitted using Sherpa, returning C-statistic and best-fit parameters
3. **Selection**: Top M models (by reduced C-stat) survive to the next generation
4. **Iteration**: LLM sees the surviving models' fit results and proposes new hypotheses
5. **Termination**: Process runs for `--generations` rounds (no early stopping)

## Evaluation Metrics

- `generations`: Number of generations run
- `oracle_calls`: Total Sherpa fit evaluations
- `found_expected_model`: Whether the expected model type was found
- `best_reduced_cstat`: Final fit quality (closer to 1.0 is better)

## Project Structure

```
xray-spectral-fitting/
├── cli.py                          # Main CLI
├── README.md                       # This file
└── data/
    └── spectra/
        └── lmc_flare/              # LMC transient spectrum
            ├── flaresp_grp1.pha    # Grouped spectrum
            ├── flaresp.pha         # Raw spectrum
            ├── flaresp.rmf         # Response matrix
            ├── flaresp.corr.arf    # Auxiliary response
            └── flaresp_bkg.pi      # Background spectrum
```

## Example Output

```
============================================================
X-RAY SPECTRAL FITTING BENCHMARK
Config: 4 → 2
Model: gpt-4o
============================================================

--- Generation 1 ---
Generated 4 hypotheses
  xstbabs.abs1 * xspowerlaw.pow1: C-stat=132.84/134=0.991
  xstbabs.abs1 * xsbbody.bb1: C-stat=128.4/134=0.958
  xstbabs.abs1 * xsbremss.brems1: C-stat=136.27/134=1.017
  xstbabs.abs1 * xscomptt.comp1: C-stat=157.74/132=1.195

Top 2 models:
  1. xstbabs.abs1 * xsbbody.bb1: 0.958
  2. xstbabs.abs1 * xspowerlaw.pow1: 0.991

...

============================================================
OPTIMIZATION COMPLETE
============================================================
Best model: xstbabs.abs1 * xsbbody.bb1
Best C-stat: 0.9582
Total generations: 10
Total oracle calls: 40
```

## Scientific Background

### C-Statistic

The C-statistic (Cash statistic) is used for Poisson-distributed data. A reduced C-stat (C-stat/dof) near 1.0 indicates a good fit.

### Common X-Ray Models

| Model | Physical Interpretation |
|-------|------------------------|
| `xspowerlaw` | Non-thermal emission (AGN, pulsars) |
| `xsbbody` | Thermal emission from compact surface (NS, stellar flare) |
| `xsbremss` | Thermal bremsstrahlung (hot plasma) |
| `xsapec` | Collisionally-ionized plasma (stellar coronae) |
| `xsdiskbb` | Accretion disk (black holes, NS) |
| `xstbabs` | Interstellar absorption |

### The LMC Flare Dataset

The included spectrum is from a short-duration X-ray transient in the Large Magellanic Cloud. The expected best-fit model is a thermal blackbody (`xsbbody`), characteristic of a stellar flare or thermonuclear burst.
