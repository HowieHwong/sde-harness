# MatLLMSearch

LLM-based Crystal Structure Generation and Optimization for Materials Discovery, integrated with the SDE-Harness framework.

## Overview

MatLLMSearch leverages large language models to generate novel crystal structures and optimize them for materials properties. This implementation is integrated with the SDE-Harness framework to provide a standardized workflow for scientific discovery.

## Features

- **Crystal Structure Generation (CSG)**: Generate novel crystal structures using evolutionary algorithms guided by LLMs
- **Crystal Structure Prediction (CSP)**: Predict ground state structures for target compounds  
- **Multi-objective optimization**: Optimize for stability (E_hull distance) and mechanical properties (bulk modulus)
- **Multiple LLM support**: Local models, OpenAI GPT, and DeepSeek
- **Structure validation**: Automated validation of generated structures
- **Comprehensive analysis**: Built-in analysis tools for experimental results

## Installation

1. Install the SDE-Harness framework (from parent directory):
```bash
cd ../..
pip install -e .
```

2. Install MatLLMSearch dependencies:
```bash
cd projects/matllmsearch
pip install -r requirements.txt
```

3. Configure models and credentials in the main SDE-Harness config directory:

4. Download required data files:
```bash
# Create data directory
mkdir -p data

# Download seed structures (optional - enables few-shot generation)
# You may download data/band_gap_processed.csv at https://drive.google.com/file/d/1DqE9wo6dqw3aSLEfBx-_QOdqmtqCqYQ5/view?usp=sharing
# Or data/band_gap_processed_5000.csv at https://drive.google.com/file/d/14e5p3EoKzOHqw7hKy8oDsaGPK6gwhnLV/view?usp=sharing

# Download phase diagram data (required for E_hull distance calculations)
wget -O data/2023-02-07-ppd-mp.pkl.gz https://figshare.com/ndownloader/files/48241624
```

**Note**: 
- All configuration is managed through the main SDE-Harness implementation
- MatLLMSearch-specific models are configured in `config/models.yaml`
- API keys are configured in `config/credentials.yaml` 

## Quick Start

### Crystal Structure Generation (CSG)
Generate novel crystal structures using evolutionary optimization:

```bash
python cli.py csg \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --population-size 100 \
    --max-iter 10 \
    --opt-goal e_hull_distance \
    --data-path data/band_gap_processed_5000.csv \
    --save-label csg_experiment
```

### Crystal Structure Prediction (CSP)
Predict ground state structures for a target compound:

```bash
python cli.py csp \
    --compound Ag6O2 \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --population-size 10 \
    --max-iter 5 \
    --save-label ag6o2_prediction
```

### Analysis
The `analyze` command evaluates generated structures and computes comprehensive metrics including:
- Structural validity and composition validity
- Structural diversity and composition diversity
- Structural novelty and composition novelty (vs reference pool)
- Overall novelty (fraction of structures that are both compositionally and structurally novel)
- M3GNet metastability
- Stability rates (CHGNet)

#### Evaluate Existing Results

**Option 1: From a CSV file**
```bash
python cli.py analyze \
    --input data/llama_test.csv \
    --output evaluation_results.json \
    --data-path data/band_gap_processed_5000.csv
```

**Option 2: From a previous CSG run directory**
```bash
python cli.py analyze \
    --results-path logs/analyze_generation \
    --output reevaluated_results.json \
    --data-path data/band_gap_processed_5000.csv
```
This will look for `generations.csv` in the specified results path.

#### Generate and Evaluate via API

Generate structures using the CSG evolutionary workflow with API models and then evaluate:

```bash
python cli.py analyze --generate \
    --model openai/gpt-5-mini \
    --data-path data/band_gap_processed_5000.csv \
    --max-iter 10 \
    --population-size 10 \
    --reproduction-size 5 \
    --parent-size 2 \
    --output gpt5_results.json
```

**Key parameters for API generation:**
- `--generate`: Flag to enable API generation (uses CSG workflow)
- `--model`: Model to use (e.g., `openai/gpt-5-mini`, `openai/gpt-4o-mini`)
- `--data-path`: Path to seed structures CSV (used as reference pool for novelty)
- `--max-iter`: Number of evolutionary iterations
- `--population-size`: Initial population size
- `--reproduction-size`: Number of offspring per generation
- `--parent-size`: Number of parent structures per group

**Note:** All generated structures are kept and deduplicated after all iterations complete before evaluation.

#### Finding Results from Previous Runs

When you run `analyze --generate`, the CSG workflow saves intermediate results to:
- `logs/analyze_generation/generations.csv`: All generated structures with properties
- `logs/analyze_generation/metrics.csv`: Per-iteration metrics

The final evaluation summary is saved to the `--output` file you specify (e.g., `gpt5_results.json`).

To re-evaluate a previous run:
```bash
python cli.py analyze \
    --results-path logs/analyze_generation \
    --output new_evaluation.json \
    --data-path data/band_gap_processed_5000.csv
```

## Configuration Options

### Models
MatLLMSearch now uses SDE-Harness unified model interface with support for:
- **Local models**: Llama, Mistral, and other Hugging Face models
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo

Model configuration is handled via the main SDE-Harness `config/models.yaml` and `config/credentials.yaml` files.

### Optimization Goals
- `e_hull_distance`: Minimize energy above convex hull (stability)
- `bulk_modulus_relaxed`: Maximize bulk modulus (mechanical properties)
- `multi-obj`: Multi-objective optimization combining both

### Structure Formats
- `poscar`: VASP POSCAR format
- `cif`: Crystallographic Information File format

## Architecture

MatLLMSearch is **fully integrated** with SDE-Harness core components:

### **Materials-Specific Components**
- **StructureGenerator**: Uses Generation class for LLM-based structure creation
- **MaterialsOracle**: Evaluates structures using CHGNet/ORB for stability and properties
- **StabilityCalculator**: DFT surrogate models for energy and mechanical property prediction

## File Structure

```
matllmsearch/
├── cli.py                          # Main command-line interface
# Configuration files are in the main SDE-Harness config/ directory:
# ../../config/models.yaml           # Model configurations
# ../../config/credentials.yaml      # API credentials
├── data/                           # Data files directory
│   ├── band_gap_processed_5000.csv     # Seed structures (optional)
│   └── 2023-02-07-ppd-mp.pkl.gz   # Phase diagram data (required)
├── src/
│   ├── modes/
│   │   ├── csg.py                 # Crystal Structure Generation mode
│   │   ├── csp.py                 # Crystal Structure Prediction mode  
│   │   └── analyze.py             # Analysis mode
│   ├── utils/
│   │   ├── structure_generator.py  # LLM-based structure generator
│   │   ├── stability_calculator.py # Structure stability evaluation
│   │   ├── data_loader.py         # Data loading utilities
│   │   └── config.py              # Configuration and prompts
│   └── evaluators/
│       └── materials_oracle.py    # Materials property oracle
├── requirements.txt
└── README.md
```

## Results

Comparison of different methods on crystal structure generation:

| Method | Structural Validity(%) | Comp Validity(%) | Metastability (E_d < 0.1 eV/atom, %) | Metastability (E_d < 0.0 eV/atom, %) | Sun Rate(%) |
|--------|---------------------|---------------|-----------------------------------|-----------------------------------|----------|
| CDVAE | 100 | 86.70 | 28.8 | - | - |
| DiffCSP | 100 | 83.25 | - | 5.06 | 3.34 |
| GPT-5-mini | 100 | 100 | 58.17 | 12.19 | 10.08 |

## Output

Results are saved in the specified log directory with the following structure:

### CSG Workflow Output (when using `--generate` or `csg` command):
- `generations.csv`: Generated structures with properties for each iteration
- `metrics.csv`: Optimization metrics over iterations

### Analysis Output (when using `analyze` command):
- `evaluation_results.json` (or file specified by `--output`): Comprehensive evaluation results including:
  - Validity metrics (structural, composition)
  - Diversity metrics (structural, composition)
  - Novelty metrics (overall, composition, structural, SUN score)
  - Stability metrics (metastability rates, E-hull statistics)
  - Success rates

## Citation

If you use MatLLMSearch in your research, please cite:

```bibtex
@misc{gan2025matllmsearch,
      title={MatLLMSearch: Crystal Structure Discovery with Evolution-Guided Large Language Models}, 
      author={Jingru Gan and Peichen Zhong and Yuanqi Du and Yanqiao Zhu and Chenru Duan and Haorui Wang and Daniel Schwalbe-Koda and Carla P. Gomes and Kristin A. Persson and Wei Wang},
      year={2025},
      eprint={2502.20933},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2502.20933}, 
}
```