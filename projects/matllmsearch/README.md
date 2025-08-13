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
Analyze experimental results:

```bash
python cli.py analyze \
    --results-path logs/my_csg_experiment \
    --experiment-name my_analysis
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

### **Core Integration**
- **`sde_harness.core.generation.Generation`**: Unified LLM interface replacing custom LLMManager
  - Supports local models, OpenAI, Anthropic, DeepSeek
  - Configuration-driven via `config/models.yaml` and `config/credentials.yaml`
  - Built-in async support and resource management

- **`sde_harness.core.oracle.Oracle`**: MaterialsOracle extends base Oracle class
  - Custom materials metrics: `stability`, `bulk_modulus`, `validity`, `multi_objective` 
  - Multi-round metrics: `materials_improvement`, `convergence_rate`
  - Batch evaluation and trend analysis

- **`sde_harness.core.prompt.Prompt`**: Dynamic prompting with custom templates
  - Zero-shot generation prompts
  - Few-shot generation with reference structures
  - Crystal structure prediction prompts
  - Variable substitution and template management

### **Materials-Specific Components**
- **StructureGenerator**: Uses Generation class for LLM-based structure creation
- **MaterialsOracle**: Evaluates structures using CHGNet/ORB for stability and properties
- **StabilityCalculator**: DFT surrogate models for energy and mechanical property prediction

## File Structure

```
matllmsearch/
├── cli.py                          # Main command-line interface
# Configuration files are in the main SDE-Harness config/ directory:
# ../../config/models.yaml           # Model configurations (MatLLMSearch models included)
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
│   │   ├── structure_generator.py  # LLM-based structure generator (uses SDE-Harness Generation)
│   │   ├── stability_calculator.py # Structure stability evaluation
│   │   ├── data_loader.py         # Data loading utilities
│   │   └── config.py              # Configuration and prompts
│   └── evaluators/
│       └── materials_oracle.py    # Materials property oracle
├── requirements.txt
└── README.md
```

## Output

Results are saved in the specified log directory with the following structure:

- `generations.csv`: Generated structures with properties for each iteration
- `metrics.csv`: Optimization metrics over iterations  
- `analysis_report.txt`: Comprehensive analysis summary

## Integration with SDE-Harness

This implementation demonstrates how to integrate domain-specific workflows with the SDE-Harness framework:

1. **ProjectBase inheritance**: Main classes inherit from `sde_harness.base.ProjectBase`
2. **Standard interfaces**: Components implement SDE-Harness Generation and Oracle interfaces
3. **Workflow integration**: Uses SDE-Harness Workflow for optimization loops
4. **Consistent patterns**: Follows SDE-Harness patterns for CLI, configuration, and output

## Citation

If you use MatLLMSearch in your research, please cite:

```bibtex
@misc{gan2025matllmsearch,
      title={Large Language Models Are Innate Crystal Structure Generators}, 
      author={Jingru Gan and Peichen Zhong and Yuanqi Du and Yanqiao Zhu and Chenru Duan and Haorui Wang and Carla P. Gomes and Kristin A. Persson and Daniel Schwalbe-Koda and Wei Wang},
      year={2025},
      eprint={2502.20933},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2502.20933}, 
}
```