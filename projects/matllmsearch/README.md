# MatLLMSearch

LLM-based Crystal Structure Generation and Optimization for Materials Discovery, integrated with the SDE-Harness framework.

## Overview

MatLLMSearch leverages large language models to generate novel crystal structures and optimize them for materials properties. This implementation is integrated with the SDE-Harness framework to provide a standardized workflow for scientific discovery.

## Features

- **Crystal Structure Generation (CSG)**: Generate novel crystal structures using evolutionary algorithms guided by LLMs
- **Crystal Structure Prediction (CSP)**: Predict ground state structures for target compounds  
- **Multi-objective optimization**: Optimize for stability (E_hull distance) and mechanical properties (bulk modulus)
- **Multiple LLM support**: Local models via vLLM, OpenAI GPT, and DeepSeek
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

3. Download required data files:
```bash
# Create data directory
mkdir -p data

# Download seed structures (optional - enables few-shot generation)
# Manual download from: https://drive.google.com/file/d/1DqE9wo6dqw3aSLEfBx-_QOdqmtqCqYQ5/view?usp=sharing
# Save as: data/band_gap_processed.csv

# Download phase diagram data (required for E_hull distance calculations)
wget -O data/2023-02-07-ppd-mp.pkl.gz https://figshare.com/ndownloader/files/48241624
```

4. (Optional) Set up API keys for external LLM providers:
```bash
export OPENAI_API_KEY="your_openai_key"
export OPENROUTER_API_KEY="your_openrouter_key"
```

## Quick Start

### Crystal Structure Generation (CSG)
Generate novel crystal structures using evolutionary optimization:

```bash
python cli.py csg \\
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \\
    --population-size 100 \\
    --max-iter 10 \\
    --opt-goal e_hull_distance \\
    --save-label my_csg_experiment
```

### Crystal Structure Prediction (CSP)
Predict ground state structures for a target compound:

```bash
python cli.py csp \\
    --compound Ag6O2 \\
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --population-size 50 \\
    --max-iter 5 \\
    --save-label ag6o2_prediction
```

### Analysis
Analyze experimental results:

```bash
python cli.py analyze \\
    --results-path logs/my_csg_experiment \\
    --experiment-name my_analysis
```

## Configuration Options

### Models
- **Local models**: Any Hugging Face compatible model via vLLM
- **OpenAI**: GPT-3.5, GPT-4 series

### Optimization Goals
- `e_hull_distance`: Minimize energy above convex hull (stability)
- `bulk_modulus_relaxed`: Maximize bulk modulus (mechanical properties)
- `multi-obj`: Multi-objective optimization combining both

### Structure Formats
- `poscar`: VASP POSCAR format
- `cif`: Crystallographic Information File format

## Architecture

MatLLMSearch is built on the SDE-Harness framework with the following components:

- **StructureGenerator**: LLM-based structure generation following SDE-Harness Generation interface
- **MaterialsOracle**: Structure evaluation using DFT surrogates (CHGNet, ORB) following SDE-Harness Oracle interface  
- **Workflow**: Evolutionary optimization workflow using SDE-Harness Workflow class
- **ProjectBase**: Main project class inheriting from SDE-Harness ProjectBase

## File Structure

```
matllmsearch/
├── cli.py                          # Main command-line interface
├── data/                           # Data files directory
│   ├── band_gap_processed.csv     # Seed structures (optional)
│   └── 2023-02-07-ppd-mp.pkl.gz   # Phase diagram data (required)
├── src/
│   ├── modes/
│   │   ├── csg.py                 # Crystal Structure Generation mode
│   │   ├── csp.py                 # Crystal Structure Prediction mode  
│   │   └── analyze.py             # Analysis mode
│   ├── utils/
│   │   ├── structure_generator.py  # LLM-based structure generator
│   │   ├── llm_manager.py         # LLM interface management
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