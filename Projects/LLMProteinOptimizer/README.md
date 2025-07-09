# Large Language Model is Secretly a Protein Sequence Optimizer


A framework for optimizing protein sequences using large language models (LLMs) and evolutionary algorithms. This project combines the power of state-of-the-art language models with traditional optimization techniques to improve protein properties.

## Overview

This project implements a protein optimization pipeline that can:
- Use large language model Llama-3 for protein sequence generation
- Support both single-objective and multi-objective optimization
- Compare performance against baseline evolutionary algorithms
- Visualize optimization results through various plotting methods

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yinkaiw/LLMProteinOptimizer.git
cd LLMProteinOptimizer
```

2. Install the required dependencies:

```bash
conda create -n lpo python=3.9
conda activate lpo
pip install -r requirements.txt
```

3. Install the GGS_utils package for ML oracle prediction:

The GGS_utils package includes the checkpoint, model, and sequence encoding function from https://github.com/kirjner/GGS.git.
We utilize it as the pretrained ML oracle model to predict the fitness of protein sequences.


## Supported Datasets

The framework currently supports the following protein datasets:

- **GB1**: B1 domain of streptococcal protein G (4 key positions: 39, 40, 41, and 54)
- **TrpB**: Beta-subunit of tryptophan synthase (4 key positions: 183, 184, 227, and 228)
- **AAV**: AAV2 Capsid protein VP1 (positions 561-588)
- **GFP**: Green Fluorescent Protein
- **Syn-3bfo**: Ig-like C2-type 2 domain protein

## Usage

### Single-Objective Optimization

For single-objective optimization, use the following command:

```bash
cd single_objective
sh run_{dataset}.sh
```

### Multi-Objective Optimization

For multi-objective optimization with Pareto front:

```bash
cd multi_objective
sh run_{dataset}.sh
```
This script will run the multi-objective optimization (pareto frontier and sum of objectives) for the specified dataset and model. Please feel free to modify the script to run the optimization with other objectives.

## Model Options

- `Llama3`: Meta's Llama 3 model (default)
- `EA`: Baseline evolutionary algorithm

## Visualization

The framework provides several visualization options:

1. Box plots showing fitness distribution across iterations
2. Pareto front visualization for multi-objective optimization
3. Trajectory visualization for optimization progress

Results are automatically saved in the following formats:
- JSON files containing detailed optimization results
- PNG files with visualization plots

## Citation

If you find this work useful, please cite our paper:

```
@article{wang2025large,
  title={Large Language Model is Secretly a Protein Sequence Optimizer},
  author={Wang, Yinkai and He, Jiaxing and Du, Yuanqi and Chen, Xiaohui and Li, Jianan Canal and Liu, Li-Ping and Xu, Xiaolin and Hassoun, Soha},
  journal={arXiv preprint arXiv:2501.09274},
  year={2025}
}
```
