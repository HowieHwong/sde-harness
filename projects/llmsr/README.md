# LLMSR: LLM-based Symbolic Regression for Equation Discovery

A scientific discovery framework for automated equation discovery using Large Language Models (LLMs) and iterative optimization.

## Overview

LLMSR is a project built on the SDE-Harness framework that uses LLMs to discover mathematical equations from data. It implements an iterative optimization process where:

1. **LLM Generation**: The LLM generates Python functions representing mathematical relationships
2. **Code Execution**: Generated code is parsed and executed with input data
3. **Parameter Optimization**: Parameters in the equations are optimized using numerical methods
4. **Evaluation**: Equations are evaluated using Normalized Mean Squared Error (NMSE)
5. **Iterative Improvement**: History of previous attempts is used to guide future generations

## Features

- **Multi-Dataset Support**: Works with various scientific datasets from HuggingFace
- **Iterative Optimization**: Uses history to improve equation discovery over multiple iterations
- **Parameter Optimization**: Automatically optimizes equation parameters using scipy
- **Multiple Metrics**: Evaluates equations using NMSE, RMSE, MAE, and convergence metrics
- **Flexible LLM Integration**: Supports various LLM providers through SDE-Harness
- **Comprehensive Logging**: Tracks all iterations and results for analysis

## Installation

### Option 1: Install as a package
```bash
# From the projects/llmsr directory
pip install -e .
```

### Option 2: Install dependencies only
```bash
pip install -r requirements.txt
```

### Setup API Credentials
Set up your API credentials in the sde-harness root directory:
- `models.yaml`: Configure your LLM models
- `credentials.yaml`: Add your API keys

### Test Installation
Run the import test to verify everything is working:
```bash
python test_imports.py
```

## Quick Start

### Using the CLI

```bash
# Discover equations for all problems in a dataset
python -m projects.llmsr.src.cli --dataset lsrtransform --output-dir results

# Discover equation for a specific problem
python -m projects.llmsr.src.cli --dataset lsrtransform --problem "problem_name" --output-dir results

# Use a different model and more iterations
python -m projects.llmsr.src.cli --dataset lsrtransform --model "openai/gpt-4" --max-iterations 10
```

### Using the Python API

```python
import asyncio
import weave
from projects.llmsr.src import LLMSRWorkflow

# Initialize weave
weave.init("my_equation_discovery")

async def main():
    # Create workflow
    workflow = LLMSRWorkflow(
        model_name="openai/gpt-4o-2024-08-06",
        max_iterations=5
    )
    
    # Setup dataset
    workflow.setup_dataset("lsrtransform")
    
    # Discover equation for a specific problem
    result = await workflow.discover_equation("problem_name", "outputs")
    
    # Print summary
    workflow.print_summary()

# Run the workflow
asyncio.run(main())
```

## Supported Datasets

- **lsrtransform**: Transformed Feynman equations
- **bio_pop_growth**: Biological population growth models
- **chem_react**: Chemical reaction kinetics
- **matsci**: Materials science equations
- **phys_osc**: Physical oscillator systems

## Architecture

### Core Components

1. **LLMSRGeneration**: Extends SDE-Harness Generation for equation-specific parsing
2. **EquationOracle**: Evaluates equations using NMSE and parameter optimization
3. **LLMSRDatasetLoader**: Loads datasets from HuggingFace
4. **EquationPromptTemplates**: Creates prompts for equation discovery
5. **LLMSRWorkflow**: Orchestrates the entire discovery process

### Workflow

1. **Data Loading**: Load problem data from HuggingFace datasets
2. **Prompt Generation**: Create prompts with variable descriptions and history
3. **LLM Generation**: Generate Python equation functions
4. **Code Parsing**: Extract and validate generated code
5. **Parameter Optimization**: Optimize equation parameters using scipy
6. **Evaluation**: Calculate NMSE and other metrics
7. **History Update**: Store results for next iteration
8. **Iteration**: Repeat until convergence or max iterations

## Configuration

### Model Configuration

Configure your LLM models in `models.yaml`:

```yaml
models:
  openai/gpt-4o-2024-08-06:
    provider: openai
    max_tokens: 1000
    temperature: 0.1
```

### Credentials Configuration

Add your API keys in `credentials.yaml`:

```yaml
credentials:
  openai:
    api_key: your_openai_api_key
```

## Output Format

Results are saved in JSON format with the following structure:

```json
{
  "outputs": ["generated_equation_code_1", "generated_equation_code_2", ...],
  "scores": [
    {"nmse": 0.123, "rmse": 0.456, "mae": 0.789},
    {"nmse": 0.098, "rmse": 0.345, "mae": 0.567},
    ...
  ],
  "prompts": ["prompt_1", "prompt_2", ...],
  "iterations": [1, 2, 3, ...],
  "metadata": {...}
}
```

## Examples

### Example 1: Simple Linear Relationship

Input data with variables `x` and `y`, where `y = 2*x + 1`

The LLM might generate:
```python
def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Mathematical function for y"""
    y = params[0] * x + params[1]
    return y
```

### Example 2: Complex Relationship

Input data with variables `t`, `v`, `x` for a harmonic oscillator:
```python
def equation(x: np.ndarray, t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Mathematical function for velocity"""
    v = params[0] * np.cos(params[1] * t + params[2]) * x
    return v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{llmsr2024,
  title={LLMSR: LLM-based Symbolic Regression for Equation Discovery},
  author={LLMSR Team},
  year={2024},
  url={https://github.com/your-repo/llmsr}
}
```
