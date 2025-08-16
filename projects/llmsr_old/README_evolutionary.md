# LLMSR Evolutionary Framework

This document describes the evolutionary framework implementation for LLMSR (Large Language Model Symbolic Regression) that uses selective example inclusion based on NMSE improvement.

## Overview

The evolutionary framework implements a multi-island evolutionary algorithm that:

1. **Selective Example Inclusion**: Only includes generated samples as in-context examples if they provide better NMSE scores than the previous best
2. **Multi-Island Buffer**: Maintains multiple islands of equation programs with different characteristics
3. **Boltzmann Sampling**: Samples examples from the buffer using Boltzmann distribution to favor better-scored programs
4. **Evolutionary Selection**: Resets weaker islands and promotes successful programs

## Key Components

### 1. Evolutionary Buffer (`core/evolutionary_buffer.py`)

The `EvolutionaryBuffer` class implements the multi-island storage system:

```python
from core.evolutionary_buffer import EvolutionaryBuffer

buffer = EvolutionaryBuffer(
    num_islands=4,                    # Number of islands
    max_programs_per_island=50,       # Max programs per island
    reset_period=100,                 # How often to reset weaker islands
    reset_fraction=0.5                # Fraction of islands to reset
)
```

**Features:**
- **Multi-island architecture**: Maintains diversity through separate populations
- **Boltzmann sampling**: Samples programs with probability proportional to exp(-score/temperature)
- **Automatic island reset**: Resets weaker islands and repopulates with successful programs
- **Score tracking**: Tracks global best NMSE and improvement statistics

### 2. Evolutionary Prompt Templates (`modes/evolutionary_prompt_templates.py`)

The `EvolutionaryPromptTemplates` class creates prompts with selective examples:

```python
from modes.evolutionary_prompt_templates import EvolutionaryPromptTemplates

templates = EvolutionaryPromptTemplates(
    num_islands=4,
    max_programs_per_island=50,
    reset_period=100,
    reset_fraction=0.5
)

# Create dynamic prompt function
prompt_fn = templates.create_dynamic_evolutionary_prompt_function(
    var_names=["output", "x", "y"],
    var_descs=["target", "input1", "input2"],
    problem_name="test_problem",
    num_examples=3,
    temperature=1.0,
    sampling_strategy="boltzmann"
)
```

**Features:**
- **Selective inclusion**: Only adds examples that improve NMSE scores
- **Dynamic sampling**: Samples examples using specified strategy (boltzmann/best/random)
- **Temperature control**: Controls exploration vs exploitation in sampling
- **Automatic buffer management**: Adds successful programs to buffer automatically

### 3. Evolutionary Workflow (`workflow_evolutionary.py`)

The `LLMSREvolutionaryWorkflow` class integrates everything:

```python
from workflow_evolutionary import LLMSREvolutionaryWorkflow

workflow = LLMSREvolutionaryWorkflow(
    model_name="openai/gpt-4o-2024-08-06",
    max_iterations=5,
    num_islands=4,
    max_programs_per_island=50,
    num_examples=3,
    temperature=1.0,
    sampling_strategy="boltzmann"
)
```

**Features:**
- **Integrated workflow**: Combines generation, evaluation, and evolutionary selection
- **Comprehensive tracking**: Tracks evolutionary statistics and performance
- **Comparison tools**: Compare with baseline approaches
- **Flexible configuration**: Easy to adjust evolutionary parameters

## Usage Examples

### Basic Usage

```python
from workflow_evolutionary import LLMSREvolutionaryWorkflow

# Initialize workflow
workflow = LLMSREvolutionaryWorkflow(
    model_name="openai/gpt-4o-2024-08-06",
    max_iterations=5,
    num_islands=4,
    num_examples=3,
    temperature=1.0
)

# Setup dataset
workflow.setup_dataset("lsrtransform")

# Discover equations
result = workflow.discover_equation_evolutionary("kepler_1", "results")

# Get best equation
best_equation = workflow.get_best_equation_evolutionary("kepler_1")
print(f"Best NMSE: {best_equation['nmse_score']:.6f}")
```

### CLI Usage

```bash
# Single problem
python src/cli_evolutionary.py --problem "kepler_1" --output-dir "results"

# Multiple problems
python src/cli_evolutionary.py --problems "kepler_1" "kepler_2" "kepler_3"

# Custom parameters
python src/cli_evolutionary.py --problem "kepler_1" \
    --num-islands 8 \
    --num-examples 5 \
    --temperature 0.5 \
    --sampling-strategy boltzmann

# Compare with baseline
python src/cli_evolutionary.py --problem "kepler_1" \
    --compare-baseline "baseline_results.json"
```

### Demonstration

```bash
# Run full demonstration
python src/demo_evolutionary.py
```

## Evolutionary Parameters

### Core Parameters

- **`num_islands`** (int): Number of islands in the evolutionary buffer (default: 4)
- **`max_programs_per_island`** (int): Maximum programs per island (default: 50)
- **`reset_period`** (int): How often to reset weaker islands (default: 100)
- **`reset_fraction`** (float): Fraction of islands to reset (default: 0.5)

### Sampling Parameters

- **`num_examples`** (int): Number of examples to include in prompts (default: 3)
- **`temperature`** (float): Temperature for Boltzmann sampling (default: 1.0)
  - Lower values = more greedy (favor better scores)
  - Higher values = more exploration
- **`sampling_strategy`** (str): Sampling strategy
  - `"boltzmann"`: Boltzmann distribution (default)
  - `"best"`: Always select best programs
  - `"random"`: Random selection

## Evolutionary Algorithm Details

### 1. Multi-Island Architecture

The framework maintains multiple islands, each containing equation programs with similar characteristics:

```
Island 0: [Program A, Program B, Program C, ...]
Island 1: [Program D, Program E, Program F, ...]
Island 2: [Program G, Program H, Program I, ...]
Island 3: [Program J, Program K, Program L, ...]
```

### 2. Boltzmann Sampling

When sampling examples for prompts, the probability of selecting a program is:

```
P(program_i) = exp(-score_i / temperature) / sum(exp(-score_j / temperature))
```

This favors better-scored programs while maintaining diversity.

### 3. Island Reset Mechanism

Periodically, weaker islands are reset and repopulated with successful programs from better islands:

```
Before reset: [Island 0: score=0.1, Island 1: score=0.3, Island 2: score=0.8, Island 3: score=0.2]
After reset:  [Island 0: score=0.1, Island 1: score=0.3, Island 2: score=0.8, Island 3: score=0.8*]
```

### 4. Selective Example Inclusion

Only programs that improve the global best NMSE score are added to the buffer:

```python
if current_nmse < best_nmse_history:
    add_to_buffer(program, current_nmse)
    best_nmse_history = current_nmse
```

## Performance Comparison

The evolutionary framework typically provides:

- **Better convergence**: More consistent improvement in NMSE scores
- **Higher quality examples**: Only successful programs are used as examples
- **Diversity maintenance**: Multi-island approach prevents premature convergence
- **Adaptive sampling**: Boltzmann sampling balances exploration and exploitation

## Integration with Original LLMSR

The framework is also implemented in the original LLMSR codebase:

```python
from llmsr.evolutionary_searcher import EvolutionaryLLMSRSearcher

searcher = EvolutionaryLLMSRSearcher(
    name="evolutionary_llmsr",
    cfg=config,
    SamplerClass=sampler.Sampler,
    global_max_sample_num=100,
    log_path="logs",
    num_islands=4,
    num_examples=3,
    temperature=1.0
)
```

## Configuration Files

### Evolutionary Configuration

You can configure evolutionary parameters in your config files:

```yaml
# config.yaml
evolutionary:
  num_islands: 4
  max_programs_per_island: 50
  reset_period: 100
  reset_fraction: 0.5
  num_examples: 3
  temperature: 1.0
  sampling_strategy: "boltzmann"
```

## Troubleshooting

### Common Issues

1. **No examples in prompts**: Check if any programs have been added to the buffer
2. **Poor convergence**: Try adjusting temperature or number of islands
3. **Memory issues**: Reduce `max_programs_per_island`
4. **Slow performance**: Reduce `num_examples` or use "best" sampling strategy

### Debugging

Enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Check buffer statistics
stats = workflow.evolutionary_templates.get_buffer_statistics()
print(f"Buffer stats: {stats}")
```

## Future Enhancements

Potential improvements to the evolutionary framework:

1. **Adaptive temperature**: Automatically adjust temperature based on progress
2. **Crossover operations**: Combine successful programs to create new ones
3. **Mutation strategies**: Apply controlled mutations to explore nearby solutions
4. **Fitness sharing**: Prevent islands from converging to the same solution
5. **Multi-objective optimization**: Consider multiple metrics beyond NMSE

## References

- [Evolutionary Algorithms for Symbolic Regression](https://www.gp-field-guide.org.uk/)
- [Boltzmann Sampling in Evolutionary Computation](https://link.springer.com/article/10.1007/s10710-019-09360-6)
- [Multi-Island Genetic Algorithms](https://ieeexplore.ieee.org/document/6790660)


