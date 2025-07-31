# Simple Optimization Example

This example demonstrates a simplified optimization task using the SDE-Harness framework, inspired by the LLMEO project but made more accessible for learning.

## Objective

Optimize mathematical expressions to achieve target values through iterative improvement using LLM-generated suggestions.

## What This Example Demonstrates

- **Iterative Optimization**: Multi-round workflow with improvement tracking
- **Custom Metrics**: Domain-specific evaluation functions
- **Dynamic Prompts**: Adaptive prompts based on previous results
- **Data Management**: Handling optimization history and progress
- **Stop Criteria**: Intelligent stopping conditions

## Files

- `run_optimization.py` - Main script to run the optimization
- `config.yaml` - Configuration file with parameters
- `sample_data.json` - Sample target values for optimization
- `README.md` - This file

## How It Works

1. **Initialization**: Start with random mathematical expressions
2. **Evaluation**: Compute how close expressions are to target values
3. **Generation**: Use LLM to suggest improved expressions
4. **Assessment**: Evaluate new expressions against targets
5. **Iteration**: Repeat until target is reached or max iterations

## Usage

```bash
# Navigate to the example directory
cd examples/project_examples/simple_optimization

# Run the basic optimization
python run_optimization.py

# Run with custom parameters
python run_optimization.py --target 42 --max-iterations 10

# Run with different mathematical operations
python run_optimization.py --operations "+-*/" --complexity 3
```

## Configuration

Edit `config.yaml` to customize:
- Target values to optimize for
- Maximum number of iterations
- Allowed mathematical operations
- Evaluation thresholds
- Model parameters

## Expected Output

```
🎯 Simple Optimization Example
==============================

Target: 42
Starting expressions: ['5 + 3', '10 * 2', '7 + 8']

Iteration 1:
  Best expression: 10 * 2 = 20
  Error: 22.0
  Generating improvements...

Iteration 2:
  Best expression: 20 + 22 = 42
  Error: 0.0
  🎉 Target achieved!

Results:
  Iterations: 2
  Final expression: 20 + 22
  Final value: 42
  Success: True
```

## Learning Objectives

After running this example, you'll understand:
- How to set up iterative optimization workflows
- Creating custom evaluation metrics
- Using LLMs for creative problem solving
- Managing optimization state and history
- Implementing intelligent stopping conditions

## Extensions

Try these modifications to deepen your understanding:
- Add more complex mathematical functions (sin, cos, log)
- Optimize for multiple targets simultaneously
- Add constraints (e.g., expressions must be under certain length)
- Implement different optimization strategies
- Add visualization of the optimization process

## Troubleshooting

**Common Issues:**
- **No improvement**: Try adjusting temperature or prompt templates
- **Invalid expressions**: Check expression validation logic
- **Slow convergence**: Increase max iterations or adjust target tolerance
- **API errors**: Verify your model configuration and API keys