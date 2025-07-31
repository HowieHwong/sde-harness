# Basic Usage Examples

This directory contains simple, focused examples that demonstrate the core functionality of each SDE-Harness component.

## Examples Overview

### 01_generation_basics.py
- **Purpose**: Learn the fundamentals of the Generation class
- **Covers**: 
  - Basic text generation with different models
  - Async generation patterns
  - Error handling for API calls
  - Model information retrieval

### 02_oracle_basics.py  
- **Purpose**: Understand evaluation and metrics with the Oracle class
- **Covers**:
  - Metric registration and management
  - Single-round evaluation
  - Multi-round evaluation with history
  - Custom metric development

### 03_prompt_basics.py
- **Purpose**: Master prompt templating and variable management
- **Covers**:
  - Built-in template usage
  - Custom template creation
  - Variable substitution
  - Dynamic prompt building

### 04_workflow_basics.py
- **Purpose**: Learn to combine components into workflows
- **Covers**:
  - Basic workflow setup
  - Synchronous and asynchronous execution
  - Multi-iteration workflows
  - Result analysis

## Running the Examples

Each example is self-contained and can be run independently:

```bash
# Run from the project root directory
python examples/basic_usage/01_generation_basics.py
python examples/basic_usage/02_oracle_basics.py  
python examples/basic_usage/03_prompt_basics.py
python examples/basic_usage/04_workflow_basics.py
```

## Prerequisites

1. **Configuration Files**: Ensure you have:
   - `config/models.yaml` with your model configurations
   - `config/credentials.yaml` with your API keys

2. **Dependencies**: Install required packages:
   ```bash
   pip install litellm transformers numpy pandas
   ```

## Learning Progression

Start with `01_generation_basics.py` and work through the examples in order. Each example builds on concepts from the previous ones.

## Customization

Feel free to modify these examples:
- Change model names to match your configuration
- Adjust parameters like temperature, max_tokens
- Add your own metrics and evaluation logic
- Experiment with different prompt templates

## Next Steps

After completing these basic examples, move on to:
- **Advanced Usage**: More sophisticated patterns and techniques
- **Project Examples**: Complete end-to-end implementations
- **Integration Examples**: Framework integration patterns