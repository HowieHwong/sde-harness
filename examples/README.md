# SDE-Harness Examples

This directory contains comprehensive examples demonstrating how to use the SDE-Harness framework for scientific discovery and experimentation.

## 📁 Structure

```
examples/
├── README.md                          # This file
├── basic_usage/                       # Simple, focused examples
│   ├── 01_generation_basics.py        # Generation class usage
│   ├── 02_oracle_basics.py            # Oracle class usage  
│   ├── 03_prompt_basics.py            # Prompt class usage
│   ├── 04_workflow_basics.py          # Basic workflow usage
│   └── README.md                      # Basic usage guide
├── advanced_usage/                    # Advanced patterns and techniques
│   ├── 01_custom_metrics.py           # Custom metric development
│   ├── 02_multi_round_workflows.py    # Multi-iteration workflows
│   ├── 03_dynamic_prompts.py          # Dynamic prompt generation
│   ├── 04_error_handling.py           # Robust error handling
│   └── README.md                      # Advanced usage guide
├── project_examples/                  # Complete project examples
│   ├── simple_optimization/           # Simple optimization task
│   ├── text_classification/           # Text classification example
│   ├── molecular_design/              # Molecular design example
│   └── README.md                      # Project examples guide
└── integration_examples/              # Framework integration examples
    ├── 01_cli_integration.py          # CLI framework integration
    ├── 02_web_api_integration.py      # Web API integration
    ├── 03_jupyter_notebook.ipynb      # Jupyter notebook example
    └── README.md                      # Integration guide
```

## 🚀 Quick Start

### 1. Basic Usage
Start with the basic examples to understand individual components:

```bash
# Run generation example
python examples/basic_usage/01_generation_basics.py

# Run oracle example  
python examples/basic_usage/02_oracle_basics.py

# Run prompt example
python examples/basic_usage/03_prompt_basics.py

# Run workflow example
python examples/basic_usage/04_workflow_basics.py
```

### 2. Advanced Patterns
Explore advanced usage patterns:

```bash
# Custom metrics
python examples/advanced_usage/01_custom_metrics.py

# Multi-round workflows
python examples/advanced_usage/02_multi_round_workflows.py
```

### 3. Complete Projects
Run complete project examples:

```bash
# Simple optimization
cd examples/project_examples/simple_optimization
python run_optimization.py

# Text classification
cd examples/project_examples/text_classification  
python run_classification.py
```

## 📚 Learning Path

### Beginner
1. **Generation Basics** - Learn how to generate text with LLMs
2. **Oracle Basics** - Understand evaluation and metrics
3. **Prompt Basics** - Master prompt templating and variables
4. **Workflow Basics** - Combine components into workflows

### Intermediate  
1. **Custom Metrics** - Create domain-specific evaluation metrics
2. **Multi-round Workflows** - Build iterative improvement systems
3. **Dynamic Prompts** - Generate adaptive prompts based on history

### Advanced
1. **Complete Projects** - Study end-to-end implementations
2. **Integration Patterns** - Learn framework integration techniques
3. **Error Handling** - Build robust, production-ready systems

## 🎯 Use Cases

The examples cover common scientific discovery patterns:

- **Optimization**: Finding optimal solutions through iterative improvement
- **Classification**: Using LLMs for classification with custom evaluation
- **Generation**: Creative content generation with quality metrics
- **Discovery**: Exploring new possibilities through guided search
- **Analysis**: Analyzing and evaluating complex outputs

## 📋 Prerequisites

Before running the examples, ensure you have:

1. **SDE-Harness installed** with all dependencies
2. **API keys configured** in `config/credentials.yaml`
3. **Model configurations** set up in `config/models.yaml`
4. **Python environment** with required packages

## 🔧 Configuration

Most examples use the default configuration files:
- `config/models.yaml` - Model configurations
- `config/credentials.yaml` - API credentials

You can customize these for your specific needs.

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure SDE-Harness is properly installed
2. **API Key Errors**: Check your credentials configuration
3. **Model Errors**: Verify model names in configuration files
4. **Permission Errors**: Check file permissions and paths

### Getting Help

- Check individual example READMEs for specific guidance
- Review the main SDE-Harness documentation
- Look at test files for additional usage patterns

## 🤝 Contributing

To add new examples:

1. Follow the existing structure and naming conventions
2. Include clear docstrings and comments
3. Add error handling and edge case handling
4. Update relevant README files
5. Test thoroughly before submitting

## 📄 License

These examples are part of the SDE-Harness project and follow the same license terms.