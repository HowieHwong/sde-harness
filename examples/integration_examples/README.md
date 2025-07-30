# Integration Examples

This directory contains examples showing how to integrate the SDE-Harness framework with other systems and interfaces.

## Examples Overview

### 01_cli_integration.py
Demonstrates how to create command-line interfaces using the SDE-Harness base classes:
- Using `CLIBase` for consistent CLI patterns
- Argument parsing and validation
- Project integration
- Help system and error handling

### 02_web_api_integration.py
Shows how to wrap SDE-Harness workflows in web APIs:
- FastAPI integration
- Async request handling
- Error handling and validation
- Response formatting

### 03_jupyter_notebook.ipynb
Interactive Jupyter notebook demonstrating:
- Interactive workflow development
- Visualization of results
- Step-by-step exploration
- Real-time experimentation

## Usage

Each integration example is self-contained:

```bash
# CLI integration
python integration_examples/01_cli_integration.py --help

# Web API integration (requires FastAPI)
pip install fastapi uvicorn
python integration_examples/02_web_api_integration.py

# Jupyter notebook
jupyter notebook integration_examples/03_jupyter_notebook.ipynb
```

## Integration Patterns

### CLI Integration
- Consistent argument handling
- Project-specific extensions
- Configuration management
- Error handling and logging

### Web API Integration
- RESTful API design
- Async workflow execution
- Request validation
- Response serialization

### Notebook Integration
- Interactive development
- Result visualization
- Iterative experimentation
- Documentation integration

## Best Practices

1. **Error Handling**: Always handle exceptions gracefully
2. **Configuration**: Use external configuration files
3. **Logging**: Implement comprehensive logging
4. **Validation**: Validate inputs at integration boundaries
5. **Documentation**: Provide clear usage examples

## Prerequisites

- SDE-Harness framework
- Additional dependencies per example:
  - FastAPI and uvicorn for web API
  - Jupyter for notebook example
  - Matplotlib/Plotly for visualizations