# SDE-Harness Examples - Complete Overview

This document provides a comprehensive overview of all examples in the SDE-Harness framework, helping you navigate and choose the right examples for your learning journey.

## 📚 Learning Path

### 🌱 **Beginner** (Start Here)
**Goal**: Understand individual components and basic usage patterns

1. **`basic_usage/01_generation_basics.py`**
   - Learn text generation with LLMs
   - Understand async patterns
   - Model management and configuration
   - Error handling basics

2. **`basic_usage/02_oracle_basics.py`**
   - Create and register evaluation metrics
   - Single-round and multi-round evaluation
   - Batch processing
   - Custom metric development

3. **`basic_usage/03_prompt_basics.py`**
   - Built-in and custom templates
   - Variable substitution
   - Dynamic prompt building
   - Template management

4. **`basic_usage/04_workflow_basics.py`**
   - Combine components into workflows
   - Multi-iteration processes
   - Stop criteria and convergence
   - Sync and async execution

### 🚀 **Intermediate** (Build Complete Systems)
**Goal**: Create end-to-end applications and understand integration patterns

5. **`project_examples/simple_optimization/`**
   - Complete optimization project
   - Iterative improvement workflows
   - Data-driven decision making
   - Configuration management

6. **`integration_examples/01_cli_integration.py`**
   - CLI application development
   - Using CLIBase for consistent interfaces
   - Argument parsing and validation
   - Production-ready command tools

### 🔬 **Advanced** (Master Complex Patterns)
**Goal**: Implement sophisticated evaluation and workflow patterns

7. **`advanced_usage/01_custom_metrics.py`**
   - Advanced metric development
   - Composite and weighted metrics
   - Domain-specific evaluation
   - Metric composition patterns

## 🎯 Examples by Use Case

### 📊 **Scientific Research**
- `basic_usage/04_workflow_basics.py` - Research methodology workflows
- `project_examples/simple_optimization/` - Mathematical optimization
- `advanced_usage/01_custom_metrics.py` - Research quality metrics

### 💻 **Software Development**
- `basic_usage/01_generation_basics.py` - Code generation and documentation
- `integration_examples/01_cli_integration.py` - Developer tool creation
- `advanced_usage/01_custom_metrics.py` - Code quality evaluation

### 🎓 **Education and Learning**
- `basic_usage/02_oracle_basics.py` - Automated assessment systems
- `basic_usage/03_prompt_basics.py` - Educational content generation
- `project_examples/simple_optimization/` - Interactive learning tools

### 🏭 **Production Systems**
- `integration_examples/01_cli_integration.py` - Production CLI tools
- `advanced_usage/01_custom_metrics.py` - Quality assurance systems
- All examples include error handling patterns

## 🔧 Examples by Technical Focus

### **Generation and LLM Usage**
- `basic_usage/01_generation_basics.py` - Core generation patterns
- `basic_usage/04_workflow_basics.py` - Generation in workflows
- `project_examples/simple_optimization/` - Iterative generation

### **Evaluation and Metrics**
- `basic_usage/02_oracle_basics.py` - Basic evaluation patterns
- `advanced_usage/01_custom_metrics.py` - Advanced metric development
- `project_examples/simple_optimization/` - Domain-specific metrics

### **Prompt Engineering**
- `basic_usage/03_prompt_basics.py` - Template and variable management
- `basic_usage/04_workflow_basics.py` - Dynamic prompt generation
- `project_examples/simple_optimization/` - Adaptive prompts

### **Workflow Orchestration**
- `basic_usage/04_workflow_basics.py` - Multi-iteration workflows
- `project_examples/simple_optimization/` - Complex workflow patterns
- `integration_examples/01_cli_integration.py` - Workflow integration

## 🎨 Examples by Complexity

### **Simple** (< 100 lines, single concept)
- Individual functions in `basic_usage/` directory
- Focused demonstrations of specific features
- Quick experimentation and learning

### **Medium** (100-300 lines, integrated concepts)
- Complete examples in `basic_usage/`
- `integration_examples/01_cli_integration.py`
- Multi-component demonstrations

### **Complex** (300+ lines, production patterns)
- `project_examples/simple_optimization/`
- `advanced_usage/01_custom_metrics.py`
- Real-world application patterns

## 🚦 Getting Started Guide

### **I'm New to SDE-Harness**
1. Start with `basic_usage/01_generation_basics.py`
2. Continue through basic_usage/ in order
3. Try the simple_optimization project
4. Explore integration examples

### **I Want to Build a Specific Application**
1. Check the "Examples by Use Case" section above
2. Start with the most relevant basic example
3. Study the corresponding project example
4. Adapt patterns to your needs

### **I Need Advanced Patterns**
1. Ensure you understand all basic concepts
2. Study `advanced_usage/01_custom_metrics.py`
3. Examine complex project examples
4. Implement custom patterns

### **I'm Integrating with Existing Systems**
1. Review `integration_examples/01_cli_integration.py`
2. Study the CLIBase and ProjectBase patterns
3. Adapt integration patterns to your stack
4. Implement error handling and validation

## 🛠️ Prerequisites by Example Level

### **Basic Examples**
```bash
# Required configuration files
config/models.yaml       # Model configurations
config/credentials.yaml  # API credentials

# Python dependencies
pip install litellm transformers numpy pandas
```

### **Project Examples**
```bash
# Additional dependencies
pip install yaml

# Sample data files (included)
# Configuration files (included per project)
```

### **Advanced Examples**
```bash
# Full framework understanding
# Experience with Python classes and inheritance
# Understanding of evaluation metrics
```

## 🎓 Learning Objectives by Example

### **After basic_usage/**
- ✅ Understand all core SDE-Harness components
- ✅ Can create simple workflows
- ✅ Know how to evaluate outputs
- ✅ Can manage prompts and templates

### **After project_examples/**
- ✅ Can build complete applications
- ✅ Understand configuration management
- ✅ Can implement iterative improvement
- ✅ Know production deployment patterns

### **After integration_examples/**
- ✅ Can integrate with existing systems
- ✅ Know CLI development patterns
- ✅ Understand error handling
- ✅ Can create user-friendly interfaces

### **After advanced_usage/**
- ✅ Can create sophisticated evaluation metrics
- ✅ Understand metric composition
- ✅ Can implement complex workflows
- ✅ Know performance optimization techniques

## 🔄 Recommended Learning Sequence

```
1. basic_usage/01_generation_basics.py
   ↓
2. basic_usage/02_oracle_basics.py
   ↓
3. basic_usage/03_prompt_basics.py
   ↓
4. basic_usage/04_workflow_basics.py
   ↓
5. project_examples/simple_optimization/
   ↓
6. integration_examples/01_cli_integration.py
   ↓
7. advanced_usage/01_custom_metrics.py
```

## 🤝 Contributing New Examples

When adding new examples:

1. **Follow the existing structure**
   - Clear docstrings and comments
   - Consistent naming and organization
   - Comprehensive error handling

2. **Include learning objectives**
   - What will users learn?
   - What prerequisites are needed?
   - How does it build on previous examples?

3. **Provide multiple complexity levels**
   - Simple demonstration of concept
   - Intermediate integration patterns
   - Advanced production techniques

4. **Test thoroughly**
   - Verify all examples run correctly
   - Include error scenarios
   - Test with different configurations

## 📞 Getting Help

- **For specific examples**: Check the README in each directory
- **For general issues**: Review the main SDE-Harness documentation
- **For advanced patterns**: Study the source code and tests
- **For contributions**: Follow the contributing guidelines

## 🎉 Next Steps

After completing these examples, you'll be ready to:
- Build production SDE-Harness applications
- Create domain-specific evaluation systems
- Integrate SDE-Harness with existing workflows
- Contribute back to the framework

Happy learning! 🚀