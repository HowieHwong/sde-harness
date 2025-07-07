# Scientific Discovery Framework - Quick Integration Guide

## Overview

This is a scientific discovery framework that supports multiple AI providers and can be easily integrated into any project for iterative AI workflows.

## Core Modules

### 1. Generator
Responsible for unified interface management with multiple AI providers

#### Basic Configuration
Setup your api key inside credentials.yaml, and models.yaml, and copy a file to your Project folder.

```python
# In your code, remeber to setup weave
import weave
weave.init("Project Name")
```

```python
from sci_demo.generation import Generation
# Method 1: Using environment variables (For quick start, you don't exactally need to set any parameters.)
generator = Generation()

```

### 2. Prompt (Prompt Management)
Supports history-aware dynamic prompt management

#### Static Prompts (For fewshot experiment)
```python
from sci_demo.prompt import Prompt
# Give your {variable name} inside custom_template
# Give your varibale key-value pair dictionary inside default_vars 
prompt = Prompt(
    custom_template="Your prompt template: {task_description}\nData: {input_data}",
    default_vars={
        "task_description": "Your task description",
        "input_data": "Your input data"
    }
)
```

#### Dynamic Prompts (For iterative workflows)
```python
def dynamic_prompt_fn(iteration, history):
    """Adjust prompts based on iteration count and history"""
    #Set your first iteration (base) pprompt
    base_prompt = Prompt(
        template_name="Predefined Prompt Tamplate",
        default_vars={
            "task": "Your task",
            "data": "Your data"
        }
    )
    
    # Adjust prompts based on historical results
    if iteration > 1:
        # Get information from history
        previous_outputs = history.get("outputs", [])
        previous_scores = history.get("scores", [])

        # Do whatever operation you need for those varible 
        # TODO
        
        # Update prompt variables
        base_prompt.add_vars(
            previous_attempt=previous_outputs[-1] if previous_outputs else "",
            previous_score=previous_scores[-1] if previous_scores else {}
        )
    
    
    return base_prompt
```

### 3. Oracle (Evaluator)
Supports single-round and multi-round evaluation metrics

#### Register Evaluation Metrics
```python
from sci_demo.oracle import Oracle

oracle = Oracle()

# Single-round metric: evaluate individual output
def your_custom_metric(prediction, reference, **kwargs):
    """Define your evaluation metric"""
    # Implement evaluation logic
    score = calculate_score(prediction, reference, kwargs)
    return score

oracle.register_metric("your_metric_name", your_custom_metric)

# Multi-round metric: analyze iterative trends
def trend_analysis_metric(history, reference, current_iteration, **kwargs):
    """Analyze trends across multiple iterations"""
    # Get all outputs and scores from history
    outputs = history.get("outputs", [])
    scores = history.get("scores", [])
    
    # Analyze trends:  we can do calculation here with access to all the past LLM output, inside history
    
    # You Can update custom data in history for simplisity and efficiency
    history.setdefault("custom_data", []).append(trend_score)
    
    return trend_score

oracle.register_multi_round_metric("trend_analysis", trend_analysis_metric)
```

### 4. Workflow
Coordinates the entire iterative process

#### Basic Workflow
```python
from sci_demo.workflow import Workflow

workflow = Workflow(
    generator=generator,
    oracle=oracle,
    max_iterations=5,
    enable_multi_round_metrics=True
)

# Run workflow
result = workflow.run_sync(
    prompt=your_prompt_or_prompt_function,
    reference=your_reference_data,
    gen_args={"model": "gpt-4o", "max_tokens": 1000, "temperature": 0.7}
)
```

For other advance usage, you are more than welcome to read through other DOC and code example inside this repo.