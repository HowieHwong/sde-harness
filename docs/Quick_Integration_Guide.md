# Scientific Discovery Framework - Quick Integration Guide

## Overview

This is a scientific discovery framework that supports multiple AI providers and can be easily integrated into any project for iterative AI workflows.

## Core Modules

### 1. Generator
Responsible for unified interface management with multiple AI providers

#### Basic Configuration
Setup your api key inside credentials.yaml and models.yaml in the root path of sde-harness.

```python
# In your code, remember to setup weave
import weave
weave.init("Project Name")
```

```python
from sde_harness.core.generation import Generation
# Retrive Models and Credentials from default yaml path (For quick start, you don't exactly need to set any parameters.)
generator = Generation()

```
Certain LLM providers might require extra generation config parameter when you calling the llm. To cope with this, you can either edit the models.yaml file or pass it in during workflow.run_sync() gen_arg parameter. 

### 2. Prompt (Prompt Management)


#### Static Prompts (For fewshot experiment)
```python
from sde_harness.core.prompt import Prompt
# Include your {variable name} in the custom_template.
# Provide your variable key-value pairs in the default_vars dictionary 
prompt = Prompt(
    custom_template="Your prompt template: {task_description}\nData: {input_data}",
    default_vars={
        "task_description": "Your task description",
        "input_data": "Your input data"
    }
)
```



#### Dynamic Prompts (For iterative workflows)
This module supports history-aware dynamic prompt management.
```python
def dynamic_prompt_fn(iteration, history):
    """Adjust prompts based on iteration count and history"""
    #Define your first iteration (base) prompt
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

        # Perform any necessary operations on these variables
        # TODO
        
        # Update prompt variables
        base_prompt.add_vars(
            previous_attempt=previous_outputs[-1] if previous_outputs else "",
            previous_score=previous_scores[-1] if previous_scores else {}
        )
    
    
    return base_prompt
```

### 3. Oracle (Evaluator)
This module supports single-round and multi-round evaluation metrics.

#### Register Evaluation Metrics
```python
from sde_harness.core.oracle import Oracle

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
    
    # Analyze trends by performing calculations using the past LLM outputs available in the 'history' parameter."
    
    # You can update custom data in history for simplicity and efficiency
    history.setdefault("custom_data", []).append(trend_score)
    
    return trend_score

oracle.register_multi_round_metric("trend_analysis", trend_analysis_metric)
```

### 4. Workflow
Coordinates the entire iterative process

#### Basic Workflow
```python
from sde_harness.core.workflow import Workflow

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
    gen_args={"model": {model_name_from_models.yaml}}, "max_tokens": 1000, "temperature": 0.7...}

)
```
For advanced usage, refer to the additional documentation and code examples in this repository.