# 🚀 Quick Setup Guide for Weave Tracking

## Installation

1. **Install Weave**:
```bash
pip install weave
```

2. **Install additional dependencies** (if not already installed):
```bash
pip install -r requirements_weave.txt
```

## Quick Start

1. **Set up your API keys** in `credentials.yaml`:
```yaml
openai:
  api_key: "your-openai-key"
gemini:
  api_key: "your-gemini-key"
claude:
  api_key: "your-claude-key"
```

2. **Configure your models** in `models.yaml`:
```yaml
gpt-4o:
  provider: "openai"
  model: "gpt-4o"
  credentials: "openai"
  __call_args:
    max_tokens: 1000
    temperature: 0.7
```

3. **Run the comprehensive tracking example**:
```bash
python examples/weave_tracking_example.py
```

## What You'll See

After running the example, check your Weave dashboard at [https://wandb.ai/](https://wandb.ai/) to see:

- ✅ Complete workflow execution traces
- ✅ Individual iteration details and performance
- ✅ Model generation metrics and efficiency
- ✅ Oracle evaluation results (single + multi-round)
- ✅ Prompt construction and history integration
- ✅ Trend analysis and performance evolution
- ✅ Custom scientific metrics and insights

## Automatic Tracking

The system now automatically tracks:

### 🔄 **Workflow Level** (`workflow.py`)
- Complete experiment runs
- Iteration-by-iteration progress
- Early stopping decisions
- Final analysis and trends

### 📡 **Generation Level** (`generation.py`)
- Model calls and responses
- Token usage and efficiency
- Provider performance
- Error handling

### 🎯 **Evaluation Level** (`oracle.py`)
- Single-round metric computation
- Multi-round historical analysis
- Batch evaluations
- Trend detection

### 📝 **Prompt Level** (`prompt.py`)
- Prompt construction
- Variable management
- History integration
- Template usage

## Example Usage

```python
import weave
from sci_demo.workflow import Workflow
from sci_demo.generation import Generation
from sci_demo.oracle import Oracle

# IMPORTANT: Initialize weave with YOUR project name before using the framework
weave.init("my_science_project")  # <- User must do this!

# Create components (all operations auto-tracked after weave.init)
gen = Generation()
oracle = Oracle()
workflow = Workflow(gen, oracle)

# Run experiment (comprehensive tracking)
result = workflow.run_sync(
    prompt=my_prompt,
    reference=reference_data,
    gen_args={"model_name": "gpt-4o"},
    history_context={"experiment": "drug_discovery_v1"}
)

# Check your Weave dashboard for complete insights!
```

## Advanced Features

### Custom Logging Within Operations
For custom experiment metadata, use `call.summary` within your tracked operations:

```python
@weave.op()
def my_experiment_function():
    # Your experiment logic here
    result = run_experiment()
    
    # Add custom experiment metadata to call summary
    call = weave.get_current_call()
    if call and call.summary:
        call.summary.update({
            "experiment_metadata": {
                "researcher": "Dr. Smith",
                "hypothesis": "Iterative prompting improves accuracy",
                "domain": "computational_chemistry"
            }
        })
    
    return result
```

### Experiment Comparison
```python
@weave.op()
def compare_configurations(experiment_configs):
    results = []
    for config in experiment_configs:
        # Log configuration to call summary
        call = weave.get_current_call()
        if call and call.summary:
            call.summary.update({"config": config})
        
        result = workflow.run_sync(...)
        results.append(result)
    
    return results
```

## Troubleshooting

- **Missing logs**: **CRITICAL** - You must call `weave.init("your_project_name")` BEFORE importing or creating any sci_demo components
- **Framework usage**: This is a framework - you control the project name via `weave.init()`
- **Authentication**: Make sure you're logged into Weights & Biases
- **Performance**: Tracking is lightweight, but can be disabled in production by not calling `weave.init()`

## Next Steps

1. ✅ Run the example to see tracking in action
2. ✅ Explore your Weave dashboard
3. ✅ Add custom metrics for your domain
4. ✅ Use insights to optimize your workflows
5. ✅ Scale to larger experiments with full observability

**🧪 Happy experimenting with complete visibility into your science workflows!** 