# 🧪 Comprehensive Weave Tracking for Science Workflows

This guide explains the comprehensive Weave tracking implementation across your science workflow system. With this setup, you'll have complete observability into every aspect of your iterative AI experiments.

## 🎯 What's Being Tracked

### 1. **Complete Workflow Orchestration** (`sci_demo/workflow.py`)

#### Main Workflow Execution (`@weave.op() run()`)
- **Workflow Configuration**: Max iterations, history settings, multi-round metrics
- **Iteration Start/End**: Progress tracking for each iteration
- **Early Stopping**: When and why workflows terminate early
- **Final Analysis**: Complete results, best iteration, trend analysis

#### Individual Iterations (`@weave.op() _run_single_iteration()`)
- **Prompt Information**: Length, type, history integration status
- **Generation Results**: Output length, evaluation type used
- **Iteration Performance**: Scores, metrics used, evaluation method

#### Trend Analysis (`@weave.op() _analyze_trends()`)
- **Performance Evolution**: How metrics change across iterations
- **Best Iteration Detection**: Which iteration performed best
- **Convergence Analysis**: Whether the model is converging

### 2. **Model Generation** (`sci_demo/generation.py`)

#### Generation Calls (`@weave.op() generate()`)
- **Input/Output**: Prompt and response tracking
- **Model Configuration**: Which model, parameters used
- **Performance Metrics**: Token usage, generation time
- **Provider Information**: Whether using OpenAI, Claude, local models, etc.

### 3. **Oracle Evaluations** (`sci_demo/oracle.py`)

#### Single-Round Metrics (`@weave.op() compute()`)
- **Metric Registration**: When new metrics are added
- **Evaluation Process**: Which metrics computed, input lengths
- **Individual Results**: Score for each metric
- **Performance Summary**: Aggregate metric performance

#### Multi-Round Metrics (`@weave.op() compute_with_history()`)
- **Historical Context**: History length, current iteration
- **Multi-Round Analysis**: Improvement rates, consistency, convergence
- **Combined Results**: Both single-round and multi-round metric results
- **Trend Detection**: Performance patterns across iterations

#### Batch Evaluations (`@weave.op() evaluate_batch()`)
- **Batch Statistics**: Mean, std, min, max for each metric
- **Batch Size**: Number of examples processed
- **Performance Distribution**: How scores are distributed

#### Trend Metrics (`@weave.op() compute_trend_metrics()`)
- **Improvement Rates**: Rate of improvement across iterations
- **Consistency Measures**: How stable performance is
- **Best/Worst Performance**: Peak and lowest scores
- **Monotonic Improvement**: Whether performance consistently improves

### 4. **Prompt Construction** (`sci_demo/prompt.py`)

#### Prompt Initialization (`__init__()`)
- **Template Information**: Type (builtin/custom), name, length
- **Variable Counts**: How many default variables provided

#### Variable Management (`@weave.op() add_vars()`)
- **Variable Updates**: Which variables added/changed
- **Variable Count Evolution**: How variable count changes

#### History Integration (`@weave.op() add_history()`)
- **History Status**: First iteration vs. subsequent iterations
- **History Statistics**: Length, previous iterations, score availability
- **Context Integration**: How much history is being incorporated

#### Prompt Building (`@weave.op() build()`)
- **Build Success**: Final prompt length, variables used
- **Template Processing**: Placeholder filling statistics
- **Build Errors**: Missing variables and debugging info

#### Example Management (`@weave.op() add_example()`)
- **Few-shot Examples**: Number of examples, their lengths

#### File Operations (`@weave.op() save/load()`)
- **Persistence**: Template and variable saving/loading

## 🚀 How to Use the Tracking

### 1. **Basic Setup**

```python
import weave
from sci_demo.generation import Generation
from sci_demo.workflow import Workflow
from sci_demo.oracle import Oracle

# IMPORTANT: Initialize weave with YOUR project name FIRST
# This is a framework - YOU control the project name
weave.init("my_science_project")

# Create components (tracking is automatic after weave.init)
gen = Generation()
oracle = Oracle()
workflow = Workflow(gen, oracle)
```

### 2. **Running Tracked Experiments**

```python
# All these operations are automatically tracked
result = workflow.run_sync(
    prompt=my_prompt,
    reference=reference_data,
    gen_args={"model_name": "gpt-4o", "max_tokens": 200},
    history_context={"experiment_id": "exp_001"}
)
```

### 3. **Custom Logging for Domain-Specific Insights**

```python
# Add custom logs for your specific domain
call.summary.update({
    "experiment_metadata": {
        "researcher": "Dr. Smith",
        "project": "Drug Discovery AI",
        "hypothesis": "Iterative prompting improves chemical accuracy"
    }
})
```

### 4. **Tracking Custom Metrics**

```python
# Register custom metrics with automatic tracking
def domain_specific_metric(pred, ref, **kwargs):
    # Your custom evaluation logic
    return score

oracle.register_metric("domain_metric", domain_specific_metric)
# Registration is automatically logged
```

## 📊 Weave Dashboard Views

### **Workflow Execution Timeline**
- See complete workflow runs from start to finish
- Track iteration-by-iteration progress
- Identify bottlenecks and optimization opportunities

### **Model Performance Dashboard**
- Compare performance across different models
- Track token usage and generation efficiency
- Monitor model-specific metrics

### **Evaluation Metrics Evolution**
- Visualize how metrics change across iterations
- Identify convergence patterns
- Compare single-round vs multi-round metrics

### **Prompt Engineering Insights**
- Track prompt length and complexity evolution
- Monitor history integration effectiveness
- Analyze template usage patterns

### **Trend Analysis Views**
- Improvement rate tracking
- Consistency analysis
- Convergence detection

## 🔍 Key Metrics to Monitor

### **Workflow-Level Metrics**
- `total_iterations`: How many iterations completed
- `final_scores`: End performance
- `best_iteration`: Peak performance iteration
- `early_stopping`: Frequency and reasons for early termination

### **Generation Metrics**
- Token usage and efficiency
- Generation time
- Model switching patterns
- Error rates

### **Evaluation Metrics**
- Individual metric scores
- Metric improvement rates
- Consistency measures
- Convergence indicators

### **Prompt Metrics**
- Prompt length evolution
- History integration success
- Template effectiveness
- Variable usage patterns

## 🎛️ Advanced Tracking Patterns

### **Experiment Comparison**
```python
# Track different experimental configurations
for config in experiment_configs:
    call.summary.update({"experiment_config": config})
    result = workflow.run_sync(...)
    call.summary.update({"experiment_result": result})
```

### **A/B Testing Workflows**
```python
# Compare different prompt strategies
for strategy in ["iterative", "few_shot", "chain_of_thought"]:
    call.summary.update({"strategy": strategy})
    prompt = create_prompt_for_strategy(strategy)
    result = workflow.run_sync(prompt=prompt, ...)
```

### **Performance Optimization**
```python
# Track optimization experiments
for temperature in [0.3, 0.5, 0.7, 0.9]:
    call.summary.update({"temperature_experiment": temperature})
    result = workflow.run_sync(
        gen_args={"temperature": temperature, ...}
    )
```

## 🛠️ Troubleshooting Tracking

### **Common Issues**

1. **Missing Logs**: **CRITICAL** - You must call `weave.init("your_project_name")` BEFORE importing or using any sci_demo components. This is a framework, not an application.
2. **Framework Design**: sci_demo is designed as a framework. Users control the weave project name and initialization.
3. **Performance Impact**: Weave tracking is lightweight, but can be completely disabled by not calling `weave.init()`
4. **Storage**: Weave handles data storage automatically once initialized

### **Debug Mode**
```python
# Enable detailed logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### **Selective Tracking**
```python
# Disable tracking for specific components if needed
# (Remove @weave.op() decorators or use conditional decorators)
```

## 📈 Success Metrics

### **Experiment Success Indicators**
- **Convergence Rate**: How often experiments converge successfully
- **Improvement Consistency**: Stable improvement across iterations
- **Efficiency**: Time to reach target performance
- **Resource Usage**: Token consumption per improvement unit

### **Model Performance**
- **Accuracy Trends**: Consistent improvement in domain metrics
- **Consistency**: Low variance in repeated experiments
- **Efficiency**: Faster convergence with experience

### **System Health**
- **Error Rates**: Low failure rates in generation/evaluation
- **Resource Utilization**: Efficient use of API calls and compute
- **Scalability**: Performance with increasing complexity

## 🔗 Integration with Existing Workflows

The tracking is designed to be **zero-configuration** for existing code:

1. **No Code Changes Required**: Existing workflow code works unchanged
2. **Automatic Discovery**: All operations are tracked automatically
3. **Rich Context**: Historical data and trends captured seamlessly
4. **Performance Monitoring**: Built-in performance and efficiency tracking

## 🎯 Next Steps

1. **Run the Example**: Execute `examples/weave_tracking_example.py`
2. **Explore Dashboard**: Check your Weave project dashboard
3. **Customize Metrics**: Add domain-specific metrics and logging
4. **Optimize Workflows**: Use insights to improve your experiments
5. **Scale Experiments**: Leverage tracking for large-scale research

---

**🧪 Happy Experimenting with Complete Observability!**

Your science workflows now have comprehensive tracking across every component, giving you unprecedented insight into your AI research process. 