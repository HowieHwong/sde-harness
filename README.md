# Science Discovery Framework

A science discovery workflow framework powered by Large Language Models, supporting iterative generate-evaluate-feedback loops designed specifically for scientific research and discovery tasks.

## 🎯 Project Overview

Science_Demo is a modular framework designed to build intelligent science discovery pipelines by combining multiple components:

- **Generation Module**: Unified text generation interface supporting both OpenAI API and Hugging Face models
- **Prompt Management**: Flexible prompt template system with dynamic variables and few-shot learning
- **Oracle Module**: Extensible evaluation system with custom metrics support
- **Workflow**: Core orchestrator supporting iterative optimization and dynamic feedback

## 🏗️ Project Architecture

```
sci_demo/
├── generation.py    # Text generation module (OpenAI + HuggingFace)
├── prompt.py        # Prompt template management system
├── oracle.py        # Evaluation metrics computation module
└── workflow.py      # Main workflow orchestrator
```

## 🚀 Core Features

### 1. Unified Generation Interface
- Supports OpenAI GPT series models
- Supports Hugging Face open-source models (e.g., Llama, Qwen, etc.)
- Asynchronous batch generation
- Automatic device management (CPU/GPU)

### 2. Flexible Prompt Management
- Built-in template system (summarization, Q&A, translation, few-shot learning)
- Dynamic variable substitution
- Template saving and loading
- Few-shot example management

### 3. Extensible Evaluation System
- Custom evaluation metrics
- Batch evaluation support
- Flexible parameter passing

### 4. Intelligent Workflow Orchestration
- Iterative optimization loops
- Dynamic prompt and metric selection
- Custom stopping criteria
- History tracking and management

## 📦 Installation

The project requires the following main dependencies:

```bash
pip install torch transformers openai asyncio
```

Detailed dependency list:
- `torch`: PyTorch deep learning framework
- `transformers`: Hugging Face model library
- `openai`: OpenAI API client
- `asyncio`: Asynchronous programming support

## 🔧 Quick Start

### Basic Usage Example

```python
from sci_demo.generation import Generation
from sci_demo.prompt import Prompt
from sci_demo.oracle import Oracle
from sci_demo.workflow import Workflow

# 1. Initialize components
generator = Generation(
    api_key="your_openai_key",  # or use HF model
    hf_model_name="meta-llama/Llama-3-7B"
)

oracle = Oracle()
oracle.register_metric("accuracy", lambda pred, ref: float(pred.strip() == ref.strip()))

# 2. Create prompt
prompt = Prompt(
    template_name="summarize",
    default_vars={"input_text": "Recent breakthroughs in protein folding."}
)

# 3. Run workflow
workflow = Workflow(generator=generator, oracle=oracle, max_iterations=3)
result = workflow.run_sync(
    prompt=prompt,
    reference="AlphaFold revolutionized protein structure prediction.",
    gen_args={"max_tokens": 100, "temperature": 0.7}
)

print(result)
```

### Advanced Usage: Dynamic Workflow

```python
# Dynamic prompt function
def dynamic_prompt(iteration, history):
    base_prompt = Prompt(template_name="summarize")
    if iteration > 1:
        # Adjust prompt based on historical results
        last_score = history["scores"][-1] if history["scores"] else {}
        if last_score.get("accuracy", 0) < 0.5:
            base_prompt.add_vars(additional="Please be more accurate and detailed")
    return base_prompt

# Dynamic metrics function
def dynamic_metrics(iteration, history):
    if iteration <= 2:
        return ["length"]
    return ["accuracy", "relevance"]

# Custom stopping condition
def stop_condition(context):
    return context["scores"].get("accuracy", 0) >= 0.9

workflow = Workflow(
    generator=generator,
    oracle=oracle,
    max_iterations=5,
    stop_criteria=stop_condition
)

result = workflow.run_sync(
    prompt=dynamic_prompt,
    reference="Target text",
    metrics=dynamic_metrics
)
```

## 📚 Detailed Documentation

### Generation Module

Supports two generation modes:

**OpenAI API Mode:**
```python
gen = Generation(api_key="your_key")
result = gen.generate("Explain quantum computing", max_tokens=100)
```

**Hugging Face Local Model:**
```python
gen = Generation(hf_model_name="meta-llama/Llama-3-7B")
result = gen.generate("Explain quantum computing", max_tokens=100)
```

### Prompt Module

**Built-in Template Usage:**
```python
# Summarization template
prompt = Prompt(template_name="summarize", default_vars={"input_text": "Text content"})

# Q&A template
prompt = Prompt(template_name="qa", default_vars={
    "context": "Context information",
    "question": "Question"
})

# Custom template
prompt = Prompt(
    custom_template="Please analyze the following scientific paper: {paper_content}",
    default_vars={"paper_content": "Paper content"}
)
```

**Few-shot Learning:**
```python
fs_prompt = Prompt(template_name="few_shot")
fs_prompt.add_example("Input example 1", "Output example 1")
fs_prompt.add_example("Input example 2", "Output example 2")
fs_prompt.add_vars(input_text="New input")
```

### Oracle Module

**Register Custom Metrics:**
```python
oracle = Oracle()

# Accuracy metric
def accuracy(prediction, reference, **kwargs):
    return float(prediction.strip().lower() == reference.strip().lower())

# BLEU score metric
def bleu_score(prediction, reference, **kwargs):
    # Implement BLEU calculation logic
    return 0.75

oracle.register_metric("accuracy", accuracy)
oracle.register_metric("bleu", bleu_score)

# Compute metrics
scores = oracle.compute(prediction="Predicted text", reference="Reference text")
```

## 🔬 Application Scenarios

1. **Scientific Literature Summarization and Optimization**
2. **Hypothesis Generation and Validation**
3. **Experimental Design Recommendations**
4. **Data Analysis Report Generation**
5. **Scientific Question-Answering Systems**
6. **Research Direction Exploration**

## 🛠️ Extension Development

### Adding New Evaluation Metrics

```python
def custom_metric(prediction, reference, **kwargs):
    # Implement your evaluation logic
    score = calculate_score(prediction, reference)
    return score

oracle.register_metric("custom_metric", custom_metric)
```

### Creating Custom Workflows

```python
class CustomWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def custom_processing(self, output):
        # Add custom processing logic
        return processed_output
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

We welcome Issues and Pull Requests!

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📧 Contact

For questions or suggestions, please contact us through Issues.

---

**Note**: Please ensure you have properly configured your OpenAI API key or downloaded the corresponding Hugging Face models before use.