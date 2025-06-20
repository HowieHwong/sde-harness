# LLMEO - LLM-based Evolutionary Optimization

## Original Code Repository
[https://github.com/deepprinciple/llmeo](https://github.com/deepprinciple/llmeo)



## 📦 Install

1. Get to the Project folder:
   ```bash
   cd Projects/LLMEO
   ```

2. Set Your API KEY:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. Download required dataset:
   ```bash
   wget https://zenodo.org/records/14328055/files/ground_truth_fitness_values.csv -P data/
   ```

4. Set conda env:
   ```bash
   conda env create -f environment.yml
   conda activate ScienceBench_LLMEO
   ```

5. (Optional) Run Test File:
   ```bash
   python test.py
   ```

## 🎯 Usage

### Command Line Interface

LLMEO provides a command line interface similar to the steer project, supporting multiple running modes:

#### Basic Usage

```bash
# Few-shot learning mode
python cli.py few-shot

# Single property optimization mode
python cli.py single-prop

# Multi-property optimization mode
python cli.py multi-prop

```

#### Running with Parameters

```bash
# Few-shot mode, 3 iterations, temperature 0.1
python cli.py few-shot --iterations 3 --temperature 0.1

# Multi-property optimization, 5000 tokens, 20 samples
python cli.py multi-prop --max-tokens 5000 --samples 20

# Single property optimization, generate 15 samples
python cli.py single-prop --num-samples 15

```

#### View Help

```bash
# View all modes
python cli.py --help

# View help for specific mode
python cli.py few-shot --help
```

### Common Parameters

All modes support the following parameters:

- `--api-key`: OpenAI API key
- `--samples`: Initial sample number (default: 10)
- `--num-samples`: Generated sample number (default: 10)
- `--max-tokens`: Maximum token number (default: 5000)
- `--temperature`: Temperature parameter (default: 0.0)
- `--seed`: Random seed (default: 42)

## 🏗️ Project Structure

```
Projects/LLMEO/
├── cli.py              # Command line entry point
├── main.py             # Original main file (preserved)
├── modes/              # Running mode modules
│   ├── __init__.py
│   ├── few_shot.py     # Few-shot learning mode
│   ├── single_prop.py  # Single property optimization mode
│   ├── multi_prop.py   # Multi-property optimization mode
│   
├── config/             # Configuration management
│   ├── __init__.py
│   └── settings.py     # Configuration management module
├── utils/              # Utility modules
│   ├── __init__.py
│   └── data_loader.py  # Data loading utilities
├── prompt.py           # Prompt templates
├── _utils.py           # Utility functions
├── test.py             # Test file
├── environment.yml     # Environment configuration
└── README.md           # This document
```

## 🔧 Extension Development

### Adding New Running Modes

1. Create a new mode file in the `modes/` directory
2. Implement the mode function, e.g., `run_new_mode(args)`
3. Import the new function in `modes/__init__.py`
4. Add new subcommand in `cli.py`

Example:

```python
# modes/new_mode.py
def run_new_mode(args):
    """Logic for running new mode"""
    print("🆕 Running new mode...")
    # Implement specific logic
    return result

# Add in cli.py
new_mode_parser = subparsers.add_parser(
    'new-mode', 
    parents=[common_args],
    help='New mode description'
)
```

### Custom Configuration

You can customize settings through configuration files:

```json
{
  "api": {
    "openai_api_key": "your-key"
  },
  "generation": {
    "max_tokens": 3000,
    "temperature": 0.1
  },
  "workflow": {
    "max_iterations": 5
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   ```bash
   export OPENAI_API_KEY="your-actual-key"
   ```

2. **Missing Data Files**
   ```bash
   # Check if data files exist
   ls data/
   # Re-download data files
   wget https://zenodo.org/records/14328055/files/ground_truth_fitness_values.csv -P data/
   ```

3. **Environment Issues**
   ```bash
   # Recreate environment
   conda env remove -n ScienceBench_LLMEO
   conda env create -f environment.yml
   conda activate ScienceBench_LLMEO
   ```

### Debug Mode

Enable detailed logging:

```bash
python cli.py few-shot --debug
```

## 📚 Examples

### Quick Start

```bash
# 1. Set up environment
export OPENAI_API_KEY="your-key"
cd Projects/LLMEO

# 2. Run Few-shot mode
python cli.py few-shot

# 3. Try multi-property optimization
python cli.py multi-prop --iterations 3
```

### Advanced Usage

```bash
# Multi-property optimization with custom parameters
python cli.py multi-prop \
  --iterations 5 \
  --samples 20 \
  --num-samples 15 \
  --max-tokens 8000 \
  --temperature 0.2 \
  --seed 123
```

## 📄 License

This project is based on the original LLMEO project and follows the corresponding license.

## 🔗 Related Links

- Original code repository: [https://github.com/deepprinciple/llmeo](https://github.com/deepprinciple/llmeo)
- Reference project cli: [https://github.com/schwallergroup/steer](https://github.com/schwallergroup/steer)
    