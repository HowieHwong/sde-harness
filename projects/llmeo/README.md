# LLMEO - LLM-based Evolutionary Optimization

## Original Code Repository
[https://github.com/deepprinciple/llmeo](https://github.com/deepprinciple/llmeo)



## 📦 Install

Run these commands from the harness root unless a step says otherwise.

### 1. Enter the LLMEO project folder

```bash
cd projects/llmeo
```

### 2. Download required dataset

```bash
wget https://zenodo.org/records/14328055/files/ground_truth_fitness_values.csv -P data/
```

### 3. Set up the conda environment

```bash
conda env create -f environment.yml
conda activate ScienceBench_LLMEO
```

### 4. Configure model files in the harness root

```bash
cd ../..

cat > models.yaml <<'EOF'
deepseek/deepseek-v4-flash:
  provider: deepseek
  model: deepseek-v4-flash
  credentials: deepseek
  __call_args:
    thinking:
      type: disabled
    allowed_openai_params:
      - thinking

deepseek/deepseek-chat:
  provider: deepseek
  model: deepseek-chat
  credentials: deepseek
EOF

cat > credentials.yaml <<'EOF'
deepseek:
  api_key: ${DEEPSEEK_API_KEY}
EOF

export DEEPSEEK_API_KEY="your-api-key-here"
cd projects/llmeo
```

### 5. Optional legacy test file

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

- `--samples`: Initial sample number (default: 10)
- `--num-samples`: Generated sample number (default: 10)
- `--max-tokens`: Maximum token number (default: 8000)
- `--iterations`: Iteration number (default: 2)
- `--model`: Model name from the harness root `models.yaml` (default: `deepseek/deepseek-v4-flash`)
- `--temperature`: Temperature parameter (default: 1.0)
- `--seed`: Random seed (default: 42)

### Verified Smoke Test

After completing the setup above, run a minimal one-iteration check:

```bash
python cli.py few-shot \
  --iterations 1 \
  --samples 1 \
  --num-samples 1 \
  --temperature 0
```

The command should validate the data files, generate a response, evaluate `top10_avg_gap`, and print a final score in the terminal. A score of `0` is possible for this tiny smoke test and does not indicate a runtime failure.

### Output

The current CLI prints evaluation results to stdout. The final line has the form:

```text
✅ Few-Shot mode completed! Final score: {'top10_avg_gap': ...}
```

## 🏗️ Project Structure

```
projects/llmeo/
├── cli.py              # Command line entry point
├── src/                # Source code directory
│   ├── modes/          # Running mode modules
│   │   ├── __init__.py
│   │   ├── few_shot.py     # Few-shot learning mode
│   │   ├── single_prop.py  # Single property optimization mode
│   │   └── multi_prop.py   # Multi-property optimization mode
│   ├── weave_utils.py  # Weave logging helpers
│   ├── utils/          # Utility modules
│   │   ├── __init__.py
│   │   ├── data_loader.py  # Data loading utilities
│   │   ├── prompt.py       # Prompt templates
│   │   └── _utils.py       # Utility functions
├── data/               # Data files
├── tests/              # Test files
│   ├── __init__.py
│   ├── conftest.py
│   ├── run_tests.py
│   ├── test_cli.py
│   ├── test_data_loader.py
│   ├── test_modes.py
│   ├── test_prompts.py
│   └── test_utils.py
├── test.py             # Test file
├── pytest.ini          # Pytest configuration
├── environment.yml     # Environment configuration
└── README.md           # This document
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   ```bash
   export DEEPSEEK_API_KEY="your-actual-key"
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

4. **Weave Login / Logging Issues**

   LLMEO disables Weave network logging by default so local checks do not require a Weights & Biases account or network access. To enable Weave logging for a real run:
   ```bash
   LLMEO_ENABLE_WEAVE=true python cli.py few-shot --iterations 1 --samples 1 --num-samples 1
   ```

5. **Legacy Model Alias**

   `deepseek/deepseek-chat` is kept in the example config for old runs, but the default is `deepseek/deepseek-v4-flash`.
   ```bash
   python cli.py few-shot --model deepseek/deepseek-v4-flash
   ```

6. **DeepSeek V4-Flash Empty Output**

   DeepSeek V4-Flash enables thinking mode by default. For this project, keep the `thinking: disabled` and `allowed_openai_params` entries shown above in `models.yaml`; otherwise the model may spend the full token budget on reasoning and return empty visible content.

## 📚 Examples

### Quick Start

```bash
# 1. Set up environment
export DEEPSEEK_API_KEY="your-key"
cd projects/llmeo

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
