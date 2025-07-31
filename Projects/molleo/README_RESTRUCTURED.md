# MolLEO - Restructured with SDE Harness

This is the restructured version of MolLEO (Molecular Language-Enhanced Evolutionary Optimization) that fully integrates with the SDE Harness framework.

## Key Changes from Original

1. **SDE Harness Integration**
   - Uses `Generation` class for all LLM communications
   - Extends `Oracle` class for molecular property evaluation
   - Uses `Prompt` class for structured prompt management
   - Compatible with `Workflow` for complex pipelines

2. **Improved Architecture**
   - Modular structure with clear separation of concerns
   - Self-contained genetic algorithm operations in `/src/ga/`
   - Integration-ready Generation class for SDE Workflow
   - Separate modes for single and multi-objective optimization

3. **Enhanced Features**
   - Context-aware LLM mutations using population information
   - Support for multiple LLM providers via YAML configuration
   - Unified error handling and retry logic
   - Integration with Weave for experiment tracking
   - Self-contained GA operations (no external dependencies)

## Usage

### Command Line Interface

```bash
# Single objective optimization with LLM
python cli.py single --oracle qed --model openai/gpt-4o-2024-08-06 --generations 20

# Single objective optimization without LLM (random mutations only)
python cli.py single --oracle qed --model none --generations 20

# Multi-objective optimization
python cli.py multi --max-obj drd2 qed --min-obj sa --model openai/gpt-4o-2024-08-06

# Pareto optimization
python cli.py multi-pareto --max-obj gsk3b qed --min-obj sa --model openai/gpt-4o-2024-08-06
```

### Python API

```python
from src.core import MolLEOOptimizer
from src.oracles import TDCOracle

# Create oracle
oracle = TDCOracle("qed")

# Create optimizer
optimizer = MolLEOOptimizer(
    oracle=oracle,
    population_size=100,
    model_name="openai/gpt-4o-2024-08-06",
    use_llm_mutations=True
)

# Run optimization
results = optimizer.optimize(
    starting_smiles=["CCO", "CCN", "CCC"],
    num_generations=20
)
```

### Integration with SDE Workflow

```python
from sde_harness.core import Generation, Workflow, Oracle, Prompt
from src.core import MolLEOOptimizer
from src.oracles import TDCOracle
from src.generation import MolLEOGeneration

# Create oracle
oracle = TDCOracle("sa")

# Create optimizer
optimizer = MolLEOOptimizer(
    oracle=oracle,
    population_size=10,
    offspring_size=10,
    mutation_rate=0.05,
    model_name="openai/gpt-4o-2024-08-06",
    use_llm_mutations=True
)

# Create Generation instance for workflow
generation = MolLEOGeneration(
    model_name="openai/gpt-4o-2024-08-06"
)

# Create workflow
workflow = Workflow(
    generator=generation,
    oracle=oracle,
    max_iterations=2,
    enable_history_in_prompts=True
)

# Run workflow with custom prompts
workflow_results = workflow.run_sync(
    prompt=analog_generation_prompt,
    reference="Generate molecular analogs",
    gen_args={
        "model": "openai/gpt-4o-2024-08-06",
        "max_tokens": 500,
        "temperature": 0.8,
    }
)
```

## Examples

### Main Examples
- `example_usage.py` - Comprehensive examples including:
  - Single-objective optimization
  - Multi-objective optimization
  - Direct SDE harness integration with Workflow

### Test Suite (`/tests/`)
- `example_no_llm.py` - Example using only random mutations
- `example_with_llm.py` - Example with LLM mutations
- `example_comprehensive.py` - Comprehensive test scenarios

## Configuration

The system uses the standard SDE harness configuration in the **root directory** of `sde-harness`:
- `models.yaml` - Model configurations
- `credentials.yaml` - API credentials

## Project Structure

```
molleo/
├── src/                    # Main source code
│   ├── core/              # Core components
│   │   ├── molleo_optimizer.py  # Main optimizer class
│   │   └── prompts.py          # Prompt templates
│   ├── ga/                # Genetic algorithm operations
│   │   ├── mutations.py   # Molecular mutations
│   │   └── crossover.py   # Molecular crossover
│   ├── generation.py      # SDE harness Generation class
│   ├── modes/            # Optimization modes
│   │   ├── single_objective.py
│   │   └── multi_objective.py
│   ├── oracles/          # Property evaluation
│   │   ├── base.py       # Base oracle class
│   │   └── tdc_oracles.py # TDC oracle implementations
│   └── utils/            # Utilities
│       ├── evolutionary_ops.py
│       └── mol_utils.py
├── tests/                # Test suite
├── cli.py               # CLI interface
├── example_usage.py     # Usage examples
└── _archive/            # Old implementation (reference only)
```

## Key Improvements in Restructured Version

1. **Unified Model Parameter**: Replaced `--mol-lm` with `--model` for consistency
   - Use any model from `models.yaml`: `--model openai/gpt-4o-2024-08-06`
   - Disable LLM for random mutations: `--model none`

2. **Self-Contained GA Operations**: 
   - Genetic algorithm code moved to `/src/ga/`
   - No dependencies on old directory structure
   - Clean separation of concerns

3. **Better Error Handling**:
   - Graceful fallback when TDC data loading fails
   - Proper exception handling in mutations
   - Clear error messages for missing configurations

4. **Enhanced Testing**:
   - Comprehensive test suite in `/tests/`
   - Examples for both LLM and non-LLM modes
   - Oracle evaluation tests

## Dependencies

See `requirements.txt` for MolLEO-specific dependencies. The parent `requirements.txt` contains core SDE harness dependencies.

## Migration from Original

If you're migrating from the original MolLEO:
1. Update CLI commands to use `--model` instead of `--mol-lm`
2. Use full model names (e.g., `openai/gpt-4o-2024-08-06`) instead of shortcuts
3. All old code is preserved in `/_archive/` for reference
4. The new structure is fully backward-compatible with the same optimization algorithms