# BioDiscoveryAgent Refactoring Guide

## Overview

BioDiscoveryAgent has been refactored to follow the modern structure used by llmeo and integrate with the sde-harness framework. This provides better modularity, testing capabilities, and experiment tracking.

## Refactoring Summary (Completed July 31, 2025)

### Major Changes Implemented

1. **Fixed Gene Scoring Display Issues**
   - Genes without measure scores (N/A) are now omitted from results
   - Changed gene listing format from indexed to simple line-by-line with scores
   - Added validation to only keep genes with valid scores in ground truth

2. **Simplified Gene Generation Logic**
   - Removed complex retry mechanism
   - Now generates requested number of genes in single attempt
   - Calculates hit rates based on valid genes only
   - Added duplicate removal within same generation step

3. **Enhanced Analysis Mode**
   - Rewrote `analyze.py` to read from `final_results.json` instead of `.npy` files
   - Added support for multiple directory structures
   - Enhanced output with gene scores and detailed statistics

4. **Fixed Tool Integration**
   - Fixed literature review tool by removing unsupported parameters
   - Verified all tools (Reactome, Critique, Literature review) working correctly
   - Moved tool files to organized structure in `src/tools/`

5. **Code Organization and Cleanup**
   - Archived deprecated files to `_archive/` folder:
     - `research_assistant.py`, `analyze.py` (original versions)
     - `test_refactor.py`, `ARCHIVE_SUMMARY.md`
     - `tools.py`, `baseline.py` (deprecated versions)
   - Created proper module structure with `__init__.py` files
   - Updated all imports to use new paths

6. **Comprehensive Test Suite**
   - Added 26 unit tests covering all major components
   - 100% pass rate after debugging
   - Tests cover: data loading, bio metrics, LLM interface, modes, CLI
   - All tests properly mock external dependencies

## New Structure

```text
BioDiscoveryAgent/
├── cli.py                      # Main entry point
├── src/                        # Source code
│   ├── __init__.py
│   ├── evaluators/             # Evaluation metrics
│   │   ├── __init__.py
│   │   └── bio_metrics.py     # Hit rate and other bio metrics
│   ├── modes/                  # Different discovery modes
│   │   ├── __init__.py
│   │   ├── perturb_genes.py   # Main gene perturbation mode
│   │   ├── baseline.py        # Baseline comparison mode
│   │   └── analyze.py         # Analysis mode (reads JSON results)
│   ├── tools/                  # Tool implementations
│   │   ├── __init__.py
│   │   ├── README.md          # Tool documentation
│   │   ├── get_lit_review.py  # Literature review (PubMed, bioRxiv)
│   │   ├── achilles.py        # Gene feature analysis
│   │   ├── gene.py            # Gene utilities, pathway analysis
│   │   └── LLM.py             # Legacy LLM wrapper
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── data_loader.py     # Dataset loading functions
│       ├── gene_utils.py      # Gene-specific utilities
│       ├── llm_interface.py   # LLM communication via sde-harness
│       ├── prompts.py         # All prompt templates
│       └── tools.py           # Tool integration wrapper
├── tests/                      # Unit tests (26 tests, 100% passing)
│   ├── __init__.py
│   ├── README.md              # Test documentation
│   ├── run_tests.py           # Test runner script
│   ├── test_bio_metrics.py    # Tests for evaluators
│   ├── test_data_loader.py    # Tests for data loading
│   ├── test_llm_interface.py  # Tests for LLM interface
│   └── test_modes.py          # Tests for different modes
├── datasets/                   # Data files
│   ├── task_prompts/          # Task descriptions (JSON)
│   │   ├── IFNG.json
│   │   ├── IL2.json
│   │   ├── Horlbeck.json
│   │   └── ...               # Other datasets
│   ├── ground_truth_*.csv     # Ground truth gene data
│   └── topmovers_*.npy        # Top performing genes
├── _archive/                   # Deprecated/original files
│   ├── research_assistant.py  # Original main script
│   ├── analyze.py            # Original analysis script
│   ├── tools.py              # Original tools module
│   ├── baseline.py           # Original baseline script
│   ├── test_refactor.py      # Refactoring test script
│   └── ARCHIVE_SUMMARY.md    # Archive documentation
├── CEGv2.txt                  # Essential genes list
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Pytest configuration
├── README.md                  # Main documentation
├── README_REFACTORING.md      # This refactoring guide
```

## Key Improvements

### 1. **Gene Validation and Scoring**
- Validates generated genes against ground truth dataset (~18,418 real genes)
- Filters out hallucinated/non-existent genes (e.g., IL38-IL98)
- Only displays genes with valid measure scores
- Improved hit rate calculation based on valid predictions

### 2. **Cleaner Output Format**
```
# Old format:
Previous Tested Genes:
1. GENE1 (score: 0.1234)
2. GENE2 (score: 0.5678)
...

# New format:
Previous Tested Genes:
GENE1: 0.1234
GENE2: 0.5678
...
```

### 3. **Better Error Handling**
- Graceful handling of missing genes in ground truth
- Clear logging of dropped/invalid genes
- Proper error messages for tool failures

### 4. **Enhanced Analysis Capabilities**
- Reads from JSON results for better compatibility
- Calculates comprehensive statistics
- Supports various directory structures
- Shows individual gene scores and progression

## Usage

### New CLI Usage

```bash
# Gene perturbation discovery (default "brief" variant)
python cli.py perturb-genes --model openai/gpt-4o-2024-08-06 --data-name IFNG --steps 5 --num-genes 128

# With specific task variant
python cli.py perturb-genes --task-variant brief-Horlbeck --data-name Horlbeck --steps 5 --num-genes 128

# With tools enabled
python cli.py perturb-genes --data-name IFNG --steps 5 --num-genes 128 \
    --lit-review --critique --reactome

# Analysis
python cli.py analyze --dataset IFNG --model gpt-4o-2024-08-06
```

### Display Limits
- Default: Shows up to 500 genes per round
- Adjustable via `max_display` parameter in code
- Full results always saved to files

### LLM Context Length
- Adjust with `--max-tokens` flag (default: 4000)
- Example: `--max-tokens 8000` for longer contexts

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m unittest discover tests -v

# Run specific test module
python -m unittest tests.test_bio_metrics -v

# Using pytest (if installed)
pytest tests/
```

Current test coverage: **26 tests, 100% passing**

## Migration from Original Code

### For Users
1. Use new CLI with cleaner argument names
2. Results now in `final_results.json` format
3. Gene validation ensures only real genes are tested
4. Tools work seamlessly with new structure

### For Developers
1. Import from new locations (e.g., `from src.utils.tools import BioDiscoveryTools`)
2. Use new data loader functions
3. Follow established patterns in test files
4. Check `_archive/` for original implementations

## Known Improvements

1. **Performance**: Faster gene validation using pandas indexing
2. **Accuracy**: Better hit rate calculation excluding invalid genes  
3. **Maintainability**: Modular structure with clear separation of concerns
4. **Reliability**: Comprehensive test coverage ensures stability
5. **Usability**: Cleaner output format and better error messages

## Future Considerations

1. Add more comprehensive integration tests
2. Implement caching for ground truth data loading
3. Add progress bars for long-running operations
4. Create API documentation
5. Set up continuous integration
