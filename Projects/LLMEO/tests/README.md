# LLMEO Test Suite

This directory contains comprehensive unit tests and integration tests for the LLMEO project.

## 🧪 Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and fixtures
├── run_tests.py             # Custom test runner script
├── README.md                # This file
├── test_utils.py            # Tests for utility functions (_utils.py)
├── test_data_loader.py      # Tests for data loading functions
├── test_prompts.py          # Tests for prompt templates
├── test_modes.py            # Tests for mode functions (few_shot, single_prop, multi_prop)
└── test_cli.py              # Integration tests for CLI functionality
```

## 🚀 Running Tests

### Using the Custom Test Runner

The easiest way to run tests is using the provided test runner:

```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests with verbose output
python tests/run_tests.py -t unit -v

# Run integration tests with coverage
python tests/run_tests.py -t integration -c

# Run slow tests
python tests/run_tests.py -t slow -v
```

### Using pytest directly

If you have pytest installed:

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run only unit tests
python -m pytest tests/ -m unit

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Using unittest

For basic testing without pytest:

```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py"

# Run specific test file
python -m unittest tests.test_utils -v

# Run specific test
python -m unittest tests.test_utils.TestUtilsFunctions.test_make_text_for_existing_tmcs_single_property -v
```

### Legacy Test Support

The original `test.py` file is still available for backward compatibility:

```bash
python test.py
```

## 📋 Test Categories

### Unit Tests (`-m unit`)
Fast, isolated tests that test individual functions and components:
- **test_utils.py**: Tests for `_utils.py` utility functions
- **test_data_loader.py**: Tests for data loading and validation functions
- **test_prompts.py**: Tests for prompt templates and their integration

### Integration Tests (`-m integration`)
Tests that verify component interaction and CLI functionality:
- **test_cli.py**: Tests for command-line interface and argument parsing
- **test_modes.py**: Integration tests for mode functions

### Slow Tests (`-m slow`)
Tests that may take longer to run or require external resources:
- Workflow execution tests
- Model generation tests (when not mocked)

## 🔧 Test Configuration

### pytest.ini
Configuration file for pytest with:
- Test discovery settings
- Marker definitions
- Output formatting
- Coverage options

### conftest.py
Contains shared fixtures and test configuration:
- `sample_ligands_csv`: Mock ligands data
- `sample_fitness_csv`: Mock fitness data
- `sample_tmc_dataframe`: Mock TMC DataFrame
- `mock_args`: Mock command line arguments
- `temp_data_files`: Temporary test data files

## 📊 Test Coverage

The test suite covers:

### Utility Functions (`src/utils/_utils.py`)
- ✅ `make_text_for_existing_tmcs()` - Text formatting for TMCs
- ✅ `retrive_tmc_from_message()` - TMC extraction from LLM responses
- ✅ `find_tmc_in_space()` - TMC matching with rotational symmetry

### Data Loading (`src/utils/data_loader.py`)
- ✅ `load_data()` - Data file loading and processing
- ✅ `setup_generator()` - Generation instance setup
- ✅ `create_prompt()` - Prompt template creation
- ✅ `setup_oracle()` - Oracle configuration
- ✅ `validate_data_files()` - Data file validation

### Prompt Templates (`src/utils/prompt.py`)
- ✅ `PROMPT_G` - Single property prompt template
- ✅ `PROMPT_MB` - Multi-property prompt template
- ✅ Template variable substitution
- ✅ Output format validation

### Mode Functions (`src/modes/`)
- ✅ `run_few_shot()` - Few-shot learning mode
- ✅ `run_single_prop()` - Single property optimization
- ✅ `run_multi_prop()` - Multi-property optimization
- ✅ Function signatures and imports

### CLI Interface (`cli.py`)
- ✅ Argument parsing for all modes
- ✅ Command routing and execution
- ✅ Error handling (KeyboardInterrupt, exceptions)
- ✅ Data validation integration

## 🛠️ Writing New Tests

### Test File Structure
```python
import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from src.module_to_test import function_to_test

class TestModuleName(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        pass
    
    def test_function_name(self):
        """Test description"""
        # Test implementation
        pass
```

### Fixtures Usage (with pytest)
```python
def test_with_fixtures(sample_tmc_dataframe, mock_args):
    # Use provided fixtures
    result = some_function(sample_tmc_dataframe, mock_args)
    assert result is not None
```

### Mocking External Dependencies
```python
@patch('module.external_dependency')
def test_with_mock(self, mock_dependency):
    mock_dependency.return_value = "mocked_result"
    # Test implementation
```

## 🐛 Common Issues

### Import Errors
- Ensure project root is in Python path
- Check relative import paths
- Verify module names match file structure

### Mock Issues
- Mock external dependencies like file I/O, network calls
- Use appropriate mock return values
- Remember to mock at the right level (import location)

### Fixture Problems
- Check fixture names match function parameters
- Ensure conftest.py is in the right location
- Verify fixture scope (function, class, module, session)

## 📈 Continuous Integration

For CI/CD integration:

```bash
# Run tests with coverage and XML output
python -m pytest tests/ --cov=src --cov-report=xml --junitxml=test-results.xml

# Run only fast tests in CI
python -m pytest tests/ -m "unit and not slow"
```

## 🤝 Contributing

When adding new functionality:

1. Write tests for new functions/classes
2. Update existing tests if behavior changes
3. Add appropriate test markers (`@pytest.mark.unit`, etc.)
4. Include docstrings for test methods
5. Mock external dependencies appropriately

## 📝 Notes

- Tests use mock data to avoid dependencies on actual API keys or large datasets
- Some integration tests may be skipped in environments without certain dependencies
- The test suite is designed to run quickly for development workflow
- For comprehensive testing including actual API calls, use a dedicated test environment