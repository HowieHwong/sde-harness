# BioDiscoveryAgent Tests

This directory contains unit tests for the BioDiscoveryAgent project.

## Running Tests

### Using unittest (built-in):
```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_data_loader

# Run with verbose output
python tests/run_tests.py
```

### Using pytest (if installed):
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_bio_metrics.py

# Run with coverage
pytest --cov=src tests/
```

## Test Structure

- `test_data_loader.py` - Tests for data loading functionality
- `test_bio_metrics.py` - Tests for biological metrics and evaluation
- `test_modes.py` - Tests for different modes (perturb_genes, baseline, analyze)
- `test_llm_interface.py` - Tests for LLM interface and API calls

## Writing New Tests

1. Create a new file starting with `test_` in the tests directory
2. Import unittest and the modules you want to test
3. Create test classes inheriting from `unittest.TestCase`
4. Write test methods starting with `test_`

Example:
```python
import unittest
from src.module import function_to_test

class TestMyModule(unittest.TestCase):
    def test_function(self):
        result = function_to_test(input_data)
        self.assertEqual(result, expected_output)
```

## Mock Data

Tests use mock data and patch external dependencies to avoid:
- API calls to LLM services
- File system operations
- Network requests
- Database connections

This ensures tests run quickly and consistently.