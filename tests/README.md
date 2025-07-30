# SDE-Harness Test Suite

This directory contains comprehensive unit tests and integration tests for the SDE-Harness framework.

## 🧪 Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Pytest configuration and fixtures
├── run_tests.py                   # Custom test runner script
├── README.md                      # This file
├── test_core_generation.py        # Tests for Generation class
├── test_core_oracle.py            # Tests for Oracle class
├── test_core_prompt.py            # Tests for Prompt class
├── test_core_workflow.py          # Tests for Workflow class
└── test_base_classes.py           # Tests for base classes (ProjectBase, CLIBase, EvaluatorBase)
```

## 🚀 Running Tests

### Using the Custom Test Runner (Recommended)

The easiest way to run tests is using the provided test runner:

```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests with verbose output
python tests/run_tests.py -t unit -v

# Run integration tests with coverage
python tests/run_tests.py -t integration -c

# Run specific module tests
python tests/run_tests.py -m generation
python tests/run_tests.py -m oracle
python tests/run_tests.py -m workflow

# Run tests matching a pattern
python tests/run_tests.py -p "test_oracle"

# Check test dependencies
python tests/run_tests.py --check-deps
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
python -m pytest tests/ --cov=sde_harness --cov-report=html
```

### Using unittest

For basic testing without pytest:

```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py"

# Run specific test file
python -m unittest tests.test_core_generation -v

# Run specific test class
python -m unittest tests.test_core_generation.TestGeneration -v
```

## 📋 Test Categories

### Unit Tests (`-m unit`)
Fast, isolated tests that test individual functions and components:
- **test_core_generation.py**: Tests for `Generation` class and related functions
- **test_core_oracle.py**: Tests for `Oracle` class and metric management
- **test_core_prompt.py**: Tests for `Prompt` class and template handling
- **test_core_workflow.py**: Tests for `Workflow` class orchestration
- **test_base_classes.py**: Tests for abstract base classes

### Integration Tests (`-m integration`)
Tests that verify component interaction:
- Cross-module functionality testing
- End-to-end workflow scenarios
- Real-world usage patterns

### Slow Tests (`-m slow`)
Tests that may take longer to run:
- Complex workflow scenarios
- Large data processing tests
- Multi-iteration simulations

### API-Dependent Tests (`-m requires_api`)
Tests that require actual API keys (skipped by default):
- Live model generation tests
- External service integration tests

## 🔧 Test Configuration

### pytest.ini
Configuration file for pytest with:
- Test discovery settings
- Marker definitions
- Output formatting
- Timeout settings
- Warning filters

### conftest.py
Contains shared fixtures and test configuration:
- `sample_models_config`: Mock models configuration
- `sample_credentials_config`: Mock credentials
- `temp_config_files`: Temporary configuration files
- `mock_generation_response`: Mock LLM responses
- `mock_weave`: Mock weave operations
- Various other fixtures for testing components

## 📊 Test Coverage

The test suite provides comprehensive coverage for:

### Core Modules (`sde_harness.core`)

#### Generation (`sde_harness.core.generation`)
- ✅ Configuration loading (`load_models_and_credentials`, `load_model_config`)
- ✅ Generation class initialization and setup
- ✅ LiteLLM integration and error handling
- ✅ Local model support (transformers)
- ✅ Async generation methods
- ✅ Model listing and information retrieval

#### Oracle (`sde_harness.core.oracle`)
- ✅ Metric registration and management
- ✅ Single-round evaluation
- ✅ Multi-round evaluation with history
- ✅ Metric unregistration and listing
- ✅ Error handling in metric execution
- ✅ Complex metric implementations

#### Prompt (`sde_harness.core.prompt`)
- ✅ Built-in template system
- ✅ Custom template support
- ✅ Variable substitution and management
- ✅ Template building with default and override variables
- ✅ History integration for iterative workflows
- ✅ Edge cases and error handling

#### Workflow (`sde_harness.core.workflow`)
- ✅ Workflow initialization and configuration
- ✅ Sync and async execution
- ✅ Multi-iteration processing
- ✅ Stop criteria implementation
- ✅ History tracking and management
- ✅ Multi-round metrics integration
- ✅ Dynamic prompt functions
- ✅ Error handling and recovery

### Base Classes (`sde_harness.base`)

#### ProjectBase
- ✅ Abstract class interface
- ✅ Component initialization (Generation, Oracle)
- ✅ Project setup and configuration
- ✅ Integration with other components

#### CLIBase
- ✅ Argument parser creation
- ✅ Common CLI arguments
- ✅ Project-specific argument extension
- ✅ Help system and error handling

#### EvaluatorBase
- ✅ Abstract evaluation interface
- ✅ Metric name management
- ✅ Evaluation method implementation

## 🛠️ Writing New Tests

### Test File Structure
```python
import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import pytest

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sde_harness.core.module_to_test import ClassToTest

class TestClassName(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def test_method_name(self):
        """Test description"""
        # Test implementation
        pass
```

### Using Fixtures (with pytest)
```python
def test_with_fixtures(sample_models_config, mock_generation_response):
    # Use provided fixtures
    result = some_function(sample_models_config)
    assert result is not None
```

### Mocking External Dependencies
```python
@patch('sde_harness.core.generation.litellm.completion')
def test_with_mock(self, mock_completion):
    mock_completion.return_value = MagicMock()
    # Test implementation
```

### Test Markers
```python
@pytest.mark.unit
def test_fast_unit_test():
    pass

@pytest.mark.integration  
def test_integration_scenario():
    pass

@pytest.mark.slow
def test_complex_workflow():
    pass

@pytest.mark.requires_api
def test_with_real_api():
    pass
```

## 🐛 Common Issues and Solutions

### Import Errors
- Ensure project root is in Python path
- Check relative import paths
- Verify module names match file structure

### Mock Issues
- Mock external dependencies like API calls, file I/O
- Use appropriate mock return values
- Mock at the correct level (import location)

### Fixture Problems
- Check fixture names match function parameters
- Ensure conftest.py is accessible
- Verify fixture scope (function, class, module, session)

### Async Test Issues
- Use `@pytest.mark.asyncio` for async tests
- Mock async functions with `AsyncMock`
- Handle event loops properly

## 📈 Continuous Integration

For CI/CD integration:

```bash
# Run tests with coverage and XML output
python -m pytest tests/ --cov=sde_harness --cov-report=xml --junitxml=test-results.xml

# Run only fast tests in CI
python -m pytest tests/ -m "unit and not slow"

# Skip API-dependent tests
python -m pytest tests/ -m "not requires_api"
```

## 🔍 Test Examples

### Testing Generation Class
```python
@patch('sde_harness.core.generation.litellm.completion')
def test_generation_success(self, mock_completion):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Generated text"
    mock_completion.return_value = mock_response
    
    gen = Generation()
    result = gen.generate("Test prompt", model_name="test/model")
    
    assert result is not None
    mock_completion.assert_called_once()
```

### Testing Oracle Metrics
```python
def test_oracle_metric_registration(self):
    oracle = Oracle()
    
    def accuracy_metric(prediction, reference, **kwargs):
        return 0.85
    
    oracle.register_metric('accuracy', accuracy_metric)
    
    result = oracle.evaluate("pred", "ref", metrics=['accuracy'])
    assert result['accuracy'] == 0.85
```

### Testing Workflow Integration
```python
@patch('sde_harness.core.workflow.weave')
async def test_workflow_execution(self, mock_weave):
    mock_generator = MagicMock()
    mock_oracle = MagicMock()
    mock_weave.op.return_value = lambda func: func
    
    workflow = Workflow(mock_generator, mock_oracle, max_iterations=2)
    
    result = await workflow.run(
        prompt=mock_prompt,
        reference="test_reference"
    )
    
    assert 'final_scores' in result
    assert len(result['iteration_scores']) == 2
```

## 🤝 Contributing

When adding new functionality:

1. Write tests for new functions/classes
2. Update existing tests if behavior changes
3. Add appropriate test markers
4. Include docstrings for test methods
5. Mock external dependencies appropriately
6. Add integration tests for complex workflows

## 📝 Dependencies

### Required
- `unittest` (Python standard library)
- `yaml` - For configuration file handling
- `pytest` - Advanced test runner (recommended)

### Optional
- `pytest-cov` - Coverage reporting
- `pytest-timeout` - Test timeout handling
- `pytest-mock` - Enhanced mocking capabilities
- `pytest-asyncio` - Async test support

Install all test dependencies:
```bash
pip install pytest pytest-cov pytest-timeout pytest-mock pytest-asyncio
```

## 📞 Support

For test-related issues:
1. Check the test output for specific error messages
2. Verify all dependencies are installed
3. Ensure configuration files are properly set up
4. Check the GitHub issues for known problems

The test suite is designed to be robust and informative, helping developers maintain code quality and catch regressions early in the development process.