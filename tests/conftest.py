"""
Test configuration and fixtures for SDE-Harness tests

This file contains pytest fixtures and configuration for the test suite.
"""

import pytest
import tempfile
import os
import sys
import yaml
from unittest.mock import MagicMock
from typing import Dict, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture
def sample_models_config():
    """Sample models configuration for testing"""
    return {
        'openai/gpt-4o-mini': {
            'provider': 'openai',
            'model': 'gpt-4o-mini',
            'credentials': 'openai'
        },
        'test/mock-model': {
            'provider': 'test',
            'model': 'mock-model',
            'credentials': 'test'
        },
        'local/test-model': {
            'provider': 'local',
            'model': 'test/model',
            'credentials': None
        }
    }


@pytest.fixture
def sample_credentials_config():
    """Sample credentials configuration for testing"""
    return {
        'openai': {
            'api_key': 'test-api-key-openai'
        },
        'test': {
            'api_key': 'test-api-key-mock'
        }
    }


@pytest.fixture
def temp_config_files(sample_models_config, sample_credentials_config):
    """Create temporary configuration files for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create models.yaml
        models_path = os.path.join(temp_dir, 'models.yaml')
        with open(models_path, 'w') as f:
            yaml.dump(sample_models_config, f)
        
        # Create credentials.yaml
        credentials_path = os.path.join(temp_dir, 'credentials.yaml')
        with open(credentials_path, 'w') as f:
            yaml.dump(sample_credentials_config, f)
        
        yield {
            'temp_dir': temp_dir,
            'models_path': models_path,
            'credentials_path': credentials_path
        }


@pytest.fixture
def mock_generation_response():
    """Mock response from Generation.generate()"""
    mock_response = MagicMock()
    mock_response.message.content = "Mock generated response from LLM"
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mock generated response from LLM"
    return mock_response


@pytest.fixture
def sample_prompt_templates():
    """Sample prompt templates for testing"""
    return {
        "test_template": "Hello {name}, please {task}.",
        "qa_template": "Question: {question}\nAnswer:",
        "complex_template": """
Task: {task}
Context: {context}
Examples:
{examples}
Please provide: {output_format}
        """.strip()
    }


@pytest.fixture
def sample_oracle_metrics():
    """Sample metrics functions for Oracle testing"""
    def accuracy_metric(prediction: Any, reference: Any, **kwargs) -> float:
        """Mock accuracy metric"""
        return 0.85
    
    def bleu_metric(prediction: Any, reference: Any, **kwargs) -> float:
        """Mock BLEU metric"""
        return 0.72
    
    def multi_round_metric(history: Dict, reference: Any, current_iteration: int, **kwargs) -> float:
        """Mock multi-round metric"""
        return current_iteration * 0.1
    
    return {
        'accuracy': accuracy_metric,
        'bleu': bleu_metric,
        'multi_round': multi_round_metric
    }


@pytest.fixture
def sample_workflow_data():
    """Sample data for workflow testing"""
    return {
        'reference_data': [
            {'input': 'test input 1', 'expected': 'test output 1'},
            {'input': 'test input 2', 'expected': 'test output 2'}
        ],
        'generation_args': {
            'model_name': 'test/mock-model',
            'temperature': 0.7,
            'max_tokens': 100
        },
        'prompt_vars': {
            'task': 'test task',
            'context': 'test context',
            'examples': 'example 1\nexample 2'
        }
    }


@pytest.fixture
def mock_weave():
    """Mock weave operations for testing"""
    mock_weave = MagicMock()
    mock_weave.init = MagicMock()
    mock_weave.op = MagicMock()
    
    # Mock the decorator behavior
    def mock_op_decorator(func):
        return func
    
    mock_weave.op.return_value = mock_op_decorator
    return mock_weave


@pytest.fixture(autouse=True)
def mock_weave_imports(monkeypatch, mock_weave):
    """Automatically mock weave imports in all tests"""
    monkeypatch.setattr('weave.init', mock_weave.init)
    monkeypatch.setattr('weave.op', mock_weave.op)


@pytest.fixture
def mock_litellm():
    """Mock litellm for testing"""
    mock_litellm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mock LLM response"
    mock_response.usage.total_tokens = 100
    mock_litellm.completion.return_value = mock_response
    return mock_litellm


@pytest.fixture
def mock_transformers():
    """Mock transformers for testing"""
    mock_transformers = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_pipeline = MagicMock()
    
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
    mock_transformers.pipeline.return_value = mock_pipeline
    
    return mock_transformers


# Configure test settings
def pytest_configure(config):
    """Configure pytest settings"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (slower, may require external resources)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "requires_api: Tests that require actual API keys"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Mark integration tests
        if "integration" in item.name.lower() or "workflow" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "slow" in item.name.lower() or "generation" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark API-dependent tests
        if "api" in item.name.lower() or "litellm" in item.name.lower():
            item.add_marker(pytest.mark.requires_api)
        
        # Mark unit tests (default for most tests)
        if not any(mark.name in ['integration', 'slow', 'requires_api'] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)