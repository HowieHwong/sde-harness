"""
Test configuration and fixtures for LLMEO tests

This file contains pytest fixtures and configuration for the test suite.
"""

import pytest
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import MagicMock

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)


@pytest.fixture
def sample_ligands_csv():
    """Sample ligands CSV content for testing"""
    return """SMILES,id,charge,connecting atom element,connecting atom index
c1ccccn1,RUCBEY-subgraph-1,0,N,1
CP(C)C,WECJIA-subgraph-3,0,P,1
N#CC,KEYRUB-subgraph-1,0,N,1
[C-]#[N+]c1c(C)cccc1C,NURKEQ-subgraph-2,0,C,1
O,MEBXUN-subgraph-1,0,O,1
n1c(cccc1C)C,BIFMOV-subgraph-1,0,N,1
CP(C)c1ccccc1,CUJYEL-subgraph-2,0,P,1
n1ccc(cc1)C,EZEXEM-subgraph-1,0,N,1"""


@pytest.fixture
def sample_fitness_csv():
    """Sample fitness CSV content for testing"""
    return """lig1,lig2,lig3,lig4,gap,polarisability
RUCBEY-subgraph-1,WECJIA-subgraph-3,KEYRUB-subgraph-1,NURKEQ-subgraph-2,3.2,450.5
MEBXUN-subgraph-1,BIFMOV-subgraph-1,CUJYEL-subgraph-2,EZEXEM-subgraph-1,2.8,380.2
KEYRUB-subgraph-1,NURKEQ-subgraph-2,MEBXUN-subgraph-1,BIFMOV-subgraph-1,3.5,435.1
WECJIA-subgraph-3,KEYRUB-subgraph-1,NURKEQ-subgraph-2,MEBXUN-subgraph-1,3.1,395.4
BIFMOV-subgraph-1,CUJYEL-subgraph-2,EZEXEM-subgraph-1,RUCBEY-subgraph-1,2.7,340.6"""


@pytest.fixture
def sample_tmc_dataframe():
    """Sample TMC DataFrame for testing"""
    return pd.DataFrame({
        'lig1': ['RUCBEY-subgraph-1', 'MEBXUN-subgraph-1', 'KEYRUB-subgraph-1'],
        'lig2': ['WECJIA-subgraph-3', 'BIFMOV-subgraph-1', 'NURKEQ-subgraph-2'],
        'lig3': ['KEYRUB-subgraph-1', 'CUJYEL-subgraph-2', 'MEBXUN-subgraph-1'],
        'lig4': ['NURKEQ-subgraph-2', 'EZEXEM-subgraph-1', 'BIFMOV-subgraph-1'],
        'gap': [3.2, 2.8, 3.5],
        'polarisability': [450.5, 380.2, 435.1]
    })


@pytest.fixture
def sample_ligand_charges():
    """Sample ligand charge dictionary for testing"""
    return {
        'RUCBEY-subgraph-1': 0,
        'WECJIA-subgraph-3': 0,
        'KEYRUB-subgraph-1': 0,
        'NURKEQ-subgraph-2': 0,
        'MEBXUN-subgraph-1': 0,
        'BIFMOV-subgraph-1': 0,
        'CUJYEL-subgraph-2': 0,
        'EZEXEM-subgraph-1': 0
    }


@pytest.fixture
def mock_args():
    """Mock arguments object for testing mode functions"""
    class MockArgs:
        def __init__(self):
            self.samples = 5
            self.num_samples = 3
            self.max_tokens = 1000
            self.iterations = 1
            self.seed = 42
            self.model = 'openai/gpt-4o-mini'
            self.temperature = 0.5
            self.mode = 'few-shot'
    
    return MockArgs()


@pytest.fixture
def temp_data_files(sample_ligands_csv, sample_fitness_csv):
    """Create temporary data files for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create data directory
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir)
        
        # Write ligands file
        ligands_path = os.path.join(data_dir, '1M-space_50-ligands-full.csv')
        with open(ligands_path, 'w') as f:
            f.write(sample_ligands_csv)
        
        # Write fitness file
        fitness_path = os.path.join(data_dir, 'ground_truth_fitness_values.csv')
        with open(fitness_path, 'w') as f:
            f.write(sample_fitness_csv)
        
        yield {
            'temp_dir': temp_dir,
            'data_dir': data_dir,
            'ligands_path': ligands_path,
            'fitness_path': fitness_path
        }


@pytest.fixture
def mock_generation():
    """Mock Generation object for testing"""
    mock_gen = MagicMock()
    mock_gen.generate.return_value = MagicMock(
        message=MagicMock(content="Mock LLM response with TMC data")
    )
    return mock_gen


@pytest.fixture
def mock_workflow():
    """Mock Workflow object for testing"""
    mock_workflow = MagicMock()
    mock_workflow.run_sync.return_value = {
        'final_scores': {'top10_avg_gap': 3.0},
        'iteration_scores': [{'top10_avg_gap': 3.0}],
        'outputs': ["Mock LLM output"]
    }
    return mock_workflow


@pytest.fixture
def mock_oracle():
    """Mock Oracle object for testing"""
    mock_oracle = MagicMock()
    mock_oracle.list_metrics.return_value = ['top10_avg_gap']
    mock_oracle.list_multi_round_metrics.return_value = ['top10_avg_gap']
    return mock_oracle


@pytest.fixture(scope="session")
def test_models_config():
    """Test models configuration for testing"""
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
        }
    }


@pytest.fixture(scope="session")
def test_credentials_config():
    """Test credentials configuration for testing"""
    return {
        'openai': {
            'api_key': 'test-api-key'
        },
        'test': {
            'api_key': 'test-api-key'
        }
    }


# Configure test settings
def pytest_configure(config):
    """Configure pytest settings"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Mark integration tests
        if "test_cli" in item.nodeid or "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "workflow" in item.name.lower() or "generation" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark unit tests (default for most tests)
        if not any(mark.name in ['integration', 'slow'] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)