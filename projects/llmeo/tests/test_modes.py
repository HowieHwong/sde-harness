"""
Unit tests for LLMEO mode functions (few_shot.py, single_prop.py, multi_prop.py)
"""

import unittest
import sys
import os
import argparse
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

from src.modes.few_shot import run_few_shot
from src.modes.single_prop import run_single_prop
from src.modes.multi_prop import run_multi_prop


class TestModeFunctions(unittest.TestCase):
    """Test mode functions"""

    def setUp(self):
        """Set up test arguments and mock data"""
        # Create mock arguments object
        self.mock_args = argparse.Namespace(
            samples=5,
            num_samples=3,
            max_tokens=1000,
            iterations=1,
            seed=42,
            model='openai/gpt-4o-mini',
            temperature=0.5
        )
        
        # Sample CSV data
        self.sample_ligands_csv = """SMILES,id,charge,connecting atom element,connecting atom index
c1ccccn1,RUCBEY-subgraph-1,0,N,1
CP(C)C,WECJIA-subgraph-3,0,P,1"""
        
        # Sample DataFrame with enough rows for sampling
        self.sample_df = pd.DataFrame({
            'lig1': ['RUCBEY-subgraph-1', 'WECJIA-subgraph-3', 'RUCBEY-subgraph-1', 'WECJIA-subgraph-3', 'RUCBEY-subgraph-1', 'WECJIA-subgraph-3'],
            'lig2': ['WECJIA-subgraph-3', 'RUCBEY-subgraph-1', 'WECJIA-subgraph-3', 'RUCBEY-subgraph-1', 'WECJIA-subgraph-3', 'RUCBEY-subgraph-1'],
            'lig3': ['RUCBEY-subgraph-1', 'WECJIA-subgraph-3', 'RUCBEY-subgraph-1', 'WECJIA-subgraph-3', 'RUCBEY-subgraph-1', 'WECJIA-subgraph-3'],
            'lig4': ['WECJIA-subgraph-3', 'RUCBEY-subgraph-1', 'WECJIA-subgraph-3', 'RUCBEY-subgraph-1', 'WECJIA-subgraph-3', 'RUCBEY-subgraph-1'],
            'gap': [3.2, 2.8, 3.5, 2.9, 3.1, 2.7],
            'polarisability': [450.5, 380.2, 460.1, 390.5, 440.8, 370.3]
        })

    @patch('src.modes.few_shot.weave')
    @patch('src.modes.few_shot.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.modes.few_shot.Workflow')
    @patch('src.modes.few_shot.Generation')
    def test_run_few_shot_basic_execution(self, mock_generation, mock_workflow, 
                                         mock_file_open, mock_read_csv, mock_weave):
        """Test basic execution of run_few_shot"""
        # Setup mocks
        mock_file_open.return_value.read.return_value = self.sample_ligands_csv
        
        # Create ligands DataFrame with id and charge columns
        ligands_df = pd.DataFrame({
            'id': ['RUCBEY-subgraph-1', 'WECJIA-subgraph-3'],
            'charge': [0, 0]
        })
        
        # Mock read_csv to return different DataFrames based on filename
        def side_effect(filename):
            if 'ground_truth_fitness_values.csv' in filename:
                return self.sample_df
            elif '1M-space_50-ligands-full.csv' in filename:
                return ligands_df
            return self.sample_df
        
        mock_read_csv.side_effect = side_effect
        
        # Mock workflow result
        mock_workflow_instance = MagicMock()
        mock_workflow_instance.run_sync.return_value = {
            'final_scores': {'top10_avg_gap': 3.0}
        }
        mock_workflow.return_value = mock_workflow_instance
        
        # Mock generation
        mock_generation_instance = MagicMock()
        mock_generation.return_value = mock_generation_instance
        
        # Run the function
        result = run_few_shot(self.mock_args)
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('final_scores', result)
        mock_weave.init.assert_called_with("few_shot_mode")

    @patch('src.modes.single_prop.weave')
    @patch('src.modes.single_prop.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.modes.single_prop.Workflow')
    @patch('src.modes.single_prop.Generation')
    def test_run_single_prop_basic_execution(self, mock_generation, mock_workflow,
                                            mock_file_open, mock_read_csv, mock_weave):
        """Test basic execution of run_single_prop"""
        # Setup mocks
        mock_file_open.return_value.read.return_value = self.sample_ligands_csv
        
        # Create ligands DataFrame with id and charge columns
        ligands_df = pd.DataFrame({
            'id': ['RUCBEY-subgraph-1', 'WECJIA-subgraph-3'],
            'charge': [0, 0]
        })
        
        # Mock read_csv to return different DataFrames based on filename
        def side_effect(filename):
            if 'ground_truth_fitness_values.csv' in filename:
                return self.sample_df
            elif '1M-space_50-ligands-full.csv' in filename:
                return ligands_df
            return self.sample_df
        
        mock_read_csv.side_effect = side_effect
        
        # Mock workflow result
        mock_workflow_instance = MagicMock()
        mock_workflow_instance.run_sync.return_value = {
            'final_scores': {'top10_avg_gap': 3.5}
        }
        mock_workflow.return_value = mock_workflow_instance
        
        # Mock generation
        mock_generation_instance = MagicMock()
        mock_generation.return_value = mock_generation_instance
        
        # Run the function
        result = run_single_prop(self.mock_args)
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('final_scores', result)
        mock_weave.init.assert_called_with("LLMEO-single-prop")

    @patch('src.modes.multi_prop.weave')
    @patch('src.modes.multi_prop.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.modes.multi_prop.Workflow')
    @patch('src.modes.multi_prop.Generation')
    def test_run_multi_prop_basic_execution(self, mock_generation, mock_workflow,
                                           mock_file_open, mock_read_csv, mock_weave):
        """Test basic execution of run_multi_prop"""
        # Setup mocks
        mock_file_open.return_value.read.return_value = self.sample_ligands_csv
        
        # Create ligands DataFrame with id and charge columns
        ligands_df = pd.DataFrame({
            'id': ['RUCBEY-subgraph-1', 'WECJIA-subgraph-3'],
            'charge': [0, 0]
        })
        
        # Mock read_csv to return different DataFrames based on filename
        def side_effect(filename):
            if 'ground_truth_fitness_values.csv' in filename:
                return self.sample_df
            elif '1M-space_50-ligands-full.csv' in filename:
                return ligands_df
            return self.sample_df
        
        mock_read_csv.side_effect = side_effect
        
        # Mock workflow result
        mock_workflow_instance = MagicMock()
        mock_workflow_instance.run_sync.return_value = {
            'final_scores': {'top10_avg_score': 1500.0}
        }
        mock_workflow.return_value = mock_workflow_instance
        
        # Mock generation
        mock_generation_instance = MagicMock()
        mock_generation.return_value = mock_generation_instance
        
        # Run the function
        result = run_multi_prop(self.mock_args)
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('final_scores', result)
        mock_weave.init.assert_called_with("LLMEO-multi_prop_mode")

    @patch('src.modes.few_shot.weave')
    @patch('src.modes.few_shot.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.modes.few_shot.Generation')
    def test_few_shot_generation_setup(self, mock_generation, mock_file_open, 
                                      mock_read_csv, mock_weave):
        """Test that few_shot properly sets up Generation with correct parameters"""
        # Setup mocks
        mock_file_open.return_value.read.return_value = self.sample_ligands_csv
        
        # Create ligands DataFrame with id and charge columns
        ligands_df = pd.DataFrame({
            'id': ['RUCBEY-subgraph-1', 'WECJIA-subgraph-3'],
            'charge': [0, 0]
        })
        
        # Mock read_csv to return different DataFrames based on filename
        def side_effect(filename):
            if 'ground_truth_fitness_values.csv' in filename:
                return self.sample_df
            elif '1M-space_50-ligands-full.csv' in filename:
                return ligands_df
            return self.sample_df
        
        mock_read_csv.side_effect = side_effect
        
        mock_generation_instance = MagicMock()
        mock_generation.return_value = mock_generation_instance
        
        # Mock workflow to avoid actual execution
        with patch('src.modes.few_shot.Workflow') as mock_workflow:
            mock_workflow_instance = MagicMock()
            mock_workflow_instance.run_sync.return_value = {'final_scores': {}}
            mock_workflow.return_value = mock_workflow_instance
            
            run_few_shot(self.mock_args)
        
        # Check that Generation was called with correct parameters
        mock_generation.assert_called_once()
        call_args = mock_generation.call_args
        self.assertIn('models_file', call_args.kwargs)
        self.assertIn('credentials_file', call_args.kwargs)

    @patch('src.modes.single_prop.weave')
    @patch('src.modes.single_prop.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)  
    @patch('src.modes.single_prop.Generation')
    def test_single_prop_multi_round_setup(self, mock_generation, mock_file_open,
                                          mock_read_csv, mock_weave):
        """Test that single_prop sets up multi-round metrics correctly"""
        # Setup mocks
        mock_file_open.return_value.read.return_value = self.sample_ligands_csv
        
        # Create ligands DataFrame with id and charge columns
        ligands_df = pd.DataFrame({
            'id': ['RUCBEY-subgraph-1', 'WECJIA-subgraph-3'],
            'charge': [0, 0]
        })
        
        # Mock read_csv to return different DataFrames based on filename
        def side_effect(filename):
            if 'ground_truth_fitness_values.csv' in filename:
                return self.sample_df
            elif '1M-space_50-ligands-full.csv' in filename:
                return ligands_df
            return self.sample_df
        
        mock_read_csv.side_effect = side_effect
        
        mock_generation_instance = MagicMock()
        mock_generation.return_value = mock_generation_instance
        
        # Mock workflow to check configuration
        with patch('src.modes.single_prop.Workflow') as mock_workflow:
            mock_workflow_instance = MagicMock()
            mock_workflow_instance.run_sync.return_value = {'final_scores': {}}
            mock_workflow.return_value = mock_workflow_instance
            
            run_single_prop(self.mock_args)
        
        # Check that Workflow was called with enable_multi_round_metrics=True
        mock_workflow.assert_called_once()
        call_kwargs = mock_workflow.call_args.kwargs
        self.assertTrue(call_kwargs.get('enable_multi_round_metrics'))

    def test_args_parameter_usage(self):
        """Test that functions properly use args parameters"""
        test_args = argparse.Namespace(
            samples=10,
            num_samples=7,
            max_tokens=2000,
            iterations=3,
            seed=123,
            model='test-model',
            temperature=0.8
        )
        
        # We can't easily test the full execution without mocking everything,
        # but we can test that the args are structured correctly
        self.assertEqual(test_args.samples, 10)
        self.assertEqual(test_args.num_samples, 7)
        self.assertEqual(test_args.max_tokens, 2000)
        self.assertEqual(test_args.iterations, 3)
        self.assertEqual(test_args.seed, 123)
        self.assertEqual(test_args.model, 'test-model')
        self.assertEqual(test_args.temperature, 0.8)

    @patch('src.modes.few_shot.weave')
    @patch('src.modes.few_shot.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.modes.few_shot.Workflow')
    @patch('src.modes.few_shot.Generation')
    def test_few_shot_data_sampling(self, mock_generation, mock_workflow,
                                   mock_file_open, mock_read_csv, mock_weave):
        """Test that few_shot properly samples data based on args"""
        # Setup mocks
        mock_file_open.return_value.read.return_value = self.sample_ligands_csv
        
        # Create a DataFrame with more rows to test sampling
        large_tmc_df = pd.DataFrame({
            'lig1': ['RUCBEY-subgraph-1'] * 10,
            'lig2': ['WECJIA-subgraph-3'] * 10,
            'lig3': ['RUCBEY-subgraph-1'] * 10,
            'lig4': ['WECJIA-subgraph-3'] * 10,
            'gap': [3.2] * 10,
            'polarisability': [450.5] * 10
        })
        
        # Create ligands DataFrame with id and charge columns
        ligands_df = pd.DataFrame({
            'id': ['RUCBEY-subgraph-1', 'WECJIA-subgraph-3'],
            'charge': [0, 0]
        })
        
        # Mock read_csv to return different DataFrames based on filename
        def side_effect(filename):
            if 'ground_truth_fitness_values.csv' in filename:
                return large_tmc_df
            elif '1M-space_50-ligands-full.csv' in filename:
                return ligands_df
            return large_tmc_df
        
        mock_read_csv.side_effect = side_effect
        
        # Mock the sample method to verify it's called with correct parameters
        mock_sample = MagicMock(return_value=self.sample_df)
        large_tmc_df.sample = mock_sample
        
        # Mock workflow
        mock_workflow_instance = MagicMock()
        mock_workflow_instance.run_sync.return_value = {'final_scores': {}}
        mock_workflow.return_value = mock_workflow_instance
        
        mock_generation_instance = MagicMock()
        mock_generation.return_value = mock_generation_instance
        
        run_few_shot(self.mock_args)
        
        # Check that sample was called with correct parameters
        mock_sample.assert_called_once_with(n=self.mock_args.samples, 
                                           random_state=self.mock_args.seed)

    @patch('src.modes.multi_prop.weave')
    @patch('src.modes.multi_prop.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.modes.multi_prop.Workflow')
    @patch('src.modes.multi_prop.Generation')
    def test_multi_prop_prompt_template(self, mock_generation, mock_workflow,
                                       mock_file_open, mock_read_csv, mock_weave):
        """Test that multi_prop uses the correct prompt template"""
        # Setup mocks
        mock_file_open.return_value.read.return_value = self.sample_ligands_csv
        
        # Create ligands DataFrame with id and charge columns
        ligands_df = pd.DataFrame({
            'id': ['RUCBEY-subgraph-1', 'WECJIA-subgraph-3'],
            'charge': [0, 0]
        })
        
        # Mock read_csv to return different DataFrames based on filename
        def side_effect(filename):
            if 'ground_truth_fitness_values.csv' in filename:
                return self.sample_df
            elif '1M-space_50-ligands-full.csv' in filename:
                return ligands_df
            return self.sample_df
        
        mock_read_csv.side_effect = side_effect
        
        mock_generation_instance = MagicMock()
        mock_generation.return_value = mock_generation_instance
        
        # Mock Prompt to capture the template used
        with patch('src.modes.multi_prop.Prompt') as mock_prompt:
            mock_prompt_instance = MagicMock()
            mock_prompt.return_value = mock_prompt_instance
            
            # Mock workflow
            mock_workflow_instance = MagicMock()
            mock_workflow_instance.run_sync.return_value = {'final_scores': {}}
            mock_workflow.return_value = mock_workflow_instance
            
            run_multi_prop(self.mock_args)
            
            # Check that Prompt was called with PROMPT_MB template
            mock_prompt.assert_called_once()
            call_args = mock_prompt.call_args
            # The template should contain multi-property specific content
            template = call_args.kwargs.get('custom_template', '')
            self.assertIn('polarisability', template)


class TestModeIntegration(unittest.TestCase):
    """Integration tests for mode functions"""
    
    def test_mode_imports(self):
        """Test that all mode functions can be imported correctly"""
        from src.modes import run_few_shot, run_single_prop, run_multi_prop
        
        # Check that functions are callable
        self.assertTrue(callable(run_few_shot))
        self.assertTrue(callable(run_single_prop))
        self.assertTrue(callable(run_multi_prop))

    def test_mode_function_signatures(self):
        """Test that mode functions have expected signatures"""
        import inspect
        
        from src.modes.few_shot import run_few_shot
        from src.modes.single_prop import run_single_prop
        from src.modes.multi_prop import run_multi_prop
        
        # All functions should take a single 'args' parameter
        for func in [run_few_shot, run_single_prop, run_multi_prop]:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            self.assertEqual(len(params), 1)
            self.assertEqual(params[0], 'args')


if __name__ == '__main__':
    unittest.main()