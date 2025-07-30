"""
Unit tests for sde_harness.base modules
"""

import unittest
import sys
import os
import argparse
from unittest.mock import patch, MagicMock
from abc import ABC
import pytest

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sde_harness.base import ProjectBase, CLIBase, EvaluatorBase


class ConcreteProject(ProjectBase):
    """Concrete implementation of ProjectBase for testing"""
    
    def _setup_project(self, **kwargs):
        """Setup test project"""
        self.test_config = kwargs.get('test_config', 'default')
        self.setup_called = True
    
    def run(self, **kwargs):
        """Run test project"""
        return {
            'status': 'completed',
            'config': self.test_config,
            'args': kwargs
        }


class ConcreteCLI(CLIBase):
    """Concrete implementation of CLIBase for testing"""
    
    def _add_project_arguments(self, parser):
        """Add project-specific arguments"""
        parser.add_argument('--test-param', default='default_value', help='Test parameter')
        parser.add_argument('--iterations', type=int, default=3, help='Number of iterations')
    
    def run_command(self, args):
        """Run the project with parsed arguments"""
        return {
            'project_name': self.project_name,
            'test_param': args.test_param,
            'iterations': args.iterations,
            'models_file': args.models_file
        }


class ConcreteEvaluator(EvaluatorBase):
    """Concrete implementation of EvaluatorBase for testing"""
    
    def setup_metrics(self):
        """Setup project-specific metrics"""
        self.metrics = ['accuracy', 'score']
    
    def evaluate(self, prediction, reference, **kwargs):
        """Simple evaluation implementation"""
        if prediction == reference:
            return {'accuracy': 1.0, 'score': 100}
        else:
            return {'accuracy': 0.0, 'score': 0}
    
    def get_metric_names(self):
        """Get available metric names"""
        return ['accuracy', 'score']


class TestProjectBase(unittest.TestCase):
    """Test ProjectBase abstract class"""

    @patch('sde_harness.base.project_base.Generation')
    @patch('sde_harness.base.project_base.Oracle')
    def test_project_base_init_default(self, mock_oracle, mock_generation):
        """Test ProjectBase initialization with default parameters"""
        mock_generation_instance = MagicMock()
        mock_oracle_instance = MagicMock()
        mock_generation.return_value = mock_generation_instance
        mock_oracle.return_value = mock_oracle_instance
        
        project = ConcreteProject()
        
        # Verify initialization
        self.assertEqual(project.generator, mock_generation_instance)
        self.assertEqual(project.oracle, mock_oracle_instance)
        self.assertIsNone(project.workflow)
        self.assertTrue(project.setup_called)
        self.assertEqual(project.test_config, 'default')
        
        # Verify Generation was called with default parameters
        mock_generation.assert_called_once_with(
            models_file="config/models.yaml",
            credentials_file="config/credentials.yaml"
        )

    @patch('sde_harness.base.project_base.Generation')
    @patch('sde_harness.base.project_base.Oracle')
    def test_project_base_init_custom(self, mock_oracle, mock_generation):
        """Test ProjectBase initialization with custom parameters"""
        mock_generation_instance = MagicMock()
        mock_oracle_instance = MagicMock()
        mock_generation.return_value = mock_generation_instance
        mock_oracle.return_value = mock_oracle_instance
        
        project = ConcreteProject(
            models_file="custom_models.yaml",
            credentials_file="custom_creds.yaml",
            test_config="custom_config"
        )
        
        # Verify custom configuration
        self.assertEqual(project.test_config, 'custom_config')
        
        # Verify Generation was called with custom parameters
        mock_generation.assert_called_once_with(
            models_file="custom_models.yaml",
            credentials_file="custom_creds.yaml"
        )

    @patch('sde_harness.base.project_base.Generation')
    @patch('sde_harness.base.project_base.Oracle')
    def test_project_base_run(self, mock_oracle, mock_generation):
        """Test ProjectBase run method"""
        mock_generation.return_value = MagicMock()
        mock_oracle.return_value = MagicMock()
        
        project = ConcreteProject()
        
        result = project.run(param1='value1', param2='value2')
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['config'], 'default')
        self.assertIn('args', result)
        self.assertEqual(result['args']['param1'], 'value1')
        self.assertEqual(result['args']['param2'], 'value2')

    def test_project_base_abstract_methods(self):
        """Test that ProjectBase cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            ProjectBase()  # Should fail because abstract methods are not implemented


class TestCLIBase(unittest.TestCase):
    """Test CLIBase abstract class"""

    def test_cli_base_init(self):
        """Test CLIBase initialization"""
        cli = ConcreteCLI("TestProject")
        
        self.assertEqual(cli.project_name, "TestProject")
        self.assertIsInstance(cli.parser, argparse.ArgumentParser)

    def test_cli_base_parser_creation(self):
        """Test parser creation with default arguments"""
        cli = ConcreteCLI("TestProject")
        
        # Test that parser has expected arguments
        args = cli.parser.parse_args([])
        
        self.assertEqual(args.models_file, "config/models.yaml")
        self.assertEqual(args.credentials_file, "config/credentials.yaml")
        self.assertEqual(args.output_dir, "outputs")
        self.assertFalse(args.verbose)

    def test_cli_base_custom_arguments(self):
        """Test CLI with custom project arguments"""
        cli = ConcreteCLI("TestProject")
        
        # Test parsing with custom arguments
        args = cli.parser.parse_args(['--test-param', 'custom_value', '--iterations', '5'])
        
        self.assertEqual(args.test_param, 'custom_value')
        self.assertEqual(args.iterations, 5)

    def test_cli_base_main_method(self):
        """Test CLI main method execution"""
        cli = ConcreteCLI("TestProject")
        
        # Mock sys.argv
        test_args = ['script_name', '--test-param', 'test_value', '--verbose']
        
        with patch('sys.argv', test_args):
            result = cli.main()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['project_name'], 'TestProject')
        self.assertEqual(result['test_param'], 'test_value')

    def test_cli_base_help_functionality(self):
        """Test CLI help functionality"""
        cli = ConcreteCLI("TestProject")
        
        # Test that help can be generated without errors
        help_string = cli.parser.format_help()
        
        self.assertIn("TestProject", help_string)
        self.assertIn("--models-file", help_string)
        self.assertIn("--test-param", help_string)

    def test_cli_base_error_handling(self):
        """Test CLI error handling for invalid arguments"""
        cli = ConcreteCLI("TestProject")
        
        # Test invalid argument type
        with self.assertRaises(SystemExit):
            cli.parser.parse_args(['--iterations', 'not_a_number'])

    def test_cli_base_abstract_methods(self):
        """Test that CLIBase cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            CLIBase("TestProject")  # Should fail because abstract methods are not implemented


class TestEvaluatorBase(unittest.TestCase):
    """Test EvaluatorBase abstract class"""

    def test_evaluator_base_init(self):
        """Test EvaluatorBase initialization"""
        evaluator = ConcreteEvaluator()
        evaluator.setup_metrics()
        
        self.assertIsInstance(evaluator, EvaluatorBase)

    def test_evaluator_base_evaluate(self):
        """Test evaluate method implementation"""
        evaluator = ConcreteEvaluator()
        evaluator.setup_metrics()
        
        # Test perfect match
        result1 = evaluator.evaluate("test", "test")
        self.assertEqual(result1['accuracy'], 1.0)
        self.assertEqual(result1['score'], 100)
        
        # Test mismatch
        result2 = evaluator.evaluate("test1", "test2")
        self.assertEqual(result2['accuracy'], 0.0)
        self.assertEqual(result2['score'], 0)

    def test_evaluator_base_get_metric_names(self):
        """Test get_metric_names method"""
        evaluator = ConcreteEvaluator()
        evaluator.setup_metrics()
        
        metrics = evaluator.get_metric_names()
        
        self.assertIsInstance(metrics, list)
        self.assertIn('accuracy', metrics)
        self.assertIn('score', metrics)

    def test_evaluator_base_evaluate_with_kwargs(self):
        """Test evaluate method with additional kwargs"""
        evaluator = ConcreteEvaluator()
        evaluator.setup_metrics()
        
        # Test that kwargs are passed through (even if not used in this implementation)
        result = evaluator.evaluate("test", "test", custom_param="value")
        
        self.assertEqual(result['accuracy'], 1.0)

    def test_evaluator_base_abstract_methods(self):
        """Test that EvaluatorBase cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            EvaluatorBase()  # Should fail because abstract methods are not implemented


class TestBaseClassIntegration(unittest.TestCase):
    """Integration tests for base classes working together"""

    @patch('sde_harness.base.project_base.Generation')
    @patch('sde_harness.base.project_base.Oracle')
    def test_project_cli_integration(self, mock_oracle, mock_generation):
        """Test Project and CLI working together"""
        mock_generation.return_value = MagicMock()
        mock_oracle.return_value = MagicMock()
        
        # Create CLI
        cli = ConcreteCLI("IntegrationTest")
        
        # Parse arguments
        args = cli.parser.parse_args(['--test-param', 'integration_value'])
        
        # Create project with CLI arguments
        project = ConcreteProject(
            models_file=args.models_file,
            credentials_file=args.credentials_file,
            test_config=args.test_param
        )
        
        # Run project
        result = project.run()
        
        self.assertEqual(result['config'], 'integration_value')
        self.assertEqual(result['status'], 'completed')

    @patch('sde_harness.base.project_base.Generation')
    @patch('sde_harness.base.project_base.Oracle')
    def test_project_evaluator_integration(self, mock_oracle, mock_generation):
        """Test Project and Evaluator working together"""
        mock_generation.return_value = MagicMock()
        mock_oracle.return_value = MagicMock()
        
        # Create project and evaluator
        project = ConcreteProject()
        evaluator = ConcreteEvaluator()
        evaluator.setup_metrics()
        
        # Simulate project generating predictions
        project_result = project.run()
        
        # Use evaluator to assess results
        evaluation = evaluator.evaluate("prediction", "reference")
        
        self.assertIn('status', project_result)
        self.assertIn('accuracy', evaluation)

    def test_multiple_inheritance_compatibility(self):
        """Test that base classes can be used with multiple inheritance"""
        
        class MultipleInheritanceProject(ConcreteProject, ConcreteEvaluator):
            """Test class inheriting from both ProjectBase and EvaluatorBase"""
            pass
        
        with patch('sde_harness.base.project_base.Generation'):
            with patch('sde_harness.base.project_base.Oracle'):
                combined = MultipleInheritanceProject()
                
                # Should have both project and evaluator functionality
                project_result = combined.run()
                eval_result = combined.evaluate("test", "test")
                
                self.assertIn('status', project_result)
                self.assertIn('accuracy', eval_result)


@pytest.mark.integration
class TestBaseClassesAdvanced(unittest.TestCase):
    """Advanced integration tests for base classes"""

    @patch('sde_harness.base.project_base.Generation')
    @patch('sde_harness.base.project_base.Oracle')
    def test_complex_project_workflow(self, mock_oracle, mock_generation):
        """Test complex project workflow scenario"""
        # Setup mocks
        mock_generation_instance = MagicMock()
        mock_oracle_instance = MagicMock()
        mock_generation.return_value = mock_generation_instance
        mock_oracle.return_value = mock_oracle_instance
        
        # Create a more complex project implementation
        class ComplexProject(ProjectBase):
            def _setup_project(self, **kwargs):
                self.experiment_name = kwargs.get('experiment_name', 'default_experiment')
                self.iterations = kwargs.get('iterations', 1)
                self.metrics = ['accuracy', 'precision', 'recall']
                
                # Setup oracle with metrics
                for metric in self.metrics:
                    self.oracle.register_metric(metric, lambda p, r: 0.5)
            
            def run(self, **kwargs):
                results = []
                for i in range(self.iterations):
                    # Simulate generation
                    prediction = f"prediction_{i}"
                    # Simulate evaluation
                    scores = self.oracle.list_metrics()
                    results.append({
                        'iteration': i,
                        'prediction': prediction,
                        'metrics': scores
                    })
                
                return {
                    'experiment': self.experiment_name,
                    'iterations': len(results),
                    'results': results
                }
        
        # Test complex project
        project = ComplexProject(
            experiment_name="advanced_test",
            iterations=3
        )
        
        result = project.run()
        
        self.assertEqual(result['experiment'], 'advanced_test')
        self.assertEqual(result['iterations'], 3)
        self.assertEqual(len(result['results']), 3)

    def test_real_world_cli_scenario(self):
        """Test realistic CLI usage scenario"""
        class RealWorldCLI(CLIBase):
            def _add_project_arguments(self, parser):
                parser.add_argument('--dataset', required=True, help='Dataset path')
                parser.add_argument('--model', default='gpt-3.5-turbo', help='Model name')
                parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
                parser.add_argument('--max-iterations', type=int, default=10, help='Max iterations')
            
            def run_command(self, args):
                return {
                    'dataset': args.dataset,
                    'model': args.model,
                    'batch_size': args.batch_size,
                    'max_iterations': args.max_iterations,
                    'config_valid': True
                }
        
        cli = RealWorldCLI("ScientificDiscovery")
        
        # Test realistic command line arguments
        args = cli.parser.parse_args([
            '--dataset', '/path/to/dataset.csv',
            '--model', 'claude-3-sonnet',
            '--batch-size', '64',
            '--max-iterations', '20',
            '--verbose'
        ])
        
        result = cli.run_command(args)
        
        self.assertEqual(result['dataset'], '/path/to/dataset.csv')
        self.assertEqual(result['model'], 'claude-3-sonnet')
        self.assertEqual(result['batch_size'], 64)
        self.assertEqual(result['max_iterations'], 20)
        self.assertTrue(result['config_valid'])


if __name__ == '__main__':
    unittest.main()