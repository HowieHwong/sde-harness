"""
Unit tests for sde_harness.core.workflow module
"""

import unittest
import sys
import os
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sde_harness.core.workflow import Workflow, remove_inputs_from_weave_log


class TestRemoveInputsFromWeaveLog(unittest.TestCase):
    """Test utility function for weave log cleaning"""

    def test_remove_inputs_basic(self):
        """Test basic input removal"""
        inputs = {
            'reference': 'some_reference_data',
            'prompt': 'test_prompt',
            'other': 'other_data'
        }
        
        result = remove_inputs_from_weave_log(inputs)
        
        self.assertEqual(result['reference'], "[Omitted from logging]")
        self.assertEqual(result['prompt'], 'test_prompt')
        self.assertEqual(result['other'], 'other_data')

    def test_remove_inputs_no_reference(self):
        """Test when no reference key exists"""
        inputs = {
            'prompt': 'test_prompt',
            'other': 'other_data'
        }
        
        result = remove_inputs_from_weave_log(inputs)
        
        self.assertEqual(result, inputs)  # Should be unchanged

    def test_remove_inputs_empty_dict(self):
        """Test with empty input dictionary"""
        inputs = {}
        
        result = remove_inputs_from_weave_log(inputs)
        
        self.assertEqual(result, {})


class TestWorkflow(unittest.TestCase):
    """Test Workflow class functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_generator = MagicMock()
        self.mock_oracle = MagicMock()
        
        # Setup mock oracle methods
        self.mock_oracle.list_metrics.return_value = ['accuracy', 'bleu']
        self.mock_oracle.list_multi_round_metrics.return_value = ['improvement']
        self.mock_oracle.evaluate.return_value = {'accuracy': 0.85, 'bleu': 0.72}
        self.mock_oracle.evaluate_multi_round.return_value = {'improvement': 0.1}

    def test_workflow_init_basic(self):
        """Test basic Workflow initialization"""
        workflow = Workflow(
            generator=self.mock_generator,
            oracle=self.mock_oracle
        )
        
        self.assertEqual(workflow.generator, self.mock_generator)
        self.assertEqual(workflow.oracle, self.mock_oracle)
        self.assertEqual(workflow.max_iterations, 3)  # default
        self.assertTrue(workflow.enable_history_in_prompts)  # default
        self.assertTrue(workflow.enable_multi_round_metrics)  # default

    def test_workflow_init_custom_params(self):
        """Test Workflow initialization with custom parameters"""
        def custom_stop_criteria(results):
            return results.get('accuracy', 0) > 0.9
        
        workflow = Workflow(
            generator=self.mock_generator,
            oracle=self.mock_oracle,
            max_iterations=5,
            stop_criteria=custom_stop_criteria,
            enable_history_in_prompts=False,
            enable_multi_round_metrics=False
        )
        
        self.assertEqual(workflow.max_iterations, 5)
        self.assertEqual(workflow.stop_criteria, custom_stop_criteria)
        self.assertFalse(workflow.enable_history_in_prompts)
        self.assertFalse(workflow.enable_multi_round_metrics)

    @patch('sde_harness.core.workflow.asyncio.run')
    def test_run_sync_basic(self, mock_asyncio_run):
        """Test run_sync method"""
        mock_result = {'final_scores': {'accuracy': 0.85}}
        mock_asyncio_run.return_value = mock_result
        
        workflow = Workflow(self.mock_generator, self.mock_oracle)
        mock_prompt = MagicMock()
        
        result = workflow.run_sync(
            prompt=mock_prompt,
            reference="test_reference"
        )
        
        self.assertEqual(result, mock_result)
        mock_asyncio_run.assert_called_once()

    @patch('sde_harness.core.workflow.weave')
    async def test_run_async_basic(self, mock_weave):
        """Test basic async run method"""
        # Setup mocks
        self.mock_generator.generate_async = AsyncMock(return_value="Generated output")
        mock_prompt = MagicMock()
        mock_prompt.build.return_value = "Built prompt"
        
        # Mock weave decorator
        mock_weave.op.return_value = lambda func: func
        
        workflow = Workflow(self.mock_generator, self.mock_oracle, max_iterations=1)
        
        result = await workflow.run(
            prompt=mock_prompt,
            reference="test_reference"
        )
        
        # Check that generator was called
        self.mock_generator.generate_async.assert_called()
        self.assertIsInstance(result, dict)
        self.assertIn('final_scores', result)

    @patch('sde_harness.core.workflow.weave')
    async def test_run_with_callable_prompt(self, mock_weave):
        """Test run with callable prompt function"""
        self.mock_generator.generate_async = AsyncMock(return_value="Generated output")
        mock_weave.op.return_value = lambda func: func
        
        def prompt_function(iteration, history):
            mock_prompt = MagicMock()
            mock_prompt.build.return_value = f"Prompt for iteration {iteration}"
            return mock_prompt
        
        workflow = Workflow(self.mock_generator, self.mock_oracle, max_iterations=1)
        
        result = await workflow.run(
            prompt=prompt_function,
            reference="test_reference"
        )
        
        self.assertIn('final_scores', result)

    @patch('sde_harness.core.workflow.weave')
    async def test_run_multiple_iterations(self, mock_weave):
        """Test run with multiple iterations"""
        self.mock_generator.generate_async = AsyncMock(return_value="Generated output")
        mock_prompt = MagicMock()
        mock_prompt.build.return_value = "Built prompt"
        mock_weave.op.return_value = lambda func: func
        
        workflow = Workflow(self.mock_generator, self.mock_oracle, max_iterations=3)
        
        result = await workflow.run(
            prompt=mock_prompt,
            reference="test_reference"
        )
        
        # Check that generator was called multiple times
        self.assertEqual(self.mock_generator.generate_async.call_count, 3)
        self.assertIn('iteration_scores', result)
        self.assertEqual(len(result['iteration_scores']), 3)

    @patch('sde_harness.core.workflow.weave')
    async def test_run_with_stop_criteria(self, mock_weave):
        """Test run with custom stop criteria"""
        self.mock_generator.generate_async = AsyncMock(return_value="Generated output")
        mock_prompt = MagicMock()
        mock_prompt.build.return_value = "Built prompt"
        mock_weave.op.return_value = lambda func: func
        
        # Stop criteria that stops after first iteration
        def stop_criteria(results):
            return True
        
        workflow = Workflow(
            self.mock_generator, 
            self.mock_oracle, 
            max_iterations=5,
            stop_criteria=stop_criteria
        )
        
        result = await workflow.run(
            prompt=mock_prompt,
            reference="test_reference"
        )
        
        # Should stop after 1 iteration due to stop criteria
        self.assertEqual(self.mock_generator.generate_async.call_count, 1)
        self.assertEqual(len(result['iteration_scores']), 1)

    @patch('sde_harness.core.workflow.weave')
    async def test_run_with_multi_round_metrics(self, mock_weave):
        """Test run with multi-round metrics enabled"""
        self.mock_generator.generate_async = AsyncMock(return_value="Generated output")
        mock_prompt = MagicMock()
        mock_prompt.build.return_value = "Built prompt"
        mock_weave.op.return_value = lambda func: func
        
        workflow = Workflow(
            self.mock_generator, 
            self.mock_oracle, 
            max_iterations=2,
            enable_multi_round_metrics=True
        )
        
        result = await workflow.run(
            prompt=mock_prompt,
            reference="test_reference"
        )
        
        # Should call multi-round evaluation
        self.mock_oracle.evaluate_multi_round.assert_called()

    @patch('sde_harness.core.workflow.weave')
    async def test_run_without_multi_round_metrics(self, mock_weave):
        """Test run with multi-round metrics disabled"""
        self.mock_generator.generate_async = AsyncMock(return_value="Generated output")
        mock_prompt = MagicMock()
        mock_prompt.build.return_value = "Built prompt"
        mock_weave.op.return_value = lambda func: func
        
        workflow = Workflow(
            self.mock_generator, 
            self.mock_oracle, 
            max_iterations=2,
            enable_multi_round_metrics=False
        )
        
        result = await workflow.run(
            prompt=mock_prompt,
            reference="test_reference"
        )
        
        # Should call regular evaluation
        self.mock_oracle.evaluate.assert_called()
        self.mock_oracle.evaluate_multi_round.assert_not_called()

    @patch('sde_harness.core.workflow.weave')
    async def test_run_with_generation_args(self, mock_weave):
        """Test run with generation arguments"""
        self.mock_generator.generate_async = AsyncMock(return_value="Generated output")
        mock_prompt = MagicMock()
        mock_prompt.build.return_value = "Built prompt"
        mock_weave.op.return_value = lambda func: func
        
        workflow = Workflow(self.mock_generator, self.mock_oracle, max_iterations=1)
        
        gen_args = {
            'model_name': 'test/model',
            'temperature': 0.7,
            'max_tokens': 100
        }
        
        result = await workflow.run(
            prompt=mock_prompt,
            reference="test_reference",
            gen_args=gen_args
        )
        
        # Check that generator was called with correct arguments
        call_args = self.mock_generator.generate_async.call_args
        for key, value in gen_args.items():
            self.assertIn(key, call_args.kwargs)
            self.assertEqual(call_args.kwargs[key], value)

    @patch('sde_harness.core.workflow.weave')
    async def test_run_history_tracking(self, mock_weave):
        """Test that run properly tracks history"""
        outputs = ["Output 1", "Output 2", "Output 3"]
        self.mock_generator.generate_async = AsyncMock(side_effect=outputs)
        
        mock_prompt = MagicMock()
        mock_prompt.build.return_value = "Built prompt"
        mock_weave.op.return_value = lambda func: func
        
        workflow = Workflow(self.mock_generator, self.mock_oracle, max_iterations=3)
        
        result = await workflow.run(
            prompt=mock_prompt,
            reference="test_reference"
        )
        
        # Check history tracking
        self.assertIn('outputs', result)
        self.assertEqual(len(result['outputs']), 3)
        self.assertEqual(result['outputs'], outputs)

    @patch('sde_harness.core.workflow.weave')
    async def test_run_error_handling(self, mock_weave):
        """Test error handling in run method"""
        self.mock_generator.generate_async = AsyncMock(side_effect=Exception("Generation failed"))
        mock_prompt = MagicMock()
        mock_prompt.build.return_value = "Built prompt"
        mock_weave.op.return_value = lambda func: func
        
        workflow = Workflow(self.mock_generator, self.mock_oracle, max_iterations=1)
        
        with self.assertRaises(Exception) as context:
            await workflow.run(
                prompt=mock_prompt,
                reference="test_reference"
            )
        
        self.assertIn("Generation failed", str(context.exception))

    def test_workflow_with_invalid_generator(self):
        """Test workflow initialization with invalid generator"""
        with self.assertRaises(TypeError):
            Workflow(generator=None, oracle=self.mock_oracle)

    def test_workflow_with_invalid_oracle(self):
        """Test workflow initialization with invalid oracle"""
        with self.assertRaises(TypeError):
            Workflow(generator=self.mock_generator, oracle=None)


class TestWorkflowAdvanced(unittest.TestCase):
    """Advanced tests for Workflow functionality"""

    def setUp(self):
        """Set up advanced test fixtures"""
        self.mock_generator = MagicMock()
        self.mock_oracle = MagicMock()
        
        # Setup more detailed mock responses
        self.mock_oracle.list_metrics.return_value = ['accuracy', 'f1_score', 'bleu']
        self.mock_oracle.list_multi_round_metrics.return_value = ['improvement_rate', 'consistency']
        
        # Progressive scores for testing multi-round behavior
        self.mock_oracle.evaluate.side_effect = [
            {'accuracy': 0.6, 'f1_score': 0.65, 'bleu': 0.7},
            {'accuracy': 0.75, 'f1_score': 0.8, 'bleu': 0.75},
            {'accuracy': 0.85, 'f1_score': 0.9, 'bleu': 0.8}
        ]
        
        self.mock_oracle.evaluate_multi_round.side_effect = [
            {'improvement_rate': 0.1, 'consistency': 0.8},
            {'improvement_rate': 0.15, 'consistency': 0.85},
            {'improvement_rate': 0.2, 'consistency': 0.9}
        ]

    @patch('sde_harness.core.workflow.weave')
    async def test_complex_workflow_scenario(self, mock_weave):
        """Test complex workflow with realistic scenario"""
        # Simulate improving outputs over iterations
        outputs = [
            "Initial attempt with basic approach",
            "Improved approach with better logic",
            "Final optimized solution"
        ]
        self.mock_generator.generate_async = AsyncMock(side_effect=outputs)
        mock_weave.op.return_value = lambda func: func
        
        # Complex prompt function that uses history
        def adaptive_prompt(iteration, history):
            mock_prompt = MagicMock()
            if iteration == 1:
                mock_prompt.build.return_value = "Initial prompt"
            else:
                prev_output = history['outputs'][-1] if history['outputs'] else ""
                mock_prompt.build.return_value = f"Improve upon: {prev_output[:20]}..."
            return mock_prompt
        
        # Stop criteria based on improvement
        def improvement_stop_criteria(results):
            if 'accuracy' in results and results['accuracy'] > 0.8:
                return True
            return False
        
        workflow = Workflow(
            generator=self.mock_generator,
            oracle=self.mock_oracle,
            max_iterations=3,
            stop_criteria=improvement_stop_criteria,
            enable_history_in_prompts=True,
            enable_multi_round_metrics=True
        )
        
        result = await workflow.run(
            prompt=adaptive_prompt,
            reference="reference_data",
            gen_args={'temperature': 0.7, 'max_tokens': 200}
        )
        
        # Verify complex workflow behavior
        self.assertIn('final_scores', result)
        self.assertIn('iteration_scores', result)
        self.assertIn('outputs', result)
        
        # Should stop early due to stop criteria (accuracy > 0.8 in iteration 2)
        self.assertLessEqual(len(result['outputs']), 3)
        
        # Verify multi-round metrics were used
        self.mock_oracle.evaluate_multi_round.assert_called()

    @patch('sde_harness.core.workflow.weave')
    async def test_workflow_with_dynamic_prompts(self, mock_weave):
        """Test workflow with prompts that change based on performance"""
        self.mock_generator.generate_async = AsyncMock(return_value="Generated output")
        mock_weave.op.return_value = lambda func: func
        
        prompt_calls = []
        
        def performance_based_prompt(iteration, history):
            prompt_calls.append((iteration, len(history.get('outputs', []))))
            
            mock_prompt = MagicMock()
            
            # Adjust prompt based on previous performance
            if iteration == 1:
                mock_prompt.build.return_value = "Initial exploration prompt"
            else:
                last_scores = history.get('scores', [{}])[-1] if history.get('scores') else {}
                if last_scores.get('accuracy', 0) < 0.7:
                    mock_prompt.build.return_value = "Focus on accuracy improvement"
                else:
                    mock_prompt.build.return_value = "Optimize for other metrics"
            
            return mock_prompt
        
        workflow = Workflow(
            self.mock_generator, 
            self.mock_oracle, 
            max_iterations=3,
            enable_history_in_prompts=True
        )
        
        result = await workflow.run(
            prompt=performance_based_prompt,
            reference="test_reference"
        )
        
        # Verify prompt function was called correctly
        self.assertEqual(len(prompt_calls), 3)
        self.assertEqual(prompt_calls[0], (1, 0))  # First iteration, no history
        self.assertEqual(prompt_calls[1], (2, 1))  # Second iteration, 1 output in history
        self.assertEqual(prompt_calls[2], (3, 2))  # Third iteration, 2 outputs in history

    @patch('sde_harness.core.workflow.weave')
    async def test_workflow_performance_monitoring(self, mock_weave):
        """Test workflow performance monitoring and metrics"""
        self.mock_generator.generate_async = AsyncMock(return_value="Generated output")
        mock_weave.op.return_value = lambda func: func
        
        # Mock generation metadata
        self.mock_generator.generate_async.return_value = MagicMock()
        self.mock_generator.generate_async.return_value.content = "Generated output"
        
        workflow = Workflow(
            self.mock_generator, 
            self.mock_oracle, 
            max_iterations=2,
            enable_multi_round_metrics=True
        )
        
        mock_prompt = MagicMock()
        mock_prompt.build.return_value = "Test prompt"
        
        result = await workflow.run(
            prompt=mock_prompt,
            reference="test_reference"
        )
        
        # Verify performance tracking
        self.assertIn('iteration_scores', result)
        self.assertIn('final_scores', result)
        
        # Check that metrics were called for each iteration
        self.assertEqual(self.mock_oracle.evaluate.call_count, 2)
        self.assertEqual(self.mock_oracle.evaluate_multi_round.call_count, 2)

    @patch('sde_harness.core.workflow.weave')
    async def test_workflow_memory_management(self, mock_weave):
        """Test workflow memory management with large histories"""
        self.mock_generator.generate_async = AsyncMock(return_value="Generated output")  
        mock_weave.op.return_value = lambda func: func
        
        # Simulate large output generation
        large_output = "x" * 10000  # 10KB output
        self.mock_generator.generate_async.return_value = large_output
        
        workflow = Workflow(
            self.mock_generator, 
            self.mock_oracle, 
            max_iterations=5
        )
        
        mock_prompt = MagicMock()
        mock_prompt.build.return_value = "Test prompt"
        
        result = await workflow.run(
            prompt=mock_prompt,
            reference="test_reference"
        )
        
        # Verify that large histories are handled
        self.assertIn('outputs', result)
        self.assertEqual(len(result['outputs']), 5)
        for output in result['outputs']:
            self.assertEqual(len(output), 10000)


@pytest.mark.integration
class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for Workflow class"""

    @patch('sde_harness.core.workflow.weave')
    async def test_end_to_end_workflow(self, mock_weave):
        """Test complete end-to-end workflow execution"""
        # Create more realistic mocks
        mock_generator = MagicMock()
        mock_oracle = MagicMock()
        mock_weave.op.return_value = lambda func: func
        
        # Simulate realistic generation outputs
        scientific_outputs = [
            "Hypothesis: Increased temperature affects reaction rate",
            "Experiment: Conducted trials at 25°C, 50°C, 75°C",
            "Results: Reaction rate doubled with each 25°C increase"
        ]
        mock_generator.generate_async = AsyncMock(side_effect=scientific_outputs)
        
        # Simulate realistic evaluation
        mock_oracle.list_metrics.return_value = ['scientific_accuracy', 'completeness']
        mock_oracle.list_multi_round_metrics.return_value = ['progression_quality']
        
        mock_oracle.evaluate.side_effect = [
            {'scientific_accuracy': 0.6, 'completeness': 0.5},
            {'scientific_accuracy': 0.8, 'completeness': 0.7},
            {'scientific_accuracy': 0.95, 'completeness': 0.9}
        ]
        
        mock_oracle.evaluate_multi_round.side_effect = [
            {'progression_quality': 0.1},
            {'progression_quality': 0.3},
            {'progression_quality': 0.5}
        ]
        
        # Create workflow
        workflow = Workflow(
            generator=mock_generator,
            oracle=mock_oracle,
            max_iterations=3,
            enable_history_in_prompts=True,
            enable_multi_round_metrics=True
        )
        
        # Create scientific prompt
        def scientific_prompt(iteration, history):
            mock_prompt = MagicMock()
            if iteration == 1:
                mock_prompt.build.return_value = "Formulate a scientific hypothesis"
            elif iteration == 2:
                mock_prompt.build.return_value = "Design an experiment to test the hypothesis"
            else:
                mock_prompt.build.return_value = "Analyze results and draw conclusions"
            return mock_prompt
        
        # Run workflow
        result = await workflow.run(
            prompt=scientific_prompt,
            reference={'field': 'chemistry', 'topic': 'reaction_kinetics'},
            gen_args={'model_name': 'scientific/model', 'temperature': 0.3}
        )
        
        # Comprehensive verification
        self.assertIn('final_scores', result)
        self.assertIn('iteration_scores', result)
        self.assertIn('outputs', result)
        
        # Check progression
        self.assertEqual(len(result['outputs']), 3)
        self.assertEqual(len(result['iteration_scores']), 3)
        
        # Verify outputs match expected scientific workflow
        self.assertIn('Hypothesis', result['outputs'][0])
        self.assertIn('Experiment', result['outputs'][1])
        self.assertIn('Results', result['outputs'][2])
        
        # Verify improvement over iterations
        scores = result['iteration_scores']
        self.assertLess(scores[0]['scientific_accuracy'], scores[1]['scientific_accuracy'])
        self.assertLess(scores[1]['scientific_accuracy'], scores[2]['scientific_accuracy'])


if __name__ == '__main__':
    unittest.main()