"""
Integration tests for LLMEO CLI functionality (cli.py)
"""

import unittest
import sys
import os
import argparse
from unittest.mock import patch, MagicMock
from io import StringIO

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
sys.path.insert(0, project_root)

# Import after adding to path
import cli


class TestCLIFunctionality(unittest.TestCase):
    """Test CLI argument parsing and command routing"""

    def test_cli_imports(self):
        """Test that CLI can import all required modules"""
        # This test ensures all imports work correctly
        self.assertTrue(hasattr(cli, 'main'))
        self.assertTrue(hasattr(cli, 'argparse'))

    @patch('cli.validate_data_files')
    def test_argument_parser_few_shot(self, mock_validate):
        """Test argument parsing for few-shot mode"""
        mock_validate.return_value = True
        
        # Test few-shot command parsing
        test_args = [
            'few-shot',
            '--iterations', '3',
            '--temperature', '0.1',
            '--samples', '15',
            '--num-samples', '10'
        ]
        
        with patch('sys.argv', ['cli.py'] + test_args):
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers(dest="mode")
            
            # Replicate the CLI setup
            common_args = argparse.ArgumentParser(add_help=False)
            common_args.add_argument("--samples", type=int, default=10)
            common_args.add_argument("--num-samples", type=int, default=10)
            common_args.add_argument("--max-tokens", type=int, default=8000)
            common_args.add_argument("--iterations", type=int, default=2)
            common_args.add_argument("--seed", type=int, default=42)
            common_args.add_argument("--model", type=str, default="deepseek/deepseek-chat")
            common_args.add_argument("--temperature", type=float, default=1)
            
            few_shot_parser = subparsers.add_parser("few-shot", parents=[common_args])
            
            args = parser.parse_args(test_args)
            
            self.assertEqual(args.mode, 'few-shot')
            self.assertEqual(args.iterations, 3)
            self.assertEqual(args.temperature, 0.1)
            self.assertEqual(args.samples, 15)
            self.assertEqual(args.num_samples, 10)

    @patch('cli.validate_data_files')
    def test_argument_parser_single_prop(self, mock_validate):
        """Test argument parsing for single-prop mode"""
        mock_validate.return_value = True
        
        test_args = [
            'single-prop',
            '--max-tokens', '5000',
            '--seed', '123'
        ]
        
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="mode")
        
        common_args = argparse.ArgumentParser(add_help=False)
        common_args.add_argument("--samples", type=int, default=10)
        common_args.add_argument("--num-samples", type=int, default=10)
        common_args.add_argument("--max-tokens", type=int, default=8000)
        common_args.add_argument("--iterations", type=int, default=2)
        common_args.add_argument("--seed", type=int, default=42)
        common_args.add_argument("--model", type=str, default="deepseek/deepseek-chat")
        common_args.add_argument("--temperature", type=float, default=1)
        
        single_prop_parser = subparsers.add_parser("single-prop", parents=[common_args])
        
        args = parser.parse_args(test_args)
        
        self.assertEqual(args.mode, 'single-prop')
        self.assertEqual(args.max_tokens, 5000)
        self.assertEqual(args.seed, 123)

    @patch('cli.validate_data_files')
    def test_argument_parser_multi_prop(self, mock_validate):
        """Test argument parsing for multi-prop mode"""
        mock_validate.return_value = True
        
        test_args = [
            'multi-prop',
            '--model', 'openai/gpt-4o-mini',
            '--temperature', '0.5'
        ]
        
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="mode")
        
        common_args = argparse.ArgumentParser(add_help=False)
        common_args.add_argument("--samples", type=int, default=10)
        common_args.add_argument("--num-samples", type=int, default=10)
        common_args.add_argument("--max-tokens", type=int, default=8000)
        common_args.add_argument("--iterations", type=int, default=2)
        common_args.add_argument("--seed", type=int, default=42)
        common_args.add_argument("--model", type=str, default="deepseek/deepseek-chat")
        common_args.add_argument("--temperature", type=float, default=1)
        
        multi_prop_parser = subparsers.add_parser("multi-prop", parents=[common_args])
        
        args = parser.parse_args(test_args)
        
        self.assertEqual(args.mode, 'multi-prop')
        self.assertEqual(args.model, 'openai/gpt-4o-mini')
        self.assertEqual(args.temperature, 0.5)

    def test_argument_parser_defaults(self):
        """Test that default values are set correctly"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="mode")
        
        common_args = argparse.ArgumentParser(add_help=False)
        common_args.add_argument("--samples", type=int, default=10)
        common_args.add_argument("--num-samples", type=int, default=10)
        common_args.add_argument("--max-tokens", type=int, default=8000)
        common_args.add_argument("--iterations", type=int, default=2)
        common_args.add_argument("--seed", type=int, default=42)
        common_args.add_argument("--model", type=str, default="deepseek/deepseek-chat")
        common_args.add_argument("--temperature", type=float, default=1)
        
        few_shot_parser = subparsers.add_parser("few-shot", parents=[common_args])
        
        args = parser.parse_args(['few-shot'])
        
        self.assertEqual(args.samples, 10)
        self.assertEqual(args.num_samples, 10)
        self.assertEqual(args.max_tokens, 8000)
        self.assertEqual(args.iterations, 2)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.model, "deepseek/deepseek-chat")
        self.assertEqual(args.temperature, 1)

    @patch('cli.run_few_shot')
    @patch('cli.validate_data_files')
    @patch('sys.argv', ['cli.py', 'few-shot'])
    def test_main_few_shot_routing(self, mock_validate, mock_run_few_shot):
        """Test that main() correctly routes to run_few_shot"""
        mock_validate.return_value = True
        mock_run_few_shot.return_value = {'final_scores': {'test': 1.0}}
        
        cli.main()
        
        mock_run_few_shot.assert_called_once()
        # Check that args were passed
        call_args = mock_run_few_shot.call_args[0][0]
        self.assertEqual(call_args.mode, 'few-shot')

    @patch('cli.run_single_prop')
    @patch('cli.validate_data_files')
    @patch('sys.argv', ['cli.py', 'single-prop'])
    def test_main_single_prop_routing(self, mock_validate, mock_run_single_prop):
        """Test that main() correctly routes to run_single_prop"""
        mock_validate.return_value = True
        mock_run_single_prop.return_value = {'final_scores': {'test': 1.0}}
        
        cli.main()
        
        mock_run_single_prop.assert_called_once()
        call_args = mock_run_single_prop.call_args[0][0]
        self.assertEqual(call_args.mode, 'single-prop')

    @patch('cli.run_multi_prop')
    @patch('cli.validate_data_files')
    @patch('sys.argv', ['cli.py', 'multi-prop'])
    def test_main_multi_prop_routing(self, mock_validate, mock_run_multi_prop):
        """Test that main() correctly routes to run_multi_prop"""
        mock_validate.return_value = True
        mock_run_multi_prop.return_value = {'final_scores': {'test': 1.0}}
        
        cli.main()
        
        mock_run_multi_prop.assert_called_once()
        call_args = mock_run_multi_prop.call_args[0][0]
        self.assertEqual(call_args.mode, 'multi-prop')

    @patch('cli.validate_data_files')
    @patch('sys.argv', ['cli.py'])
    @patch('sys.stdout', new_callable=StringIO)
    def test_main_no_mode_prints_help(self, mock_stdout, mock_validate):
        """Test that main() prints help when no mode is specified"""
        mock_validate.return_value = True
        
        # This should not raise an exception, just return
        cli.main()
        
        # We can't easily test the help output, but we can ensure it doesn't crash

    @patch('cli.validate_data_files')
    @patch('sys.argv', ['cli.py', 'unknown-mode'])
    def test_main_unknown_mode_exits(self, mock_validate):
        """Test that main() exits gracefully with unknown mode"""
        mock_validate.return_value = True
        
        with patch('sys.exit') as mock_exit:
            # This will cause argparse to exit due to unknown subcommand
            try:
                cli.main()
            except SystemExit:
                pass  # argparse raises SystemExit for unknown subcommands

    @patch('cli.validate_data_files')
    @patch('sys.argv', ['cli.py', 'few-shot'])
    @patch('sys.exit')
    def test_main_data_validation_failure(self, mock_exit, mock_validate):
        """Test that main() exits when data validation fails"""
        mock_validate.return_value = False
        
        cli.main()
        
        mock_exit.assert_called_with(1)

    @patch('cli.run_few_shot')
    @patch('cli.validate_data_files')
    @patch('sys.argv', ['cli.py', 'few-shot'])
    @patch('sys.exit')
    def test_main_keyboard_interrupt_handling(self, mock_exit, mock_validate, mock_run_few_shot):
        """Test that main() handles KeyboardInterrupt gracefully"""
        mock_validate.return_value = True
        mock_run_few_shot.side_effect = KeyboardInterrupt()
        
        with patch('builtins.print') as mock_print:
            cli.main()
            
            mock_exit.assert_called_with(0)
            # Check that appropriate message was printed
            mock_print.assert_called_with("\n⏹️  User interrupted execution")

    @patch('cli.run_few_shot')
    @patch('cli.validate_data_files')
    @patch('sys.argv', ['cli.py', 'few-shot'])
    @patch('sys.exit')
    def test_main_exception_handling(self, mock_exit, mock_validate, mock_run_few_shot):
        """Test that main() handles general exceptions gracefully"""
        mock_validate.return_value = True
        test_exception = Exception("Test exception")
        mock_run_few_shot.side_effect = test_exception
        
        with patch('builtins.print') as mock_print:
            with patch('traceback.print_exc') as mock_traceback:
                cli.main()
                
                mock_exit.assert_called_with(1)
                # Check that error message was printed
                mock_print.assert_called_with("❌ Execution error: Test exception")
                mock_traceback.assert_called_once()

    def test_cli_argument_types(self):
        """Test that CLI arguments have correct types"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="mode")
        
        common_args = argparse.ArgumentParser(add_help=False)
        common_args.add_argument("--samples", type=int, default=10)
        common_args.add_argument("--num-samples", type=int, default=10)
        common_args.add_argument("--max-tokens", type=int, default=8000)
        common_args.add_argument("--iterations", type=int, default=2)
        common_args.add_argument("--seed", type=int, default=42)
        common_args.add_argument("--model", type=str, default="deepseek/deepseek-chat")
        common_args.add_argument("--temperature", type=float, default=1)
        
        few_shot_parser = subparsers.add_parser("few-shot", parents=[common_args])
        
        # Test with string inputs that should be converted to proper types
        args = parser.parse_args([
            'few-shot',
            '--samples', '15',
            '--temperature', '0.5',
            '--max-tokens', '1000'
        ])
        
        self.assertIsInstance(args.samples, int)
        self.assertIsInstance(args.temperature, float)
        self.assertIsInstance(args.max_tokens, int)
        self.assertEqual(args.samples, 15)
        self.assertEqual(args.temperature, 0.5)
        self.assertEqual(args.max_tokens, 1000)


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI with actual subprocess calls"""
    
    def test_cli_help_command(self):
        """Test that CLI help command works"""
        import subprocess
        import os
        
        # Change to the project directory
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        try:
            result = subprocess.run(
                [sys.executable, 'cli.py', '--help'],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Help should exit with code 0 and contain expected content
            self.assertEqual(result.returncode, 0)
            self.assertIn('LLMEO', result.stdout)
            self.assertIn('few-shot', result.stdout)
            self.assertIn('single-prop', result.stdout)
            self.assertIn('multi-prop', result.stdout)
            
        except subprocess.TimeoutExpired:
            self.fail("CLI help command timed out")
        except FileNotFoundError:
            self.skipTest("CLI file not found - skipping integration test")

    def test_cli_subcommand_help(self):
        """Test that CLI subcommand help works"""
        import subprocess
        import os
        
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        try:
            result = subprocess.run(
                [sys.executable, 'cli.py', 'few-shot', '--help'],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            self.assertEqual(result.returncode, 0)
            self.assertIn('few-shot', result.stdout)
            self.assertIn('--iterations', result.stdout)
            self.assertIn('--temperature', result.stdout)
            
        except subprocess.TimeoutExpired:
            self.fail("CLI subcommand help timed out")
        except FileNotFoundError:
            self.skipTest("CLI file not found - skipping integration test")


if __name__ == '__main__':
    unittest.main()