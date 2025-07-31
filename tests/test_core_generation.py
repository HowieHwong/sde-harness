"""
Unit tests for sde_harness.core.generation module
"""

import unittest
import sys
import os
import tempfile
import yaml
from unittest.mock import patch, MagicMock, mock_open
import pytest

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sde_harness.core.generation import (
    load_models_and_credentials,
    load_model_config,
    Generation
)


class TestLoadModelsAndCredentials(unittest.TestCase):
    """Test loading of models and credentials configuration"""

    def test_load_models_and_credentials_success(self):
        """Test successful loading of valid configuration files"""
        models_data = {'test_model': {'provider': 'test', 'model': 'test'}}
        credentials_data = {'test': {'api_key': 'test_key'}}
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('yaml.safe_load') as mock_yaml:
                mock_yaml.side_effect = [models_data, credentials_data]
                
                models, credentials = load_models_and_credentials()
                
                self.assertEqual(models, models_data)
                self.assertEqual(credentials, credentials_data)
                self.assertEqual(mock_file.call_count, 2)

    def test_load_models_file_not_found(self):
        """Test handling of missing models file"""
        with patch('builtins.open', side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError) as context:
                load_models_and_credentials()
            
            self.assertIn("Models configuration file not found", str(context.exception))

    def test_load_credentials_file_not_found(self):
        """Test handling of missing credentials file"""
        models_data = {'test_model': {'provider': 'test'}}
        
        with patch('builtins.open') as mock_file:
            mock_file.side_effect = [
                mock_open(read_data="test").return_value,
                FileNotFoundError()
            ]
            with patch('yaml.safe_load', return_value=models_data):
                with self.assertRaises(FileNotFoundError) as context:
                    load_models_and_credentials()
                
                self.assertIn("Credentials configuration file not found", str(context.exception))

    def test_invalid_yaml_models_file(self):
        """Test handling of invalid YAML in models file"""
        with patch('builtins.open', mock_open()):
            with patch('yaml.safe_load', side_effect=yaml.YAMLError("Invalid YAML")):
                with self.assertRaises(ValueError) as context:
                    load_models_and_credentials()
                
                self.assertIn("Invalid YAML in models file", str(context.exception))

    def test_empty_models_file(self):
        """Test handling of empty models file"""
        with patch('builtins.open', mock_open()):
            with patch('yaml.safe_load', side_effect=[None, {}]):
                with self.assertRaises(ValueError) as context:
                    load_models_and_credentials()
                
                self.assertIn("is empty or invalid", str(context.exception))

    def test_empty_credentials_file_allowed(self):
        """Test that empty credentials file is allowed"""
        models_data = {'test_model': {'provider': 'test'}}
        
        with patch('builtins.open', mock_open()):
            with patch('yaml.safe_load', side_effect=[models_data, None]):
                models, credentials = load_models_and_credentials()
                
                self.assertEqual(models, models_data)
                self.assertEqual(credentials, {})


class TestLoadModelConfig(unittest.TestCase):
    """Test model configuration loading"""

    def setUp(self):
        """Set up test data"""
        self.models = {
            'test_model': {
                'provider': 'test',
                'model': 'test-model',
                'credentials': 'test_creds'
            },
            'model_no_creds': {
                'provider': 'test',
                'model': 'test-model-2',
                'credentials': None
            }
        }
        self.credentials = {
            'test_creds': {
                'api_key': 'test_key',
                'api_base': 'test_base'
            }
        }

    def test_load_model_config_success(self):
        """Test successful model config loading"""
        config = load_model_config('test_model', self.models, self.credentials)
        
        self.assertEqual(config['provider'], 'test')
        self.assertEqual(config['model'], 'test-model')
        self.assertEqual(config['credentials']['api_key'], 'test_key')
        self.assertEqual(config['credentials']['api_base'], 'test_base')

    def test_load_model_config_no_credentials(self):
        """Test model config loading without credentials"""
        config = load_model_config('model_no_creds', self.models, self.credentials)
        
        self.assertEqual(config['provider'], 'test')
        self.assertEqual(config['model'], 'test-model-2')
        self.assertEqual(config['credentials'], {})

    def test_load_model_config_model_not_found(self):
        """Test handling of non-existent model"""
        with self.assertRaises(KeyError) as context:
            load_model_config('nonexistent_model', self.models, self.credentials)
        
        self.assertIn("not found in models_file", str(context.exception))

    def test_load_model_config_credentials_not_found(self):
        """Test handling of missing credentials"""
        models = {
            'test_model': {
                'provider': 'test',
                'model': 'test-model',
                'credentials': 'missing_creds'
            }
        }
        
        with self.assertRaises(KeyError) as context:
            load_model_config('test_model', models, self.credentials)
        
        self.assertIn("Credentials", str(context.exception))


class TestGeneration(unittest.TestCase):
    """Test Generation class"""

    def setUp(self):
        """Set up test data"""
        self.sample_models = {
            'test/model': {
                'provider': 'test',
                'model': 'test-model',
                'credentials': 'test'
            }
        }
        self.sample_credentials = {
            'test': {
                'api_key': 'test-key'
            }
        }

    @patch('sde_harness.core.generation.load_models_and_credentials')
    def test_generation_init_success(self, mock_load):
        """Test successful Generation initialization"""
        mock_load.return_value = (self.sample_models, self.sample_credentials)
        
        gen = Generation()
        
        self.assertIsNotNone(gen.models)
        self.assertIsNotNone(gen.credentials)
        self.assertEqual(gen.models, self.sample_models)
        self.assertEqual(gen.credentials, self.sample_credentials)

    @patch('sde_harness.core.generation.load_models_and_credentials')
    def test_generation_init_with_custom_files(self, mock_load):
        """Test Generation initialization with custom config files"""
        mock_load.return_value = (self.sample_models, self.sample_credentials)
        
        gen = Generation(models_file="custom_models.yaml", credentials_file="custom_creds.yaml")
        
        mock_load.assert_called_once_with("custom_models.yaml", "custom_creds.yaml")

    @patch('sde_harness.core.generation.load_models_and_credentials')
    @patch('sde_harness.core.generation.litellm')
    def test_generate_litellm_success(self, mock_litellm, mock_load):
        """Test successful generation with litellm"""
        mock_load.return_value = (self.sample_models, self.sample_credentials)
        
        # Mock litellm response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated text"
        mock_response.usage.total_tokens = 50
        mock_litellm.completion.return_value = mock_response
        
        gen = Generation()
        result = gen.generate("Test prompt", model_name="test/model")
        
        self.assertIsNotNone(result)
        mock_litellm.completion.assert_called_once()

    @patch('sde_harness.core.generation.load_models_and_credentials')
    @patch('sde_harness.core.generation.litellm')
    def test_generate_litellm_failure(self, mock_litellm, mock_load):
        """Test generation failure with litellm"""
        mock_load.return_value = (self.sample_models, self.sample_credentials)
        mock_litellm.completion.side_effect = Exception("LiteLLM error")
        
        gen = Generation()
        
        with self.assertRaises(RuntimeError) as context:
            gen.generate("Test prompt", model_name="test/model")
        
        self.assertIn("LiteLLM generation failed", str(context.exception))

    @patch('sde_harness.core.generation.load_models_and_credentials')
    def test_generate_invalid_model(self, mock_load):
        """Test generation with invalid model name"""
        mock_load.return_value = (self.sample_models, self.sample_credentials)
        
        gen = Generation()
        
        with self.assertRaises(KeyError):
            gen.generate("Test prompt", model_name="nonexistent/model")

    @patch('sde_harness.core.generation.load_models_and_credentials')
    @patch('sde_harness.core.generation.AutoTokenizer')
    @patch('sde_harness.core.generation.AutoModelForCausalLM')
    @patch('sde_harness.core.generation.pipeline')
    def test_generate_local_model_success(self, mock_pipeline, mock_model, mock_tokenizer, mock_load):
        """Test successful generation with local model"""
        models = {
            'local/test': {
                'provider': 'local',
                'model': 'test/model',
                'credentials': None
            }
        }
        mock_load.return_value = (models, {})
        
        # Mock tokenizer and model
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4]
        mock_tokenizer_instance.decode.return_value = "Generated local text"
        mock_model_instance.generate.return_value = [[1, 2, 3, 4, 5, 6]]
        
        gen = Generation()
        result = gen.generate("Test prompt", model_name="local/test")
        
        self.assertIsNotNone(result)
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()

    @patch('sde_harness.core.generation.load_models_and_credentials')
    @patch('sde_harness.core.generation.litellm')
    async def test_generate_async(self, mock_litellm, mock_load):
        """Test async generation"""
        mock_load.return_value = (self.sample_models, self.sample_credentials)
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Async generated text"
        mock_litellm.completion.return_value = mock_response
        
        gen = Generation()
        result = await gen.generate_async("Test prompt", model_name="test/model")
        
        self.assertIsNotNone(result)

    # Note: Generation class doesn't have model_info method, so this test is removed


@pytest.mark.integration
class TestGenerationIntegration(unittest.TestCase):
    """Integration tests for Generation class"""

    def test_generation_with_temp_files(self):
        """Test Generation with temporary configuration files"""
        models_data = {
            'test/model': {
                'provider': 'test',
                'model': 'test-model',
                'credentials': 'test'
            }
        }
        credentials_data = {
            'test': {'api_key': 'test-key'}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary config files
            models_path = os.path.join(temp_dir, 'models.yaml')
            credentials_path = os.path.join(temp_dir, 'credentials.yaml')
            
            with open(models_path, 'w') as f:
                yaml.dump(models_data, f)
            
            with open(credentials_path, 'w') as f:
                yaml.dump(credentials_data, f)
            
            # Test Generation initialization
            gen = Generation(models_file=models_path, credentials_file=credentials_path)
            
            self.assertEqual(gen.models, models_data)
            self.assertEqual(gen.credentials, credentials_data)


if __name__ == '__main__':
    unittest.main()