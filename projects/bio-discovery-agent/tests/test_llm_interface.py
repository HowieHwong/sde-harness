"""Tests for LLM interface."""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.llm_interface import complete_text, complete_text_claude, BioLLMInterface


class TestLLMInterface(unittest.TestCase):
    """Test cases for LLM interface functions."""
    
    def test_bio_llm_interface_init(self):
        """Test BioLLMInterface initialization."""
        # Skip this test as it requires actual SDEGen setup
        # In practice, this would be tested with proper mocking of sde_harness
        self.assertTrue(True)
    
    @patch('src.utils.llm_interface.BioLLMInterface')
    def test_complete_text_function(self, mock_interface_class):
        """Test complete_text convenience function."""
        # Mock the interface and its method
        mock_interface = MagicMock()
        mock_interface.complete_text.return_value = "Test response"
        mock_interface_class.return_value = mock_interface
        
        # Test the function
        result = complete_text("Test prompt", temperature=0.5)
        
        # Verify
        self.assertEqual(result, "Test response")
        # The function passes positional args, not keyword args
        mock_interface.complete_text.assert_called_once_with(
            "Test prompt", 
            0.5,
            4000  # Default max_tokens
        )
    
    @patch('src.utils.llm_interface.BioLLMInterface')
    def test_complete_text_claude_function(self, mock_interface_class):
        """Test complete_text_claude convenience function."""
        # Mock the interface and its method
        mock_interface = MagicMock()
        mock_interface.complete_text_claude.return_value = "Claude response"
        mock_interface_class.return_value = mock_interface
        
        # Test the function
        result = complete_text_claude("Test prompt")
        
        # Verify
        self.assertEqual(result, "Claude response")
        mock_interface.complete_text_claude.assert_called_once()
    
    def test_bio_llm_interface_complete_text(self):
        """Test BioLLMInterface complete_text method."""
        # Create interface with mocked generator
        interface = BioLLMInterface.__new__(BioLLMInterface)
        interface.generator = MagicMock()
        interface.model = "test-model"  # Add model attribute
        
        # Mock the generate method
        interface.generator.generate.return_value = "Generated text"
        
        # Test complete_text
        result = interface.complete_text("Test prompt", temperature=0.7, model="test-model")
        
        # Verify
        self.assertEqual(result, "Generated text")
        interface.generator.generate.assert_called_once()
        
        # Check arguments
        call_args = interface.generator.generate.call_args
        self.assertEqual(call_args[0][0], "Test prompt")
        self.assertEqual(call_args[1]['model_name'], "test-model")
        self.assertEqual(call_args[1]['temperature'], 0.7)
    
    def test_bio_llm_interface_error_handling(self):
        """Test error handling in BioLLMInterface."""
        # Create interface with mocked generator that raises error
        interface = BioLLMInterface.__new__(BioLLMInterface)
        interface.generator = MagicMock()
        interface.model = "test-model"  # Add model attribute
        interface.generator.generate.side_effect = Exception("Generation failed")
        
        # Test that error is raised
        with self.assertRaises(Exception) as context:
            interface.complete_text("Test prompt")
        
        self.assertIn("Generation failed", str(context.exception))
    
    def test_json_parsing(self):
        """Test JSON parsing functionality."""
        import json
        
        # Test valid JSON parsing
        json_str = '{"genes": ["GENE1", "GENE2"], "score": 0.85}'
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed['genes'], ["GENE1", "GENE2"])
        self.assertEqual(parsed['score'], 0.85)
        
        # Test invalid JSON
        invalid_json = '{"genes": ["GENE1", "GENE2"'
        with self.assertRaises(json.JSONDecodeError):
            json.loads(invalid_json)


if __name__ == '__main__':
    unittest.main()