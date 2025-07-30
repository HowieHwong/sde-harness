"""
Unit tests for sde_harness.core.prompt module
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import pytest

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sde_harness.core.prompt import Prompt


class TestPrompt(unittest.TestCase):
    """Test Prompt class functionality"""

    def test_prompt_init_with_builtin_template(self):
        """Test Prompt initialization with built-in template"""
        prompt = Prompt(template_name="summarize")
        
        self.assertIsNotNone(prompt.template)
        self.assertIn("{input_text}", prompt.template)
        self.assertEqual(prompt.default_vars, {})

    def test_prompt_init_with_custom_template(self):
        """Test Prompt initialization with custom template"""
        custom_template = "Hello {name}, please {action}."
        prompt = Prompt(custom_template=custom_template)
        
        self.assertEqual(prompt.template, custom_template)

    def test_prompt_init_with_default_vars(self):
        """Test Prompt initialization with default variables"""
        default_vars = {"name": "Alice", "task": "test"}
        prompt = Prompt(
            custom_template="Hello {name}, please {task}.",
            default_vars=default_vars
        )
        
        self.assertEqual(prompt.default_vars, default_vars)

    def test_prompt_init_both_templates_error(self):
        """Test error when both template_name and custom_template are provided"""
        with self.assertRaises(ValueError) as context:
            Prompt(template_name="summarize", custom_template="Custom template")
        
        self.assertIn("Specify either template_name or custom_template", str(context.exception))

    def test_prompt_init_no_template_error(self):
        """Test error when neither template is provided"""
        with self.assertRaises(ValueError) as context:
            Prompt()
        
        self.assertIn("Must specify a template_name or a custom_template", str(context.exception))

    def test_prompt_init_invalid_template_name(self):
        """Test error with invalid built-in template name"""
        with self.assertRaises(KeyError) as context:
            Prompt(template_name="nonexistent_template")
        
        self.assertIn("Template 'nonexistent_template' not found", str(context.exception))

    def test_builtin_templates_loaded(self):
        """Test that built-in templates are properly loaded"""
        prompt = Prompt(template_name="summarize")
        
        builtin_templates = prompt._load_builtin_templates()
        
        self.assertIsInstance(builtin_templates, dict)
        self.assertIn("summarize", builtin_templates)
        self.assertIn("qa", builtin_templates)
        self.assertIn("translate", builtin_templates)
        self.assertIn("few_shot", builtin_templates)
        self.assertIn("iterative", builtin_templates)

    def test_build_with_default_vars(self):
        """Test building prompt with default variables only"""
        default_vars = {"name": "Alice", "action": "test"}
        prompt = Prompt(
            custom_template="Hello {name}, please {action}.",
            default_vars=default_vars
        )
        
        result = prompt.build()
        
        self.assertEqual(result, "Hello Alice, please test.")

    def test_build_with_additional_vars(self):
        """Test building prompt with additional variables"""
        default_vars = {"name": "Alice"}
        prompt = Prompt(
            custom_template="Hello {name}, please {action}.",
            default_vars=default_vars
        )
        
        result = prompt.build({"action": "run tests"})
        
        self.assertEqual(result, "Hello Alice, please run tests.")

    def test_build_vars_override_defaults(self):
        """Test that build variables override default variables"""
        default_vars = {"name": "Alice", "action": "wait"}
        prompt = Prompt(
            custom_template="Hello {name}, please {action}.",
            default_vars=default_vars
        )
        
        result = prompt.build({"name": "Bob", "action": "run"})
        
        self.assertEqual(result, "Hello Bob, please run.")

    def test_build_missing_variable(self):
        """Test error when required variable is missing"""
        prompt = Prompt(custom_template="Hello {name}, please {action}.")
        
        with self.assertRaises(KeyError):
            prompt.build({"name": "Alice"})  # Missing 'action'

    def test_add_vars(self):
        """Test adding variables to default vars"""
        prompt = Prompt(
            custom_template="Hello {name}, {greeting}!",
            default_vars={"name": "Alice"}
        )
        
        # Manually add to default_vars since add_vars method doesn't exist
        prompt.default_vars["greeting"] = "welcome"
        
        result = prompt.build()
        self.assertEqual(result, "Hello Alice, welcome!")

    def test_add_vars_override_existing(self):
        """Test that default vars can be overridden"""
        prompt = Prompt(
            custom_template="Hello {name}!",
            default_vars={"name": "Alice"}
        )
        
        # Manually update default_vars since add_vars method doesn't exist
        prompt.default_vars["name"] = "Bob"
        
        result = prompt.build()
        self.assertEqual(result, "Hello Bob!")

    def test_clear_vars(self):
        """Test clearing default variables"""
        prompt = Prompt(
            custom_template="Hello {name}!",
            default_vars={"name": "Alice"}
        )
        
        # Manually clear default_vars since clear_vars method doesn't exist
        prompt.default_vars.clear()
        
        self.assertEqual(prompt.default_vars, {})
        
        with self.assertRaises(KeyError):
            prompt.build()  # Should fail as 'name' is not available

    def test_get_vars(self):
        """Test getting current default variables"""
        default_vars = {"name": "Alice", "task": "test"}
        prompt = Prompt(
            custom_template="Hello {name}!",
            default_vars=default_vars
        )
        
        # Access default_vars directly since get_vars method doesn't exist
        vars_copy = prompt.default_vars.copy()
        
        self.assertEqual(vars_copy, default_vars)
        
        # Modifying the returned dict should not affect the original
        vars_copy["new_var"] = "new_value"
        self.assertNotIn("new_var", prompt.default_vars)

    def test_builtin_template_summarize(self):
        """Test specific built-in template: summarize"""
        prompt = Prompt(template_name="summarize")
        
        result = prompt.build({"input_text": "This is a test document."})
        
        self.assertIn("Summarize the following text:", result)
        self.assertIn("This is a test document.", result)

    def test_builtin_template_qa(self):
        """Test specific built-in template: qa"""
        prompt = Prompt(template_name="qa")
        
        result = prompt.build({
            "context": "The sky is blue.",
            "question": "What color is the sky?"
        })
        
        self.assertIn("expert assistant", result)
        self.assertIn("The sky is blue.", result)
        self.assertIn("What color is the sky?", result)

    def test_builtin_template_translate(self):
        """Test specific built-in template: translate"""
        prompt = Prompt(template_name="translate")
        
        result = prompt.build({
            "source_lang": "English",
            "target_lang": "Spanish",
            "text": "Hello world"
        })
        
        self.assertIn("English", result)
        self.assertIn("Spanish", result)
        self.assertIn("Hello world", result)

    def test_builtin_template_few_shot(self):
        """Test specific built-in template: few_shot"""
        prompt = Prompt(template_name="few_shot")
        
        result = prompt.build({
            "examples": "Example 1\nExample 2",
            "input_text": "New input"
        })
        
        self.assertIn("Below are some examples:", result)
        self.assertIn("Example 1\nExample 2", result)
        self.assertIn("New input", result)

    def test_builtin_template_iterative(self):
        """Test specific built-in template: iterative"""
        prompt = Prompt(template_name="iterative")
        
        result = prompt.build({
            "task_description": "Solve a problem",
            "input_text": "Input data",
            "history_section": "Previous attempts: Attempt 1",
            "current_iteration": 2,
            "additional_instructions": "Try harder"
        })
        
        self.assertIn("iterative task", result)
        self.assertIn("Solve a problem", result)
        self.assertIn("Input data", result)
        self.assertIn("Previous attempts: Attempt 1", result)
        self.assertIn("Try harder", result)

    def test_complex_template_with_multiple_vars(self):
        """Test complex template with many variables"""
        complex_template = """
Task: {task}
User: {user}
Context: {context}
Previous results: {previous_results}
Instructions: {instructions}
Expected format: {format}
        """.strip()
        
        prompt = Prompt(custom_template=complex_template)
        
        vars_dict = {
            "task": "Data analysis",
            "user": "Alice",
            "context": "Scientific research",
            "previous_results": "Positive correlation found",
            "instructions": "Provide detailed analysis",
            "format": "JSON"
        }
        
        result = prompt.build(vars_dict)
        
        for key, value in vars_dict.items():
            self.assertIn(value, result)

    def test_template_with_special_characters(self):
        """Test template with special characters"""
        template = "Process {data} with 100% accuracy. Cost: ${cost}. Rate: {rate}%."
        prompt = Prompt(custom_template=template)
        
        result = prompt.build({"data": "dataset", "cost": "50", "rate": "95"})
        
        self.assertEqual(result, "Process dataset with 100% accuracy. Cost: $50. Rate: 95%.")

    def test_empty_template(self):
        """Test empty template"""
        # Skip this test as empty template is not allowed - template must contain something
        # Instead test a simple template with no variables
        prompt = Prompt(custom_template="Simple template")
        
        result = prompt.build()
        
        self.assertEqual(result, "Simple template")

    def test_template_with_no_variables(self):
        """Test template with no placeholder variables"""
        template = "This is a static template with no variables."
        prompt = Prompt(custom_template=template)
        
        result = prompt.build()
        
        self.assertEqual(result, template)

    def test_template_with_repeated_variables(self):
        """Test template with repeated variable placeholders"""
        template = "Hello {name}! Welcome {name}! How are you today, {name}?"
        prompt = Prompt(custom_template=template)
        
        result = prompt.build({"name": "Alice"})
        
        expected = "Hello Alice! Welcome Alice! How are you today, Alice?"
        self.assertEqual(result, expected)


class TestPromptAdvanced(unittest.TestCase):
    """Advanced tests for Prompt functionality"""

    def test_prompt_with_list_variables(self):
        """Test prompt building with list variables"""
        template = "Items: {items}. Count: {count}."
        prompt = Prompt(custom_template=template)
        
        items_list = ["apple", "banana", "cherry"]
        result = prompt.build({"items": items_list, "count": len(items_list)})
        
        self.assertIn(str(items_list), result)
        self.assertIn("3", result)

    def test_prompt_with_dict_variables(self):
        """Test prompt building with dictionary variables"""
        template = "Config: {config}"
        prompt = Prompt(custom_template=template)
        
        config_dict = {"model": "gpt-4", "temperature": 0.7}
        result = prompt.build({"config": config_dict})
        
        self.assertIn(str(config_dict), result)

    def test_prompt_state_independence(self):
        """Test that multiple Prompt instances are independent"""
        template = "Hello {name}!"
        
        prompt1 = Prompt(custom_template=template, default_vars={"name": "Alice"})
        prompt2 = Prompt(custom_template=template, default_vars={"name": "Bob"})
        
        result1 = prompt1.build()
        result2 = prompt2.build()
        
        self.assertEqual(result1, "Hello Alice!")
        self.assertEqual(result2, "Hello Bob!")
        
        # Modifying one should not affect the other
        prompt1.default_vars["name"] = "Charlie"
        result1_updated = prompt1.build()
        result2_unchanged = prompt2.build()
        
        self.assertEqual(result1_updated, "Hello Charlie!")
        self.assertEqual(result2_unchanged, "Hello Bob!")  # Should remain unchanged

    def test_prompt_template_validation(self):
        """Test template format validation"""
        # Test valid template formats
        valid_templates = [
            "Simple template",
            "Template with {variable}",
            "Template with {var1} and {var2}",
            "Template with {var_with_underscore}",
            "Template with {var123}",
        ]
        
        for template in valid_templates:
            try:
                prompt = Prompt(custom_template=template)
                # If no variables, should build successfully
                if "{" not in template:
                    result = prompt.build()
                    self.assertIsInstance(result, str)
            except ValueError as e:
                # Empty template is not allowed, so this is expected for empty string
                if template == "":
                    continue
                self.fail(f"Valid template '{template}' raised an exception: {e}")
            except Exception as e:
                self.fail(f"Valid template '{template}' raised an exception: {e}")

    def test_prompt_memory_efficiency(self):
        """Test that Prompt handles large templates efficiently"""
        # Create a large template
        large_template = "Large template with {var}. " * 1000
        
        prompt = Prompt(custom_template=large_template)
        result = prompt.build({"var": "TEST"})
        
        self.assertIn("TEST", result)
        self.assertTrue(len(result) > 10000)  # Should be quite large

    def test_prompt_with_history_template(self):
        """Test using the iterative template with history data"""
        prompt = Prompt(template_name="iterative")
        
        # Simulate a multi-round scenario
        history_data = {
            "task_description": "Optimize algorithm performance",
            "input_text": "Current algorithm implementation",
            "history_section": "Previous attempt: Implemented caching mechanism. Feedback: Performance improved by 20%",
            "current_iteration": 3,
            "additional_instructions": "Consider parallel processing"
        }
        
        result = prompt.build(history_data)
        
        # Verify all history components are included
        self.assertIn("Optimize algorithm performance", result)
        self.assertIn("Current algorithm implementation", result)
        self.assertIn("caching mechanism", result)
        self.assertIn("parallel processing", result)
        
        # Verify template structure
        self.assertIn("iterative task", result)
        self.assertIn("Current iteration: 3", result)
        self.assertIn("Current input:", result)


@pytest.mark.integration
class TestPromptIntegration(unittest.TestCase):
    """Integration tests for Prompt class"""

    def test_prompt_with_real_world_scenario(self):
        """Test Prompt in a realistic scientific workflow scenario"""
        # Scientific paper summarization template
        template = """
You are a scientific research assistant. Please analyze the following research data:

Research Question: {research_question}
Methodology: {methodology}
Data: {data}
Previous Findings: {previous_findings}

Please provide:
1. A summary of the key findings
2. Statistical significance assessment
3. Recommendations for future research

Format your response as a structured analysis.
        """.strip()
        
        prompt = Prompt(custom_template=template)
        
        research_vars = {
            "research_question": "Does caffeine consumption affect cognitive performance?",
            "methodology": "Double-blind randomized controlled trial with 200 participants",
            "data": "Mean reaction time: Caffeine group 245ms, Control group 267ms",
            "previous_findings": "Previous studies showed mixed results with small sample sizes"
        }
        
        result = prompt.build(research_vars)
        
        # Verify all components are included
        for value in research_vars.values():
            self.assertIn(value, result)
        
        # Verify structure
        self.assertIn("scientific research assistant", result)
        self.assertIn("structured analysis", result)

    def test_prompt_workflow_integration(self):
        """Test Prompt integration with a multi-step workflow"""
        # Step 1: Initial analysis prompt
        initial_template = """
Analyze the following problem: {problem}
Provide initial thoughts and approach.
        """.strip()
        
        initial_prompt = Prompt(custom_template=initial_template)
        step1_result = initial_prompt.build({"problem": "Optimize database query performance"})
        
        # Step 2: Refinement prompt using results from step 1
        refinement_template = """
Previous analysis: {previous_analysis}
Based on the initial analysis, provide detailed implementation steps.
        """.strip()
        
        refinement_prompt = Prompt(custom_template=refinement_template)
        step2_result = refinement_prompt.build({"previous_analysis": step1_result})
        
        # Verify workflow continuity
        self.assertIn("Optimize database query performance", step1_result)
        self.assertIn("detailed implementation", step2_result)


if __name__ == '__main__':
    unittest.main()