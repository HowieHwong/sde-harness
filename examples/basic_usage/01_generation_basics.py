#!/usr/bin/env python3
"""
Generation Basics Example

This example demonstrates the core functionality of the Generation class,
including basic text generation, async patterns, and model management.

Run this example:
    python examples/basic_usage/01_generation_basics.py
"""

import sys
import os
import asyncio
from typing import Dict, Any

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from sde_harness.core import Generation


def example_basic_generation():
    """Example 1: Basic text generation"""
    print("🔷 Example 1: Basic Text Generation")
    print("-" * 50)

    # Initialize the generator with configuration files
    generator = Generation(
        models_file=f"{project_root}/models.yaml",
        credentials_file=f"{project_root}/credentials.yaml",
    )

    # Simple generation
    prompt = "What are the key principles of scientific research?"

    try:
        response = generator.generate(
            prompt=prompt,
            model_name="openai/gpt-4o-mini",  # Adjust based on your config
            max_tokens=200,
            # temperature=0.7
        )

        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print(f"Response type: {type(response)}")

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        print("💡 Check your model configuration and API keys")

    print("\n")


async def example_async_generation():
    """Example 2: Asynchronous text generation"""
    print("🔷 Example 2: Async Text Generation")
    print("-" * 50)

    generator = Generation(
        models_file=f"{project_root}/models.yaml",
        credentials_file=f"{project_root}/credentials.yaml",
    )

    prompts = [
        "Explain machine learning in one sentence.",
        "What is the scientific method?",
        "Define artificial intelligence briefly.",
    ]

    try:
        # Generate responses asynchronously
        tasks = [
            generator.generate_async(
                prompt=prompt,
                model_name="openai/gpt-4o-mini",
                max_tokens=100,
                temperature=0.5,
            )
            for prompt in prompts
        ]

        responses = await asyncio.gather(*tasks)

        for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
            print(f"{i}. Prompt: {prompt}")
            print(f"   Response: {response}")
            print()

    except Exception as e:
        print(f"❌ Error during async generation: {e}")

    print()


def example_generation_with_parameters():
    """Example 3: Generation with different parameters"""
    print("🔷 Example 3: Generation with Different Parameters")
    print("-" * 50)

    generator = Generation(
        models_file=f"{project_root}/models.yaml",
        credentials_file=f"{project_root}/credentials.yaml",
    )

    prompt = "Write a short poem about science."

    # Different temperature settings
    temperatures = [0.0, 0.5, 1.0]

    for temp in temperatures:
        try:
            response = generator.generate(
                prompt=prompt,
                model_name="openai/gpt-4o-mini",
                max_tokens=150,
                temperature=temp,
            )

            print(f"Temperature {temp}:")
            print(f"Response: {response}")
            print("-" * 30)

        except Exception as e:
            print(f"❌ Error with temperature {temp}: {e}")

    print()


def example_model_information():
    """Example 4: Getting model information"""
    print("🔷 Example 4: Model Information")
    print("-" * 50)

    try:
        generator = Generation(
            models_file=f"{project_root}/models.yaml",
            credentials_file=f"{project_root}/credentials.yaml",
        )

        # List available models
        models = generator.list_models()
        print("Available models:")
        for model in models:
            print(f"  - {model}")

        print()

        # Get model information
        if models:
            model_name = models[0]
            try:
                info = generator.model_info(model_name)
                print(f"Information for {model_name}:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"Could not get info for {model_name}: {e}")

    except Exception as e:
        print(f"❌ Error getting model information: {e}")

    print()


def example_error_handling():
    """Example 5: Error handling patterns"""
    print("🔷 Example 5: Error Handling")
    print("-" * 50)

    generator = Generation(
        models_file=f"{project_root}/models.yaml",
        credentials_file=f"{project_root}/credentials.yaml",
    )

    # Test with invalid model
    try:
        response = generator.generate(
            prompt="Test prompt", model_name="invalid/model-name", max_tokens=50
        )
        print(f"Unexpected success: {response}")
    except Exception as e:
        print(f"✅ Expected error caught: {type(e).__name__}: {e}")

    # Test with extremely long prompt (depending on model limits)
    try:
        very_long_prompt = "This is a test. " * 10000  # Very long prompt
        response = generator.generate(
            prompt=very_long_prompt, model_name="openai/gpt-4o-mini", max_tokens=50
        )
        print(f"Long prompt succeeded: {len(response)} chars")
    except Exception as e:
        print(f"✅ Long prompt error handled: {type(e).__name__}")

    print()


def example_batch_generation():
    """Example 6: Generating multiple responses"""
    print("🔷 Example 6: Batch Generation")
    print("-" * 50)

    generator = Generation(
        models_file=f"{project_root}/models.yaml",
        credentials_file=f"{project_root}/credentials.yaml",
    )

    base_prompt = "Complete this scientific fact: The speed of light is"

    try:
        # Generate multiple variations
        responses = []
        for i in range(3):
            response = generator.generate(
                prompt=base_prompt,
                model_name="openai/gpt-4o-mini",
                max_tokens=100,
                temperature=0.8,  # Higher temperature for variety
            )
            responses.append(response)

        print(f"Base prompt: {base_prompt}")
        print("\nGenerated responses:")
        for i, response in enumerate(responses, 1):
            print(f"{i}. {response}")
            print()

    except Exception as e:
        print(f"❌ Error in batch generation: {e}")

    print()


def main():
    """Run all generation examples"""
    print("🚀 SDE-Harness Generation Examples")
    print("=" * 60)
    print()

    # Run synchronous examples
    example_basic_generation()
    example_generation_with_parameters()
    example_model_information()
    example_error_handling()
    example_batch_generation()

    # Run async example
    print("Running async example...")
    asyncio.run(example_async_generation())

    print("✅ All generation examples completed!")
    print("\n💡 Next Steps:")
    print("- Try examples/basic_usage/02_oracle_basics.py")
    print("- Experiment with different models and parameters")
    print("- Check your config files for available models")


if __name__ == "__main__":
    main()
