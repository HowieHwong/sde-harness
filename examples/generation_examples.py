"""
Examples demonstrating the usage of the updated Generation class
with support for multiple LLM providers via YAML configuration.
"""

import asyncio
import os
import sys
from sde_harness.core.generation import Generation


def basic_usage_example():
    """Basic usage example with different model providers."""
    
    # Get parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Initialize the Generation class using YAML configuration
    with Generation(
        models_file=os.path.join(parent_dir, "models.yaml"), 
        credentials_file=os.path.join(parent_dir, "credentials.yaml")
    ) as gen:

        # Test prompt
        prompt = "Explain the concept of quantum entanglement in simple terms."

        # Test different models (update these based on your models.yaml)
        models_to_test = [
            "openai/gpt-4o-2024-08-06",
            "anthropic/claude-3-7-sonnet-20250219",
            "gemini/gemini-2.5-flash",
        ]

        for model_name in models_to_test:
            try:
                print(f"\n--- Testing {model_name} ---")
                result = gen.generate(
                    prompt=prompt,
                    model_name=model_name,
                    max_tokens=200,
                    temperature=0.7,
                )
                print(f"Provider: {result['provider']}")
                print(f"Model: {result['model']}")
                print(f"Response: {result['text'][:200]}...")
                if result["usage"]:
                    print(f"Usage: {result['usage']}")
            except Exception as e:
                print(f"Error with {model_name}: {e}")


async def async_batch_example():
    """Example of asynchronous batch processing."""
    
    # Get parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with Generation(
        models_file=os.path.join(parent_dir, "models.yaml"), 
        credentials_file=os.path.join(parent_dir, "credentials.yaml"), 
        max_workers=4
    ) as gen:

        # Multiple prompts for batch processing
        prompts = [
            "What is artificial intelligence?",
            "Explain machine learning in one paragraph.",
            "What are the applications of deep learning?",
            "How does natural language processing work?",
            "What is the future of AI research?",
        ]

        print("Running batch generation...")

        # Process all prompts concurrently
        try:
            results = await gen.generate_batch_async(
                prompts=prompts,
                model_name="openai/gpt-4o-2024-08-06",  # Update based on your models.yaml
                max_tokens=150,
                temperature=0.6,
            )

            # Display results
            for i, result in enumerate(results):
                print(f"\nPrompt {i+1}: {prompts[i]}")
                print(f"Response: {result['text'][:100]}...")
                print(
                    f"Tokens used: {result['usage']['total_tokens'] if result['usage'] else 'N/A'}"
                )

        except Exception as e:
            print(f"Batch generation failed: {e}")


def chat_conversation_example():
    """Example of maintaining a conversation context."""
    
    # Get parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with Generation(
        models_file=os.path.join(parent_dir, "models.yaml"), 
        credentials_file=os.path.join(parent_dir, "credentials.yaml")
    ) as gen:

        # Conversation history
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant specializing in science.",
            },
            {"role": "user", "content": "What is photosynthesis?"},
        ]

        print("Starting a conversation about photosynthesis...")

        try:
            # First exchange
            result = gen.generate(
                messages=messages,
                model_name="openai/gpt-4o-2024-08-06",  # Update based on your models.yaml
                max_tokens=200,
            )

            print(f"AI: {result['text']}")

            # Add AI response to conversation
            messages.append({"role": "assistant", "content": result["text"]})

            # Follow-up question
            messages.append(
                {
                    "role": "user",
                    "content": "How does this process affect climate change?",
                }
            )

            result = gen.generate(
                messages=messages, model_name="openai/gpt-4o-2024-08-06", max_tokens=200
            )

            print(f"AI: {result['text']}")

        except Exception as e:
            print(f"Conversation failed: {e}")


def model_comparison_example():
    """Compare responses from different models for the same prompt."""
    
    # Get parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with Generation(
        models_file=os.path.join(parent_dir, "models.yaml"), 
        credentials_file=os.path.join(parent_dir, "credentials.yaml")
    ) as gen:

        prompt = "Write a creative short story about a robot discovering emotions."

        # Update these model names based on your models.yaml configuration
        models = [
            "openai/gpt-4o-2024-08-06",
            "anthropic/claude-3-7-sonnet-20250219",
            "gemini/gemini-2.5-flash",
        ]

        print(f"Prompt: {prompt}\n")

        for model_name in models:
            try:
                result = gen.generate(
                    prompt=prompt,
                    model_name=model_name,
                    max_tokens=300,
                    temperature=0.8,  # Higher temperature for creativity
                )

                print(f"--- {model_name.upper()} ---")
                print(result["text"])
                print(
                    f"(Provider: {result['provider']}, Finish reason: {result.get('finish_reason', 'N/A')})"
                )
                print()

            except Exception as e:
                print(f"Error with {model_name}: {e}\n")


def huggingface_example():
    """Example using Hugging Face models (local models)."""
    
    # Get parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with Generation(
        models_file=os.path.join(parent_dir, "models.yaml"), 
        credentials_file=os.path.join(parent_dir, "credentials.yaml")
    ) as gen:

        # Test with a local HuggingFace model
        # Make sure you have this model configured in your models.yaml with provider: local
        hf_model_names = [
            "huggingface/Qwen/Qwen3-0.6B",  # Based on your models.yaml
            # "huggingface/Qwen/Qwen3-32B",  # Uncomment if you want to test the larger model
        ]

        prompt = "Hello, how are you today?"

        for model_name in hf_model_names:
            try:
                print(f"Testing local HuggingFace model: {model_name}")
                result = gen.generate(
                    prompt=prompt, model_name=model_name, max_tokens=50, temperature=0.7
                )

                print(f"Response: {result['text']}")
                print(f"Provider: {result['provider']}")
                print()

            except Exception as e:
                print(f"Error with {model_name}: {e}")
                print(
                    "Make sure the model is properly configured in models.yaml and downloaded locally\n"
                )


def error_handling_example():
    """Example demonstrating proper error handling."""
    
    # Get parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with Generation(
        models_file=os.path.join(parent_dir, "models.yaml"), 
        credentials_file=os.path.join(parent_dir, "credentials.yaml")
    ) as gen:

        print("Testing error handling scenarios...\n")

        # Test with non-existent model
        try:
            result = gen.generate(prompt="Test prompt", model_name="non-existent-model")
        except KeyError as e:
            print(f"✓ Caught expected KeyError for non-existent model: {e}")

        # Test with empty prompt for local model
        try:
            result = gen.generate(prompt="", model_name="huggingface/Qwen/Qwen3-0.6B")
        except ValueError as e:
            print(f"✓ Caught expected ValueError for empty prompt: {e}")

        # Test with both prompt and messages
        try:
            result = gen.generate(
                prompt="Test",
                messages=[{"role": "user", "content": "Test"}],
                model_name="openai/gpt-4o-2024-08-06",
            )
        except ValueError as e:
            print(f"✓ Caught expected ValueError for both prompt and messages: {e}")

        print("\nError handling tests completed!")


def check_configuration():
    """Check if configuration files exist and are properly set up."""
    
    # Check in parent directory (project root)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_files = ["models.yaml", "credentials.yaml"]
    missing_files = []

    for file in config_files:
        full_path = os.path.join(parent_dir, file)
        if not os.path.exists(full_path):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing configuration files: {', '.join(missing_files)}")
        print("\nPlease ensure you have:")
        print("1. models.yaml - with your model configurations")
        print("2. credentials.yaml - with your API credentials")
        print("\nSee models.template.yaml and credentials.template.yaml for examples.")
        return False
    else:
        print("✓ Configuration files found")
        return True


if __name__ == "__main__":
    print("=== Generation Class Examples ===\n")

    # Check configuration
    if not check_configuration():
        sys.exit(1)

    # Run examples
    try:
        print("\n1. Basic Usage Example")
        basic_usage_example()

        print("\n" + "=" * 50)
        print("2. Chat Conversation Example")
        chat_conversation_example()

        print("\n" + "=" * 50)
        print("3. Model Comparison Example")
        model_comparison_example()

        print("\n" + "=" * 50)
        print("4. Async Batch Example")
        asyncio.run(async_batch_example())

        print("\n" + "=" * 50)
        print("5. HuggingFace Example")
        huggingface_example()

        print("\n" + "=" * 50)
        print("6. Error Handling Example")
        error_handling_example()

    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
