"""
Examples demonstrating the usage of the updated Generation class
with support for GPT-4o, Gemini, Claude, and Hugging Face models.
"""

import asyncio
import os
import sys
from sci_demo.generation import Generation


def basic_usage_example():
    """Basic usage example with different model providers."""
    
    # Initialize the Generation class with API keys
    gen = Generation(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        claude_api_key=os.getenv("CLAUDE_API_KEY"),
    )
    
    # List available models
    print("Available models:")
    available_models = gen.list_available_models()
    for provider, models in available_models.items():
        print(f"  {provider}: {models}")
    
    # Test prompt
    prompt = "Explain the concept of quantum entanglement in simple terms."
    
    # Test different models
    models_to_test = [
        "gpt-4o",                        # Latest OpenAI model
        "gpt-4-turbo",                   # OpenAI GPT-4 Turbo
        "gemini-1.5-pro",               # Google Gemini Pro
        "claude-3-5-sonnet-20241022",   # Anthropic Claude 3.5 Sonnet
    ]
    
    for model in models_to_test:
        try:
            print(f"\n--- Testing {model} ---")
            result = gen.generate(
                prompt=prompt,
                model=model,
                max_tokens=200,
                temperature=0.7
            )
            print(f"Provider: {result['provider']}")
            print(f"Model: {result['model']}")
            print(f"Response: {result['text'][:200]}...")
            if result['usage']:
                print(f"Usage: {result['usage']}")
        except Exception as e:
            print(f"Error with {model}: {e}")


async def async_batch_example():
    """Example of asynchronous batch processing."""
    
    gen = Generation(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        claude_api_key=os.getenv("CLAUDE_API_KEY"),
        max_workers=4
    )
    
    # Multiple prompts for batch processing
    prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in one paragraph.",
        "What are the applications of deep learning?",
        "How does natural language processing work?",
        "What is the future of AI research?"
    ]
    
    print("Running batch generation with GPT-4o...")
    
    # Process all prompts concurrently
    results = await gen.generate_batch_async(
        prompts=prompts,
        model="gpt-4o",
        max_tokens=150,
        temperature=0.6
    )
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {result['text'][:100]}...")
        print(f"Tokens used: {result['usage']['total_tokens'] if result['usage'] else 'N/A'}")


def chat_conversation_example():
    """Example of maintaining a conversation context."""
    
    gen = Generation(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Conversation history
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant specializing in science."},
        {"role": "user", "content": "What is photosynthesis?"}
    ]
    
    print("Starting a conversation about photosynthesis...")
    
    # First exchange
    result = gen.generate(
        prompt="",  # Empty prompt since we're using messages
        model="gpt-4o",
        max_tokens=200,
        messages=messages
    )
    
    print(f"AI: {result['text']}")
    
    # Add AI response to conversation
    messages.append({"role": "assistant", "content": result['text']})
    
    # Follow-up question
    messages.append({"role": "user", "content": "How does this process affect climate change?"})
    
    result = gen.generate(
        prompt="",
        model="gpt-4o",
        max_tokens=200,
        messages=messages
    )
    
    print(f"AI: {result['text']}")


def model_comparison_example():
    """Compare responses from different models for the same prompt."""
    
    gen = Generation(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        claude_api_key=os.getenv("CLAUDE_API_KEY"),
    )
    
    prompt = "Write a creative short story about a robot discovering emotions."
    
    models = ["gpt-4o", "gemini-1.5-pro", "claude-3-5-sonnet-20241022"]
    
    print(f"Prompt: {prompt}\n")
    
    for model in models:
        try:
            result = gen.generate(
                prompt=prompt,
                model=model,
                max_tokens=300,
                temperature=0.8  # Higher temperature for creativity
            )
            
            print(f"--- {model.upper()} ---")
            print(result['text'])
            print(f"(Provider: {result['provider']}, Finish reason: {result.get('finish_reason', 'N/A')})")
            print()
            
        except Exception as e:
            print(f"Error with {model}: {e}\n")


def huggingface_example():
    """Example using Hugging Face models (requires local model)."""
    
    # Note: This requires downloading a model locally
    # Uncomment and modify the model name as needed
    
    # gen = Generation(hf_model_name="microsoft/DialoGPT-medium")
    # 
    # result = gen.generate(
    #     prompt="Hello, how are you?",
    #     model="huggingface",  # Will use the configured HF model
    #     max_tokens=50,
    #     temperature=0.7
    # )
    # 
    # print(f"HuggingFace model response: {result['text']}")
    
    print("HuggingFace example commented out - uncomment and set model name to use")


if __name__ == "__main__":
    print("=== Generation Class Examples ===\n")
    
    # Check if API keys are available
    api_keys_available = {
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Gemini": bool(os.getenv("GEMINI_API_KEY")),
        "Claude": bool(os.getenv("CLAUDE_API_KEY")),
    }
    
    print("API Keys Status:")
    for provider, available in api_keys_available.items():
        status = "✓ Available" if available else "✗ Not set"
        print(f"  {provider}: {status}")
    print()
    
    if not any(api_keys_available.values()):
        print("No API keys found. Please set environment variables:")
        print("  export OPENAI_API_KEY='your-openai-key'")
        print("  export GEMINI_API_KEY='your-gemini-key'")
        print("  export CLAUDE_API_KEY='your-claude-key'")
        sys.exit(1)
    
    # Run examples
    try:
        print("1. Basic Usage Example")
        basic_usage_example()
        
        print("\n" + "="*50)
        print("2. Chat Conversation Example")
        chat_conversation_example()
        
        print("\n" + "="*50)
        print("3. Model Comparison Example")
        model_comparison_example()
        
        print("\n" + "="*50)
        print("4. Async Batch Example")
        asyncio.run(async_batch_example())
        
        print("\n" + "="*50)
        print("5. HuggingFace Example")
        huggingface_example()
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}") 