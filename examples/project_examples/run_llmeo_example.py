#!/usr/bin/env python3
"""
Example of how to run LLMEO project from the framework.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def run_llmeo_example():
    """Run a simple LLMEO example."""

    print("🧪 Running LLMEO Example")
    print("=" * 50)

    try:
        # Import LLMEO project components
        from projects.llmeo.src.modes.few_shot import run_few_shot

        # Example parameters
        example_params = {
            "iterations": 3,
            "temperature": 0.7,
            "model_name": "openai/gpt-4o-mini",
            "max_tokens": 150,
        }

        print(f"Parameters: {example_params}")
        print("\n🚀 Starting LLMEO few-shot mode...")

        # Note: This would require proper API keys and data files to run
        # For demonstration purposes, we'll just show the import works
        print("✅ LLMEO modules imported successfully!")
        print("📝 To run LLMEO:")
        print("   1. Set up API keys in <sde_harness_root>/credentials.yaml")
        print("   2. Download required data files")
        print("   3. Run: cd projects/llmeo && python cli.py few-shot --model openai/gpt-4o-mini --iterations 3 --temperature 0.7 --max_tokens 150")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the project structure is set up correctly.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_llmeo_example()
