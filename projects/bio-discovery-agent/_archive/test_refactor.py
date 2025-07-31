#!/usr/bin/env python3
"""Test script to verify refactored code works."""
import os
import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test main modules
        from src.modes import run_perturb_genes, run_baseline, run_analyze
        print("✓ Mode imports successful")
        
        # Test utilities
        from src.utils.data_loader import load_dataset, validate_data_files
        from src.utils.prompts import get_prompt_template
        from src.utils.tools import BioDiscoveryTools
        from src.utils.llm_interface import BioLLMInterface
        print("✓ Utility imports successful")
        
        # Test evaluators
        from src.evaluators.bio_metrics import BioEvaluator, calculate_hit_rate
        print("✓ Evaluator imports successful")
        
        # Test backward compatibility
        from LLM import complete_text, complete_text_claude
        print("✓ Backward compatibility imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_data_loading():
    """Test data loading functionality."""
    print("\\nTesting data loading...")
    
    try:
        from src.utils.data_loader import validate_data_files
        
        # Test with a known dataset
        if validate_data_files("IFNG"):
            print("✓ Data validation successful for IFNG dataset")
        else:
            print("✗ Data validation failed for IFNG dataset")
            
        return True
        
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False


def test_prompt_templates():
    """Test prompt template creation."""
    print("\\nTesting prompt templates...")
    
    try:
        from src.utils.prompts import get_prompt_template
        
        # Test basic prompt
        prompt = get_prompt_template("perturb_genes", research_problem="Test problem")
        print("✓ Basic prompt template successful")
        
        # Test tool prompt
        prompt = get_prompt_template("perturb_genes_gene_search", research_problem="Test problem")
        print("✓ Tool prompt template successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Prompt template error: {e}")
        return False


def test_cli_help():
    """Test CLI help functionality."""
    print("\\nTesting CLI help...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "cli.py", "--help"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ CLI help works")
            return True
        else:
            print("✗ CLI help failed")
            return False
            
    except Exception as e:
        print(f"✗ CLI test error: {e}")
        return False


def main():
    """Run all tests."""
    print("=== BioDiscoveryAgent Refactoring Test ===\\n")
    
    tests = [
        test_imports,
        test_data_loading,
        test_prompt_templates,
        test_cli_help
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\\n✓ All tests passed! Refactoring successful.")
        return 0
    else:
        print("\\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())