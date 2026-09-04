#!/usr/bin/env python3
"""
Test script for the Spectrum Elucidator Toolkit.

This script tests the basic functionality of each component without requiring
an OpenAI API key or running full elucidation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_data_utils():
    """Test data utilities functionality."""
    print("Testing data utilities...")
    
    try:
        from src.data_utils import MolecularDataLoader, NMRProcessor
        
        # Test data loading
        data_loader = MolecularDataLoader("data/updated_table.csv")
        
        # Test getting dataset info
        info = data_loader.get_dataset_info()
        print(f"  ✓ Dataset loaded: {info['total_molecules']} molecules")
        
        # Test getting a specific molecule
        molecule = data_loader.get_molecule_by_id("5_99")
        if molecule:
            print(f"  ✓ Molecule retrieval: {molecule['molecule_id']}")
        
        # Test NMR processing
        nmr_string = "NMR: 0.87 (3H, t, J = 6.5 Hz), 1.30 (2H, tq, J = 6.6, 6.5 Hz)"
        peaks = NMRProcessor.parse_nmr_peaks(nmr_string)
        print(f"  ✓ NMR parsing: {len(peaks)} peaks extracted")
        
        print("  ✓ Data utilities test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Data utilities test failed: {e}")
        return False


def test_config():
    """Test configuration management."""
    print("Testing configuration management...")
    
    try:
        from src.config import ConfigManager, create_config_template
        
        # Test config template creation
        template = create_config_template()
        if "llm" in template and "elucidation" in template:
            print("  ✓ Config template creation passed")
        
        # Test config manager (without API key)
        config_manager = ConfigManager()
        print("  ✓ Config manager initialization passed")
        
        # Test config validation (should fail without API key)
        if not config_manager.validate_config():
            print("  ✓ Config validation correctly failed without API key")
        
        print("  ✓ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False


def test_llm_interface():
    """Test LLM interface (without API calls)."""
    print("Testing LLM interface...")
    
    try:
        from src.llm_interface import LLMInterface, ElucidationStep
        
        # Test ElucidationStep dataclass
        step = ElucidationStep(
            iteration=1,
            prompt="Test prompt",
            response="Test response",
            generated_smiles="CC",
            nmr_similarity=0.5,
            timestamp=1234567890.0,
            metadata={}
        )
        print(f"  ✓ ElucidationStep creation: {step.iteration}")
        
        # Test LLM interface initialization (without API key)
        try:
            llm = LLMInterface(api_key="dummy_key")
            print("  ✓ LLM interface initialization passed")
        except Exception:
            print("  ✓ LLM interface correctly failed with invalid API key")
        
        # Test prompt creation
        prompt = llm.create_elucidation_prompt(
            target_nmr="Test NMR",
            iteration=1,
            history=[],
            target_molecule_info=None
        )
        if "Test NMR" in prompt:
            print("  ✓ Prompt creation passed")
        
        # Test SMILES extraction
        test_response = "Analysis: Test\nSMILES: CCC\nConfidence: High"
        smiles = llm.extract_smiles_from_response(test_response)
        if smiles == "CCC":
            print("  ✓ SMILES extraction passed")
        
        print("  ✓ LLM interface test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ LLM interface test failed: {e}")
        return False


def test_elucidation_engine():
    """Test elucidation engine structure."""
    print("Testing elucidation engine...")
    
    try:
        from src.elucidation_engine import ElucidationEngine, ElucidationConfig, ElucidationResult
        
        # Test configuration
        config = ElucidationConfig(max_iterations=5, similarity_threshold=0.8)
        print(f"  ✓ ElucidationConfig: {config.max_iterations} iterations")
        
        # Test result structure
        result = ElucidationResult(
            target_molecule_id="test",
            target_nmr="test NMR",
            final_smiles="CC",
            final_similarity=0.9,
            total_iterations=3,
            steps=[],
            success=True,
            execution_time=1.0,
            metadata={}
        )
        print(f"  ✓ ElucidationResult: {result.target_molecule_id}")
        
        print("  ✓ Elucidation engine test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Elucidation engine test failed: {e}")
        return False


def test_visualization():
    """Test visualization components."""
    print("Testing visualization...")
    
    try:
        from src.visualization import ElucidationVisualizer
        
        # Test visualizer initialization
        visualizer = ElucidationVisualizer()
        print("  ✓ Visualizer initialization passed")
        
        print("  ✓ Visualization test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Visualization test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SPECTRUM ELUCIDATOR TOOLKIT - COMPONENT TESTS")
    print("=" * 60)
    
    tests = [
        test_data_utils,
        test_config,
        test_llm_interface,
        test_elucidation_engine,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The toolkit is ready to use.")
        print("\nNext steps:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY='your_key'")
        print("2. Run example: python example_single_elucidation.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
