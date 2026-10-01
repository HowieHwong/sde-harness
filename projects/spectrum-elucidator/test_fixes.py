#!/usr/bin/env python3
"""
Simple test to verify the fixes work correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_utils import NMRProcessor
from src.elucidation_engine import ElucidationEngine


def test_fixes():
    """Test that the fixes work correctly."""
    
    print("Testing Fixes...")
    
    # Test 1: NMR similarity calculation
    print("\n1. Testing NMR similarity calculation...")
    
    test_nmr = "NMR: 0.87 (3H, t,  J  = 6.5 Hz), 1.30 (2H, tq,  J  = 6.6, 6.5 Hz)"
    
    # This should work without errors
    try:
        similarity = NMRProcessor.calculate_nmr_similarity(test_nmr, test_nmr, tolerance=0.05)
        print(f"✅ NMR similarity calculation works: {similarity:.3f}")
    except Exception as e:
        print(f"❌ NMR similarity calculation failed: {e}")
    
    # Test 2: Summary generation
    print("\n2. Testing summary generation...")
    
    # Create a mock result for testing
    from src.llm_interface import ElucidationStep
    from src.elucidation_engine import ElucidationResult
    from dataclasses import dataclass
    
    # Mock step
    step = ElucidationStep(
        iteration=1,
        prompt="test",
        response="test",
        generated_smiles="CC",
        nmr_similarity=0.5,
        timestamp=1234567890,
        metadata={'test': 'data'}
    )
    
    # Mock result
    result = ElucidationResult(
        target_molecule_id="test",
        target_nmr="test",
        target_c_nmr="test",
        final_smiles="CC",
        final_similarity=0.5,
        total_iterations=1,
        steps=[step],
        success=True,
        execution_time=1.0,
        metadata={'test': 'data'}
    )
    
    # Test summary generation
    try:
        # This should work without the KeyError
        summary = ElucidationEngine.get_elucidation_summary(None, result)
        print(f"✅ Summary generation works")
        print(f"   best_similarity: {summary.get('best_similarity', 'NOT_FOUND')}")
        print(f"   final_similarity: {summary.get('final_similarity', 'NOT_FOUND')}")
    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    test_fixes()
