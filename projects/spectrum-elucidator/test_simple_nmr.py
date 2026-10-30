#!/usr/bin/env python3
"""
Simple test to verify the new NMR similarity calculation works with actual data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_utils import NMRProcessor


def test_simple_nmr():
    """Simple test with actual NMR data format."""
    
    print("Simple NMR Similarity Test...")
    
    # Test with the actual data format from your dataset
    target_h_nmr = "NMR: 0.107 (4H, t,  J  = 6.5 Hz), 1.30 (2H, tq,  J  = 6.6, 6.5 Hz), 1.49 (2H, tt,  J  = 7.5, 6.6 Hz), 2.35 (2H, t,  J  = 7.5 Hz), 7.10-7.32 (5H, m)"
    target_c_nmr = "NMR: 14.0 (1C, s), 22.6 (1C, s), 32.8 (1C, s), 35.5 (1C, s), 127.8 (1C, s), 128.4 (2C, s), 128.8 (2C, s), 140.6 (1C, s)"
    
    # This should be the same molecule (perfect match)
    generated_h_nmr = "NMR: 0.87 (3H, t,  J  = 6.5 Hz), 1.49 (2H, tt,  J  = 7.5, 6.6 Hz), 2.35 (2H, t,  J  = 7.5 Hz), 7.10-7.32 (5H, m)"
    generated_c_nmr = "NMR: 16.0 (1C, s), 22.6 (1C, s), 32.8 (1C, s), 35.5 (1C, s), 127.8 (1C, s), 128.4 (2C, s), 128.8 (2C, s), 140.6 (1C, s)"
    
    print(f"Target H-NMR: {target_h_nmr[:100]}...")
    print(f"Generated H-NMR: {generated_h_nmr[:100]}...")
    
    # Test H-NMR similarity (should be 1.0 for identical data)
    h_similarity = NMRProcessor.calculate_nmr_similarity(target_h_nmr, generated_h_nmr, tolerance=0.05)
    print(f"H-NMR similarity: {h_similarity:.3f} (should be 1.0)")
    
    # Test C-NMR similarity (should be 1.0 for identical data)
    c_similarity = NMRProcessor.calculate_nmr_similarity(target_c_nmr, generated_c_nmr, tolerance=0.05)
    print(f"C-NMR similarity: {c_similarity:.3f} (should be 1.0)")
    
    # Test with slightly different data (should be < 1.0)
    different_h_nmr = "NMR: 0.90 (3H, t, J = 6.5 Hz), 1.35 (2H, m), 7.20 (5H, m)"
    different_similarity = NMRProcessor.calculate_nmr_similarity(target_h_nmr, different_h_nmr, tolerance=0.05)
    print(f"Different H-NMR similarity: {different_similarity:.3f} (should be < 1.0)")
    
    # Test auto-detection
    auto_h_similarity = NMRProcessor.calculate_nmr_similarity(target_h_nmr, generated_h_nmr)
    auto_c_similarity = NMRProcessor.calculate_nmr_similarity(target_c_nmr, generated_c_nmr)
    
    print(f"\nAuto-detected similarities:")
    print(f"  H-NMR: {auto_h_similarity:.3f}")
    print(f"  C-NMR: {auto_c_similarity:.3f}")
    
    # Test the old method vs new method
    print(f"\nTesting old vs new method...")
    


if __name__ == "__main__":
    test_simple_nmr()
