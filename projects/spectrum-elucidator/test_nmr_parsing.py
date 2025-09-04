#!/usr/bin/env python3
"""
Test script to verify NMR parsing works correctly with the actual data format.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_utils import NMRProcessor


def test_nmr_parsing():
    """Test NMR parsing with actual data format."""
    
    print("Testing NMR parsing with actual data format...")
    
    # Test H-NMR format from your data
    test_h_nmr = "NMR: 0.87 (3H, t,  J  = 6.5 Hz), 1.30 (2H, tq,  J  = 6.6, 6.5 Hz), 1.49 (2H, tt,  J  = 7.5, 6.6 Hz)"
    
    print(f"\nTest H-NMR: {test_h_nmr}")
    peaks = NMRProcessor.parse_nmr_peaks(test_h_nmr)
    print(f"Parsed {len(peaks)} peaks:")
    
    for i, peak in enumerate(peaks):
        print(f"  Peak {i+1}: {peak}")
    
    # Test C-NMR format
    test_c_nmr = "NMR: 14.0 (1C, s), 22.6 (1C, s), 32.8 (1C, s), 35.5 (1C, s), 127.8 (1C, s), 128.4 (2C, s)"
    
    print(f"\nTest C-NMR: {test_c_nmr}")
    peaks = NMRProcessor.parse_nmr_peaks(test_c_nmr)
    print(f"Parsed {len(peaks)} peaks:")
    
    for i, peak in enumerate(peaks):
        print(f"  Peak {i+1}: {peak}")
    
    # Test similarity calculation
    print(f"\nTesting similarity calculation...")
    
    # Same NMR should have similarity = 1.0
    similarity = NMRProcessor.calculate_nmr_similarity(test_h_nmr, test_h_nmr, tolerance=0.1)
    print(f"Same H-NMR similarity: {similarity:.3f} (should be 1.0)")
    
    # Different NMR should have lower similarity
    different_nmr = "NMR: 0.90 (3H, t, J = 6.5 Hz), 1.35 (2H, m), 7.20 (5H, m)"
    similarity = NMRProcessor.calculate_nmr_similarity(test_h_nmr, different_nmr, tolerance=0.1)
    print(f"Different H-NMR similarity: {similarity:.3f} (should be < 1.0)")
    
    # Test with tolerance
    similarity = NMRProcessor.calculate_nmr_similarity(test_h_nmr, different_nmr, tolerance=0.5)
    print(f"Different H-NMR similarity (tolerance=0.5): {similarity:.3f}")


if __name__ == "__main__":
    test_nmr_parsing()
