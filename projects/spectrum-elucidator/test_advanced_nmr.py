#!/usr/bin/env python3
"""
Test script to verify the advanced NMR similarity calculation works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_utils import NMRProcessor


def test_advanced_nmr_similarity():
    """Test the advanced NMR similarity calculation."""
    
    print("Testing Advanced NMR Similarity Calculation...")
    
    # Test H-NMR data (from your actual data format)
    test_h_nmr1 = "NMR: 0.87 (3H, t,  J  = 6.5 Hz), 1.30 (2H, tq,  J  = 6.6, 6.5 Hz), 1.49 (2H, tt,  J  = 7.5, 6.6 Hz)"
    test_h_nmr2 = "NMR: 1.90 (2H, t, J = 6.5 Hz), 1.35 (2H, m), 7.20 (5H, m)"
    
    print(f"\nTest H-NMR 1: {test_h_nmr1}")
    print(f"Test H-NMR 2: {test_h_nmr2}")
    
    # Test peak splitting
    peaks1 = NMRProcessor.split_nmr_peaks(test_h_nmr1)
    peaks2 = NMRProcessor.split_nmr_peaks(test_h_nmr2)
    
    print(f"\nSplit peaks 1: {peaks1}")
    print(f"Split peaks 2: {peaks2}")
    
    # Test data extraction
    data1 = []
    for peak in peaks1:
        data1.extend(NMRProcessor.extract_1h_data(peak))
    
    data2 = []
    for peak in peaks2:
        data2.extend(NMRProcessor.extract_1h_data(peak))
    
    print(f"\nExtracted H-NMR data 1: {data1}")
    print(f"Extracted H-NMR data 2: {data2}")
    
    # Test similarity calculation
    similarity = NMRProcessor.calculate_nmr_similarity_advanced(
        test_h_nmr1, test_h_nmr2, "H", tolerance=0.1
    )
    print(f"\nH-NMR similarity (tolerance=0.1): {similarity:.3f}")
    
    # Test C-NMR data
    test_c_nmr1 = "NMR: 14.0 (1C, s), 22.6 (1C, s), 32.8 (1C, s), 35.5 (1C, s), 127.8 (1C, s), 128.4 (2C, s)"
    test_c_nmr2 = "NMR: 14.2 (1C, s), 22.8 (1C, s), 33.0 (1C, s), 35.7 (1C, s), 127.9 (1C, s), 128.5 (2C, s)"
    
    print(f"\nTest C-NMR 1: {test_c_nmr1}")
    print(f"Test C-NMR 2: {test_c_nmr2}")
    
    # Test C-NMR peak splitting
    c_peaks1 = NMRProcessor.split_nmr_peaks(test_c_nmr1)
    c_peaks2 = NMRProcessor.split_nmr_peaks(test_c_nmr2)
    
    print(f"\nSplit C-NMR peaks 1: {c_peaks1}")
    print(f"Split C-NMR peaks 2: {c_peaks2}")
    
    # Test C-NMR data extraction
    c_data1 = []
    for peak in c_peaks1:
        c_data1.extend(NMRProcessor.extract_13c_data(peak))
    
    c_data2 = []
    for peak in c_peaks2:
        c_data2.extend(NMRProcessor.extract_13c_data(peak))
    
    print(f"\nExtracted C-NMR data 1: {c_data1}")
    print(f"Extracted C-NMR data 2: {c_data2}")
    
    # Test C-NMR similarity
    c_similarity = NMRProcessor.calculate_nmr_similarity_advanced(
        test_c_nmr1, test_c_nmr2, "C", tolerance=0.1
    )
    print(f"\nC-NMR similarity (tolerance=0.1): {c_similarity:.3f}")
    
    # Test auto-detection
    auto_similarity = NMRProcessor.calculate_nmr_similarity(test_h_nmr1, test_h_nmr2, tolerance=0.1)
    print(f"\nAuto-detected similarity: {auto_similarity:.3f}")
    
    # Test identical NMR (should give similarity = 1.0)
    identical_similarity = NMRProcessor.calculate_nmr_similarity(test_h_nmr1, test_h_nmr1, tolerance=0.1)
    print(f"\nIdentical H-NMR similarity: {identical_similarity:.3f} (should be 1.0)")
    
    # Test with your actual data format
    actual_h_nmr = "NMR: 0.87 (3H, t,  J  = 6.5 Hz), 1.30 (2H, tq,  J  = 6.6, 6.5 Hz), 1.49 (2H, tt,  J  = 7.5, 6.6 Hz), 2.35 (2H, t,  J  = 7.5 Hz), 7.10-7.32 (5H, m)"
    actual_c_nmr = "NMR: 14.0 (1C, s), 22.6 (1C, s), 32.8 (1C, s), 35.5 (1C, s), 127.8 (1C, s), 128.4 (2C, s), 128.8 (2C, s), 140.6 (1C, s)"
    
    print(f"\nActual H-NMR: {actual_h_nmr}")
    print(f"Actual C-NMR: {actual_c_nmr}")
    
    # Test with actual data
    actual_h_peaks = NMRProcessor.split_nmr_peaks(actual_h_nmr)
    actual_c_peaks = NMRProcessor.split_nmr_peaks(actual_c_nmr)
    
    print(f"\nActual H-NMR peaks: {actual_h_peaks}")
    print(f"Actual C-NMR peaks: {actual_c_peaks}")
    
    # Test similarity with actual data
    actual_h_similarity = NMRProcessor.calculate_nmr_similarity_advanced(
        actual_h_nmr, actual_h_nmr, "H", tolerance=0.05
    )
    actual_c_similarity = NMRProcessor.calculate_nmr_similarity_advanced(
        actual_c_nmr, actual_c_nmr, "C", tolerance=0.05
    )
    
    print(f"\nActual H-NMR self-similarity: {actual_h_similarity:.3f} (should be 1.0)")
    print(f"Actual C-NMR self-similarity: {actual_c_similarity:.3f} (should be 1.0)")


if __name__ == "__main__":
    test_advanced_nmr_similarity()
