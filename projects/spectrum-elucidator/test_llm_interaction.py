#!/usr/bin/env python3
"""
Test script to debug LLM interaction and SMILES extraction.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.llm_interface import LLMInterface


def test_llm_interaction():
    """Test LLM interaction and SMILES extraction."""
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        return
    
    print("Testing LLM interaction and SMILES extraction...")
    
    # Initialize LLM interface
    llm = LLMInterface(api_key=api_key, model="gpt-4")
    
    # Test prompt
    test_nmr = "NMR: 0.87 (3H, t, J = 6.5 Hz), 1.30 (2H, m), 7.10-7.32 (5H, m)"
    
    prompt = llm.create_elucidation_prompt(
        target_nmr=test_nmr,
        iteration=1,
        history=[],
        target_molecule_info={}
    )
    
    print(f"\nGenerated prompt:\n{prompt}")
    
    # Query LLM
    print("\nQuerying LLM...")
    response = llm.query_llm(prompt, temperature=0.3)
    
    print(f"\nLLM response:\n{response}")
    
    # Extract SMILES
    print("\nExtracting SMILES...")
    smiles = llm.extract_smiles_from_response(response)
    
    if smiles:
        print(f"Extracted SMILES: {smiles}")
        
        # Validate SMILES
        is_valid = llm.validate_smiles(smiles)
        print(f"SMILES validation: {'Valid' if is_valid else 'Invalid'}")
    else:
        print("No SMILES extracted!")
        
        # Try fallback
        print("Attempting fallback extraction...")
        fallback = llm._extract_fallback_smiles(response)
        if fallback:
            print(f"Fallback SMILES: {fallback}")
        else:
            print("No fallback SMILES generated")


if __name__ == "__main__":
    test_llm_interaction()
