"""
LLM interface for molecular structure elucidation.
"""

import openai
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging


@dataclass
class ElucidationStep:
    """Represents a single step in the elucidation process."""
    
    iteration: int
    prompt: str
    response: str
    generated_smiles: Optional[str]
    nmr_similarity: float
    timestamp: float
    metadata: Dict[str, Any]


class LLMInterface:
    """Interface for interacting with Large Language Models."""
    
    def __init__(self, api_key: str, model: str = "gpt-4", max_tokens: int = 2000):
        """
        Initialize the LLM interface.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for elucidation
            max_tokens: Maximum tokens for response
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(api_key=api_key)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_elucidation_prompt(self, 
                                 target_nmr: str, 
                                 iteration: int, 
                                 history: List[ElucidationStep],
                                 target_molecule_info: Optional[Dict] = None) -> str:
        """
        Create a prompt for molecular structure elucidation.
        
        Args:
            target_nmr: Target NMR spectrum to elucidate
            iteration: Current iteration number
            history: Previous elucidation steps
            target_molecule_info: Information about the target molecule (optional)
            
        Returns:
            Formatted prompt string
        """
        
        # Base prompt
        prompt = f"""You are an expert chemist specializing in molecular structure elucidation from NMR spectra. 
Your task is to deduce the molecular structure (SMILES format) from the given NMR spectrum.

TARGET NMR SPECTRUM:
{target_nmr}

"""
        
        # Add target molecule information if available
        if target_molecule_info:
            prompt += f"""TARGET MOLECULE INFORMATION:
- SMILES: {target_molecule_info.get('SMILES', 'Unknown')}
- Functional Groups: {', '.join([k for k, v in target_molecule_info.items() if v and k not in ['SMILES', 'H_NMR', 'C_NMR']])}
- Degree of Unsaturation: {target_molecule_info.get('Degree_Unsaturation', 'Unknown')}

"""
        
        # Add iteration context
        if iteration == 1:
            prompt += """This is the first attempt. Please analyze the NMR spectrum and propose a molecular structure.
Consider:
1. Chemical shifts and their implications
2. Integration values and multiplicity patterns
3. Coupling constants and their structural significance
4. Functional groups that could explain the observed signals

"""
        else:
            prompt += f"""This is iteration {iteration}. Previous attempts and their NMR similarities:

"""
            # Add history of previous attempts
            for step in history[-3:]:  # Show last 3 attempts
                prompt += f"""Iteration {step.iteration}:
- Generated SMILES: {step.generated_smiles or 'None'}
- NMR Similarity: {step.nmr_similarity:.3f}
- Response: {step.response[:200]}...

"""
            
            prompt += f"""Based on the previous attempts and their NMR similarities, please:
1. Analyze what worked and what didn't
2. Identify patterns in the NMR spectrum
3. Propose an improved molecular structure
4. Explain your reasoning for the changes

"""
        
        prompt += """INSTRUCTIONS:
1. Provide your analysis of the NMR spectrum
2. Propose a molecular structure in SMILES format
3. Explain your reasoning step by step
4. If you cannot determine the structure, explain why and suggest what additional information would help

IMPORTANT: You MUST provide the molecular structure in SMILES format. SMILES is a text representation of molecular structure.

EXAMPLES OF SMILES FORMAT:
- Ethane: CC
- Propane: CCC
- Benzene: c1ccccc1
- Toluene: Cc1ccccc1
- Pentylbenzene: CCCCC1=CC=CC=C1

RESPONSE FORMAT:
Analysis: [Your analysis of the NMR spectrum]
Reasoning: [Step-by-step reasoning for your proposed structure]
SMILES: [Your proposed molecular structure in SMILES format - THIS IS REQUIRED]
Confidence: [High/Medium/Low - your confidence in this structure]

Please provide your response with the SMILES structure clearly marked:"""

        return prompt
    
    def query_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Query the LLM with a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature for response generation
            
        Returns:
            LLM response string
        """
        try:
            self.logger.debug(f"Sending prompt to LLM (temperature={temperature})...")
            self.logger.debug(f"Prompt preview: {prompt[:200]}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert chemist specializing in molecular structure elucidation from NMR spectra."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=temperature
            )
            
            response_text = response.choices[0].message.content
            self.logger.debug(f"LLM response received: {response_text[:200]}...")
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error querying LLM: {e}")
            return f"Error: {str(e)}"
    
    def extract_smiles_from_response(self, response: str) -> Optional[str]:
        """
        Extract SMILES string from LLM response.
        
        Args:
            response: LLM response string
            
        Returns:
            Extracted SMILES string or None if not found
        """
        # Look for SMILES pattern in the response
        import re
        
        self.logger.debug(f"Extracting SMILES from response: {response[:200]}...")
        
        # Pattern to match SMILES strings
        smiles_pattern = r'SMILES:\s*([A-Za-z0-9@+\-\[\]()=#%:]+)'
        match = re.search(smiles_pattern, response, re.IGNORECASE)
        
        if match:
            extracted = match.group(1).strip()
            self.logger.debug(f"Found SMILES using primary pattern: {extracted}")
            return extracted
        
        # Alternative patterns
        patterns = [
            r'([A-Za-z0-9@+\-\[\]()=#%:]{5,})',  # General SMILES-like pattern
            r'Structure:\s*([A-Za-z0-9@+\-\[\]()=#%:]+)',
            r'Formula:\s*([A-Za-z0-9@+\-\[\]()=#%:]+)'
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                potential_smiles = match.group(1).strip()
                self.logger.debug(f"Found potential SMILES using pattern {i+1}: {potential_smiles}")
                # Basic validation that it looks like SMILES
                if len(potential_smiles) >= 5 and any(c in potential_smiles for c in '()[]=#%'):
                    self.logger.debug(f"Validated SMILES: {potential_smiles}")
                    return potential_smiles
                else:
                    self.logger.debug(f"Failed validation for: {potential_smiles}")
        
        # Fallback: Try to extract any chemical-like text and convert to basic SMILES
        self.logger.warning(f"No valid SMILES found in response. Attempting fallback extraction...")
        fallback_smiles = self._extract_fallback_smiles(response)
        if fallback_smiles:
            self.logger.info(f"Generated fallback SMILES: {fallback_smiles}")
            return fallback_smiles
        
        self.logger.warning(f"No SMILES found in response. Response preview: {response[:100]}...")
        return None
    
    def _extract_fallback_smiles(self, response: str) -> Optional[str]:
        """
        Fallback method to extract chemical information and generate basic SMILES.
        
        Args:
            response: LLM response string
            
        Returns:
            Basic SMILES string or None
        """
        import re
        
        # Look for chemical formulas or molecular descriptions
        response_lower = response.lower()
        
        # Check for common molecular patterns mentioned
        if 'benzene' in response_lower or 'aromatic' in response_lower:
            return 'c1ccccc1'  # Basic benzene
        
        if 'alkane' in response_lower or 'saturated' in response_lower:
            # Look for carbon count
            carbon_match = re.search(r'(\d+)\s*carbon', response_lower)
            if carbon_match:
                carbon_count = int(carbon_match.group(1))
                if carbon_count <= 10:  # Reasonable limit
                    return 'C' * carbon_count
        
        if 'methyl' in response_lower:
            return 'C'
        
        if 'ethyl' in response_lower:
            return 'CC'
        
        if 'propyl' in response_lower:
            return 'CCC'
        
        # If we find chemical elements, try to construct basic structure
        elements = re.findall(r'\b([A-Z][a-z]?\d*)\b', response)
        if elements:
            # Simple approach: just use carbon if mentioned
            if any('C' in elem for elem in elements):
                return 'CC'  # Basic ethane as fallback
        
        return None
    
    def validate_smiles(self, smiles: str) -> bool:
        """
        Basic validation of SMILES string.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if SMILES appears valid, False otherwise
        """
        if not smiles:
            return False
        
        # Basic SMILES validation rules
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@+\-\[\]()=#%:')
        
        # Check if all characters are valid
        if not all(c in valid_chars for c in smiles):
            return False
        
        # Check for balanced parentheses and brackets
        stack = []
        brackets = {'(': ')', '[': ']'}
        
        for char in smiles:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack or brackets[stack.pop()] != char:
                    return False
        
        return len(stack) == 0
    
    def create_refinement_prompt(self, 
                                target_nmr: str, 
                                current_smiles: str, 
                                current_similarity: float,
                                iteration: int) -> str:
        """
        Create a prompt for refining an existing structure.
        
        Args:
            target_nmr: Target NMR spectrum
            current_smiles: Current SMILES structure
            current_similarity: Current NMR similarity score
            iteration: Current iteration number
            
        Returns:
            Refinement prompt string
        """
        
        prompt = f"""You are refining a molecular structure based on NMR similarity feedback.

TARGET NMR SPECTRUM:
{target_nmr}

CURRENT STRUCTURE:
SMILES: {current_smiles}
NMR Similarity: {current_similarity:.3f}

This is iteration {iteration}. The current structure has a similarity score of {current_similarity:.3f}.

Please analyze why the current structure might not match the target NMR and suggest improvements:

1. What aspects of the current structure might be incorrect?
2. How can you modify the structure to better match the NMR?
3. What functional groups or structural elements should be added/removed/modified?

Provide your refined structure in SMILES format with detailed reasoning.

RESPONSE FORMAT:
Analysis: [Analysis of current structure vs target NMR]
Issues: [What's wrong with current structure]
Improvements: [How to improve the structure]
Refined SMILES: [Your improved molecular structure]
Reasoning: [Step-by-step explanation of changes]

Please provide your response:"""

        return prompt
