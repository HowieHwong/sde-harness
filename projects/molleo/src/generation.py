"""
Generation module for MolLEO that integrates with SDE harness
"""

from typing import List, Dict, Any, Optional, Union
from sde_harness.core import Generation
import json


class MolLEOGeneration(Generation):
    """
    Generation class for molecular design using LLMs.
    Integrates with the SDE harness framework.
    """
    
    def __init__(self, model_name: str = "openai/gpt-4o-2024-08-06"):
        """
        Initialize the MolLEO generation module.
        
        Args:
            model_name: Name of the model to use for generation
        """
        import os
        # Get the path to the sde-harness root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sde_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        
        models_file = os.path.join(sde_root, "models.yaml")
        credentials_file = os.path.join(sde_root, "credentials.yaml")
        
        super().__init__(
            model_name=model_name,
            models_file=models_file,
            credentials_file=credentials_file
        )
    
    def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict]] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Union[List[str], str]:
        """
        Generate molecular structures or modifications based on the prompt.
        
        Args:
            prompt: The generation prompt
            messages: Optional message history
            model_name: Optional model name override
            **kwargs: Additional generation parameters
            
        Returns:
            Generated molecules or text response
        """
        # Call parent class's generate method
        result = super().generate(
            prompt=prompt,
            messages=messages,
            model_name=model_name,
            **kwargs
        )
        
        # Extract the text from the result
        if isinstance(result, dict) and 'text' in result:
            return result['text']
        return result
    
    def parse_molecules(self, response: Union[str, List[str]]) -> List[str]:
        """
        Parse molecules from LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of SMILES strings
        """
        molecules = []
        
        # Handle list of responses
        if isinstance(response, list):
            for r in response:
                molecules.extend(self._extract_smiles(r))
        else:
            molecules.extend(self._extract_smiles(response))
        
        return molecules
    
    def _extract_smiles(self, text: str) -> List[str]:
        """
        Extract SMILES strings from text.
        
        Args:
            text: Text containing SMILES
            
        Returns:
            List of valid SMILES strings
        """
        import re
        from rdkit import Chem
        
        smiles_list = []
        
        # Try to parse as JSON first
        try:
            data = json.loads(text)
            if isinstance(data, list):
                smiles_list.extend(data)
            elif isinstance(data, dict):
                if "molecules" in data:
                    smiles_list.extend(data["molecules"])
                elif "smiles" in data:
                    if isinstance(data["smiles"], list):
                        smiles_list.extend(data["smiles"])
                    else:
                        smiles_list.append(data["smiles"])
        except:
            # If not JSON, try to extract SMILES patterns
            # Look for lines that might be SMILES
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Check if it looks like SMILES (contains typical SMILES characters)
                    if re.match(r'^[A-Za-z0-9\(\)\[\]@\+\-=#$\/\\\.]+$', line):
                        smiles_list.append(line)
        
        # Validate SMILES
        valid_smiles = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
        
        return valid_smiles