"""Structure Generator for MatLLMSearch using SDE-harness framework"""

import json
import re
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

from .llm_manager import LLMManager
from .config import PROMPT_PATTERN_CSG, PROMPT_PATTERN_CSG_ZEROSHOT, PROMPT_PATTERN_CSP


class StructureGenerator:
    """Structure generator that integrates with SDE-harness Generation framework"""
    
    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 4000,
                 tensor_parallel_size: int = 4, gpu_memory_utilization: float = 0.84,
                 fmt: str = "poscar", task: str = "csg", args: Any = None):
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fmt = fmt
        self.task = task
        self.args = args
        
        # Initialize LLM manager
        self.llm_manager = LLMManager(
            base_model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Select prompt pattern
        self.prompt_pattern = self._get_prompt_pattern()
    
    def _get_prompt_pattern(self) -> str:
        """Get appropriate prompt pattern based on task and model"""
        if "csg" in self.task:
            return PROMPT_PATTERN_CSG
        elif self.task == "csp":
            return PROMPT_PATTERN_CSP
        else:
            return PROMPT_PATTERN_CSG
    
    def generate(self, parent_structures: List[Structure], num_offspring: int = 5) -> List[Structure]:
        """Generate new structures based on parent structures"""
        
        if not parent_structures or len(parent_structures) == 0:
            # Zero-shot generation
            instructions = [PROMPT_PATTERN_CSG_ZEROSHOT.format(
                fmt=self.fmt,
                random_seed=self.args.seed
            )] * self.args.population_size
        else:
            # Prepare input structures for prompting
            input_groups = self._group_structures(parent_structures)
            instructions = self._prepare_instructions(input_groups, num_offspring)
        
        # Generate responses using LLM
        responses = self.llm_manager.generate(instructions)
        
        # Parse structures from responses
        structures = self._parse_structures_from_responses(responses)
        
        return structures
    
    def _group_structures(self, structures: List[Structure]) -> List[List[Structure]]:
        """Group structures for context-aware generation"""
        context_size = getattr(self.args, 'context_size', 5)
        groups = []
        
        for i in range(0, len(structures), context_size):
            group = structures[i:i + context_size]
            groups.append(group)
        
        return groups
    
    def _prepare_instructions(self, input_groups: List[List[Structure]], num_offspring: int) -> List[str]:
        """Prepare instructions for LLM generation"""
        instructions = []
        
        for group in input_groups:
            # Convert structures to JSON format
            input_json = self._structures_to_json(group)
            
            # Format instruction
            if self.task == "csg":
                instruction = PROMPT_PATTERN_CSG.format(
                    input=input_json,
                    rep_size=num_offspring,
                    fmt=self.fmt
                )
            elif self.task == "csp":
                instruction = PROMPT_PATTERN_CSP.format(
                    input=input_json,
                    rep_size=num_offspring,
                    fmt=self.fmt,
                    compound=getattr(self.args, 'compound', 'Unknown')
                )
            else:
                instruction = PROMPT_PATTERN_CSG.format(
                    input=input_json,
                    rep_size=num_offspring,
                    fmt=self.fmt
                )
            
            instructions.append(instruction)
        
        return instructions
    
    def _structures_to_json(self, structures: List[Structure]) -> str:
        """Convert structures to JSON format for prompting"""
        structures_dict = {}
        
        for i, structure in enumerate(structures):
            if structure is None:
                continue
                
            structure_str = self._structure_to_string(structure)
            structures_dict[str(i)] = {
                "formula": structure.composition.reduced_formula,
                self.fmt: structure_str
            }
        
        return json.dumps(structures_dict, indent=2)
    
    def _structure_to_string(self, structure: Structure) -> str:
        """Convert Structure to formatted string"""
        if self.fmt.lower() == 'poscar':
            return self._to_poscar_string(structure)
        elif self.fmt.lower() == 'cif':
            return str(structure.to(fmt='cif'))
        else:
            return str(structure.to(fmt=self.fmt))
    
    def _to_poscar_string(self, structure: Structure, precision: int = 12) -> str:
        """Convert Structure to POSCAR format string"""
        species = []
        counts = []
        current_sp = None
        count = 0
        
        for site in structure:
            if site.species_string != current_sp:
                if current_sp is not None:
                    species.append(current_sp)
                    counts.append(count)
                current_sp = site.species_string
                count = 1
            else:
                count += 1
        
        species.append(current_sp)
        counts.append(count)
        
        fmt_str = f"{{:.{precision}f}}"
        
        lines = [
            " ".join(f"{sp}{cnt}" for sp, cnt in zip(species, counts)),  # Formula line
            "1.0",  # Scale factor
            # Lattice vectors
            "\\n".join("  " + " ".join(fmt_str.format(x) for x in row) 
                     for row in structure.lattice.matrix),
            " ".join(species),  # Species symbols
            " ".join(map(str, counts)),  # Counts
            "direct",  # Coordinate type
            # Site coordinates
            "\\n".join("   " + " ".join(fmt_str.format(x) for x in site.frac_coords) + 
                     f" {site.species_string}" for site in structure)
        ]
        
        return "\\n".join(lines)
    
    def _parse_structures_from_responses(self, responses: List[str]) -> List[Structure]:
        """Parse Structure objects from LLM responses"""
        structures = []
        
        for response in responses:
            response_structures = self._parse_single_response(response)
            structures.extend(response_structures)
        
        return structures
    
    def _parse_single_response(self, response: str) -> List[Structure]:
        """Parse structures from a single response"""
        structures = []
        
        # Clean JSON response
        cleaned_response = self._clean_json_response(response)
        
        # Extract structure strings
        structure_strings = self._extract_structure_strings(cleaned_response)
        
        # Parse each structure string
        for struct_str in structure_strings:
            try:
                structure = Structure.from_str(struct_str, fmt=self.fmt)
                if self._validate_structure(structure):
                    structures.append(structure)
            except Exception as e:
                print(f"Error parsing structure: {e}")
                continue
        
        return structures
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from LLM"""
        # Remove code blocks and other formatting
        response = re.sub(r'```.*?\\n|```|\\[|\\]', '', response)
        response = re.sub(r'["""]', '"', response)
        response = re.sub(r'(\\{)(\\d+)(:)', r'\\1"\\2"\\3', response)
        
        return response.strip()
    
    def _extract_structure_strings(self, text: str) -> List[str]:
        """Extract structure strings from JSON response"""
        pattern = r'"' + self.fmt + r'":\\s*"([^"]*?)(?:"\\s*}|\\Z)'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        structure_strings = []
        for match in matches:
            struct_str = match.group(1).replace('\\\\n', '\\n').strip()
            if struct_str:
                structure_strings.append(struct_str)
        
        return structure_strings
    
    def _validate_structure(self, structure: Structure) -> bool:
        """Basic structure validation"""
        try:
            # Check if structure has valid composition
            if structure.composition.num_atoms <= 0:
                return False
            
            # Check if structure has reasonable volume
            if structure.volume <= 0 or structure.volume >= 30 * structure.composition.num_atoms:
                return False
            
            # Check if structure is 3D periodic
            if not structure.is_3d_periodic:
                return False
            
            return True
            
        except Exception:
            return False