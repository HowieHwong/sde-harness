"""Structure Generator for MatLLMSearch using SDE-harness framework"""

import json
import re
import numpy as np
import time
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path

# Initialize Weave for tracing
try:
    import weave
    weave.init("matllmsearch-generation")
except ImportError:
    pass
except Exception as e:
    pass

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

from sde_harness.core.generation import Generation
from sde_harness.core.prompt import Prompt
from .config import PROMPT_PATTERN_CSG, PROMPT_PATTERN_CSG_ZEROSHOT, PROMPT_PATTERN_CSP


class StructureGenerator:
    """Structure generator that integrates with SDE-harness Generation framework"""
    
    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 4000,
                 fmt: str = "poscar", task: str = "csg", args: Any = None):
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fmt = fmt
        self.task = task
        self.args = args
        
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent.parent 
        config_dir = project_root / "config"
        
        models_file = config_dir / "models.yaml"
        credentials_file = config_dir / "credentials.yaml"
        
        # Verify files exist
        if not models_file.exists():
            raise FileNotFoundError(
                f"Models configuration file not found: {models_file}\n"
                f"Current file: {current_file}\n"
                f"Project root: {project_root}\n"
                f"Config dir: {config_dir}"
            )
        if not credentials_file.exists():
            raise FileNotFoundError(
                f"Credentials file not found: {credentials_file}\n"
                f"Current file: {current_file}\n"
                f"Project root: {project_root}\n"
                f"Config dir: {config_dir}"
            )
        
        self.generator = Generation(
            models_file=str(models_file),
            credentials_file=str(credentials_file), 
            model_name=model
        )

        # Store generation parameters
        # Determine if this is a local model or API model
        # For OpenAI/Anthropic/etc, use max_tokens; for local models, use max_new_tokens
        is_local_model = model.startswith("local/") or not any(x in model for x in ["openai", "anthropic", "gemini", "deepseek"])
        
        if is_local_model:
            self.gen_args = {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "truncation": True
            }
        else:
            # For API models (OpenAI, Anthropic, etc.), use max_tokens
            self.gen_args = {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        
        # Initialize SDE-Harness Prompt templates
        self._init_prompt_templates()
    
    def _init_prompt_templates(self) -> None:
        """Initialize SDE-Harness Prompt templates with custom MatLLMSearch templates"""
        
        # CSG Zero-shot template
        self.csg_zeroshot_prompt = Prompt(
            custom_template=PROMPT_PATTERN_CSG_ZEROSHOT,
            default_vars={
                "fmt": self.fmt,
                "random_seed": getattr(self.args, 'seed', 42)
            }
        )
        
        # CSG template (with input structures)
        self.csg_prompt = Prompt(
            custom_template=PROMPT_PATTERN_CSG,
            default_vars={
                "fmt": self.fmt,
                "rep_size": 5  # Will be updated per call
            }
        )
        
        # CSP template
        self.csp_prompt = Prompt(
            custom_template=PROMPT_PATTERN_CSP,
            default_vars={
                "fmt": self.fmt,
                "compound": getattr(self.args, 'compound', 'Unknown'),
                "rep_size": 5  # Will be updated per call
            }
        )
    
    def generate(self, parent_structures: List[Structure], num_offspring: int = 5) -> List[Structure]:
        """Generate new structures based on parent structures"""
        
        if not parent_structures or len(parent_structures) == 0:
            # Zero-shot generation using SDE-Harness Prompt
            zeroshot_instruction = self.csg_zeroshot_prompt.build()
            instructions = [zeroshot_instruction] * self.args.population_size
        else:
            # Prepare input structures for prompting
            input_groups = self._group_structures(parent_structures)
            instructions = self._prepare_instructions_with_prompt(input_groups, num_offspring)
        
        # Generate responses using SDE-Harness Generation
        responses = []
        for i, instruction in enumerate(instructions):
            try:
                result = self.generator.generate(
                    prompt=instruction,
                    model_name=self.model,
                    **self.gen_args
                )
                response_text = result.get("text", "") if isinstance(result, dict) else str(result)
                responses.append(response_text)
                
            except Exception as e:
                print(f"Generation error for instruction {i+1}: {e}")
                if i == 0:  # Show full error for first failure
                    import traceback
                    print(f"Full traceback:")
                    traceback.print_exc()
                    print(f"Model: {self.model}")
                    print(f"Gen args: {self.gen_args}")
                responses.append("")
        
        # Parse structures from responses
        structures = self._parse_structures_from_responses(responses)
        
        return structures
    
    def _group_structures(self, structures: List[Structure]) -> List[List[Structure]]:
        """Group structures for context-aware generation"""
        parent_size = getattr(self.args, 'parent_size', 2)
        groups = []
        
        for i in range(0, len(structures), parent_size):
            group = structures[i:i + parent_size]
            groups.append(group)
        
        return groups
    
    def _prepare_instructions_with_prompt(self, input_groups: List[List[Structure]], num_offspring: int) -> List[str]:
        """Prepare instructions using SDE-Harness Prompt class"""
        instructions = []
        
        for group in input_groups:
            # Convert structures to JSON format
            input_json = self._structures_to_json(group)
            
            # Select appropriate prompt template and build instruction
            if self.task == "csg":
                # Update variables for this specific generation
                self.csg_prompt.add_vars(
                    input=input_json,
                    rep_size=num_offspring
                )
                instruction = self.csg_prompt.build()
                
            elif self.task == "csp":
                # Update variables for this specific generation
                self.csp_prompt.add_vars(
                    input=input_json,
                    rep_size=num_offspring
                )
                instruction = self.csp_prompt.build()
                
            else:
                # Default to CSG
                self.csg_prompt.add_vars(
                    input=input_json,
                    rep_size=num_offspring
                )
                instruction = self.csg_prompt.build()
            
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
    
    def _to_poscar_string(self, structure: Structure, precision: int = 8) -> str:
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
        """Parse structures from a single response - simple approach looking for format pattern"""
        structures = []
        
        # Simple pattern: look for '"fmt": "content"' in the response (where fmt is poscar or cif)
        import re
        # Pattern handles various endings: closing quote, or end of string/object
        pattern = rf'"{self.fmt}"\s*:\s*"([^"]*(?:\\.[^"]*)*?)(?:"|$|(?=\s*}}))' 
        matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            try:
                # Extract and clean the structure content
                structure_content = match.group(1)
                
                # Replace escaped newlines with actual newlines (handle both \\n and \n)
                structure_content = structure_content.replace('\\\\n', '\n')  # Double backslash first
                structure_content = structure_content.replace('\\n', '\n')    # Then single backslash
                
                # Remove any extra escaping
                structure_content = structure_content.replace('\\"', '"')
                structure_content = structure_content.replace('\\\\', '\\')   # Fix remaining double backslashes
                
                # Fix POSCAR if needed - reconstruct atom counts from actual coordinates
                if self.fmt.lower() == 'poscar':
                    structure_content = self._fix_poscar_counts(structure_content)
                
                # Try to parse as Structure
                structure = Structure.from_str(structure_content, fmt=self.fmt)
                if self._validate_structure(structure):
                    structures.append(structure)
                    
            except Exception as e:
                continue
        
        return structures
    
    def _fix_poscar_counts(self, poscar_content: str) -> str:
        """Fix POSCAR by reconstructing atom counts from actual coordinate lines"""
        lines = poscar_content.strip().split('\n')
        
        if len(lines) < 8:  # Minimum lines needed for valid POSCAR
            return poscar_content
            
        try:
            # Find coordinate section (after "direct" or "cartesian")
            coord_start_idx = -1
            for i, line in enumerate(lines):
                if line.strip().lower() in ['direct', 'cartesian']:
                    coord_start_idx = i + 1
                    break
            
            if coord_start_idx == -1:
                return poscar_content
            
            # Count actual atoms by element from coordinate lines
            element_counts = {}
            for i in range(coord_start_idx, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 4:  # x y z element
                    element = parts[3]
                    element_counts[element] = element_counts.get(element, 0) + 1
            
            if not element_counts:
                return poscar_content
            
            # Reconstruct POSCAR with correct counts
            # Update formula line (line 0)
            formula = ""
            for element, count in element_counts.items():
                formula += f"{element}{count} "
            lines[0] = formula.strip()
            
            # Update element symbols line (usually line 5)
            elements = list(element_counts.keys())
            if len(lines) > 5:
                lines[5] = " ".join(elements)
            
            # Update count line (usually line 6)  
            counts = [str(element_counts[el]) for el in elements]
            if len(lines) > 6:
                lines[6] = " ".join(counts)
            
            return '\n'.join(lines)
            
        except Exception:
            # If reconstruction fails, return original
            return poscar_content
    
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