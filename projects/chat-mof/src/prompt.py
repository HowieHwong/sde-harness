"""
MOF Prompt class - manages prompts for MOF name generation
"""

import sys
import os
from typing import Dict, Any, Optional, List

# Add sde_harness to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from sde_harness.core import Prompt


class MOFPrompt(Prompt):
    """
    MOF-specific prompt management class.
    Handles prompt templates for generating MOF names with surface area optimization.
    """
    
    def __init__(self, template_name: Optional[str] = None, custom_template: Optional[str] = None):
        """
        Initialize MOF prompt with built-in templates for MOF generation.
        
        Args:
            template_name: Built-in template name ('mof_generation', 'mof_iterative', etc.)
            custom_template: Custom prompt template string
        """
        # Use default MOF generation template if none specified
        if template_name is None and custom_template is None:
            template_name = "mof_generation"
            
        super().__init__(template_name=template_name, custom_template=custom_template)
    
    def _load_builtin_templates(self) -> Dict[str, str]:
        """
        Define MOF-specific prompt templates.
        """
        base_templates = super()._load_builtin_templates()
        
        mof_templates = {
            "mof_generation": """You are an expert in metal-organic frameworks (MOFs) and materials science.

I want you to generate names of MOFs that might have high surface areas (> {target_surface_area} m²/g).

Here are some example MOFs with their surface areas:
{examples}

Based on these patterns and your knowledge of MOF chemistry, generate {num_samples} new MOF names that could potentially have surface areas greater than {target_surface_area} m²/g.

Focus on:
1. MOFs with high porosity and large pore volumes
2. Frameworks with lightweight linkers and high void fractions  
3. MOFs known for gas storage applications
4. Structures with interconnected pore networks

Please provide ONLY the MOF names, one per line, without explanations.""",

            "mof_iterative": """You are an expert in metal-organic frameworks (MOFs) working on an iterative optimization task.

OBJECTIVE: Generate MOF names that have surface areas > {target_surface_area} m²/g

ITERATION {current_iteration}/{max_iterations}

BEST MOFs FOUND SO FAR:
{best_mofs}

RECENTLY TESTED MOFs:
{recent_history}

Based on the successful patterns above and your chemistry knowledge, generate {num_samples} NEW MOF names that could have even higher surface areas.

Key insights from successful MOFs:
- Look for patterns in metal nodes, linkers, and topologies that led to high surface areas
- Consider structural features that maximize porosity
- Build upon successful frameworks while exploring variations

Requirements:
1. Generate completely NEW names not in the history above
2. Focus on MOFs likely to exceed {target_surface_area} m²/g surface area
3. Provide ONLY MOF names, one per line""",

            "mof_exploration": """You are exploring the chemical space of metal-organic frameworks (MOFs).

Generate {num_samples} diverse MOF names that could potentially have high surface areas.

Consider various:
- Metal nodes: Zn, Cu, Zr, Al, Cr, Fe, Co, Ni, etc.
- Organic linkers: carboxylates, imidazolates, phosphonates, etc.  
- Topologies: pcu, soc, rho, fcu, etc.
- Common MOF families: UiO, MIL, ZIF, HKUST, etc.

Provide ONLY the MOF names, one per line."""
        }
        
        # Merge base templates with MOF-specific ones
        base_templates.update(mof_templates)
        return base_templates
    
    def build_generation_prompt(
        self,
        target_surface_area: float = 1000.0,
        num_samples: int = 10,
        examples: str = "",
        **kwargs
    ) -> str:
        """
        Build a prompt for initial MOF generation.
        
        Args:
            target_surface_area: Minimum surface area target (m²/g)
            num_samples: Number of MOF names to generate
            examples: Example MOFs with surface areas
            **kwargs: Additional template variables
            
        Returns:
            Formatted prompt string
        """
        variables = {
            'target_surface_area': target_surface_area,
            'num_samples': num_samples,
            'examples': examples,
            **kwargs
        }
        return self.build(variables)
    
    def build_iterative_prompt(
        self,
        current_iteration: int,
        max_iterations: int,
        target_surface_area: float,
        num_samples: int,
        best_mofs: str,
        recent_history: str,
        **kwargs
    ) -> str:
        """
        Build a prompt for iterative MOF optimization.
        
        Args:
            current_iteration: Current iteration number
            max_iterations: Total number of iterations
            target_surface_area: Surface area target
            num_samples: Number of new MOFs to generate
            best_mofs: String of best MOFs found so far
            recent_history: String of recently tested MOFs
            **kwargs: Additional template variables
            
        Returns:
            Formatted prompt string
        """
        # Switch to iterative template temporarily
        original_template = self.template
        self.template = self.builtin_templates["mof_iterative"]
        
        variables = {
            'current_iteration': current_iteration,
            'max_iterations': max_iterations,
            'target_surface_area': target_surface_area,
            'num_samples': num_samples,
            'best_mofs': best_mofs,
            'recent_history': recent_history,
            **kwargs
        }
        
        prompt = self.build(variables)
        
        # Restore original template
        self.template = original_template
        
        return prompt
    
    def format_mof_examples(self, mof_data: List[Dict[str, Any]]) -> str:
        """
        Format MOF examples for inclusion in prompts.
        
        Args:
            mof_data: List of dicts with 'name' and 'surface_area' keys
            
        Returns:
            Formatted string of MOF examples
        """
        if not mof_data:
            return "No examples available."
            
        examples = []
        for mof in mof_data:
            name = mof.get('name', 'Unknown')
            surface_area = mof.get('surface_area', 'Unknown')
            examples.append(f"- {name}: {surface_area} m²/g")
            
        return "\n".join(examples)