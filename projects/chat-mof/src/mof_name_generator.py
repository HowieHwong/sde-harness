"""
MOF Name Generator - generates specific MOF names using real naming patterns
"""

from typing import Dict, Any, List
import pandas as pd
from .prompt import MOFPrompt


class MOFNameGeneratorPrompt(MOFPrompt):
    """
    Prompt class specifically for generating MOF names based on real database patterns.
    """
    
    def _load_builtin_templates(self) -> Dict[str, str]:
        """
        Define MOF name generation templates based on real database patterns.
        """
        base_templates = super()._load_builtin_templates()
        
        mof_name_templates = {
            "mof_name_generation": """You are an expert in metal-organic frameworks (MOFs) and their naming conventions.

I want you to generate specific MOF names that could potentially have high surface areas (> {target_surface_area} m²/g).

Here are examples of REAL MOF names from our database with their surface areas:
{real_mof_examples}

MOF naming patterns I've observed:
- CSD codes: 6-letter codes like VEJYIT, ETAXUT, HABRUZ, PENNON
- Literature codes: acs.cgd.6b00163_something, ja5b02777_si_008

Based on these patterns and your knowledge of high surface area MOFs, generate {num_samples} specific MOF names that could potentially exist and have surface areas > {target_surface_area} m²/g.

THINKING PROCESS (work through this step-by-step):
1. Analyze the successful examples above - what patterns do you notice in high-SA MOFs?
2. Consider which metals and frameworks typically yield high surface areas
3. Think about realistic CSD code patterns - 6 letters, often with certain letter combinations
4. Consider variations of successful MOFs (like adding 01, 02 suffixes)
5. Balance between following known patterns and exploring reasonable variations

Focus on:
1. Following realistic naming patterns (especially 6-letter CSD codes)
2. MOFs known for high porosity (UiO, ZIF, MIL, HKUST families)
3. Incorporating high-surface-area metals like Cu, Zn, Co
4. Variations of known high-performance MOFs

After your thinking process, write "BELOW ARE GENERATED MOFS:" and then provide ONLY the MOF names, one per line, without explanations or descriptions.""",

            "iterative_mof_names": """You are generating MOF names iteratively to find high surface area materials.

TARGET: Surface area > {target_surface_area} m²/g

ITERATION {current_iteration}/{max_iterations}

SUCCESSFUL MOFs FOUND SO FAR (these exist in database with high surface areas):
{successful_mofs}

FAILED MOFs TESTED (these either don't exist in database or have low surface areas):
{failed_mofs}

REAL MOF EXAMPLES from database:
{database_examples}

ANALYSIS AND LEARNING:
Based on the patterns above, analyze what makes MOF names successful vs failed:

SUCCESSFUL PATTERNS (what worked):
- Study the naming patterns of successful MOFs above
- Look for common structural features, naming conventions
- Consider metal types, framework families that succeeded

FAILED PATTERNS (what to avoid):
- Avoid naming patterns that led to failures
- Don't repeat MOF names that weren't found in database
- Learn from systematic names that didn't exist

SUCCESSFUL MOFs FOUND SO FAR:
{successful_mofs}

RECENTLY TESTED MOFs (including failures):
{failed_mofs}

REAL MOF EXAMPLES from database:
{database_examples}

Based on the patterns from successful MOFs and database examples, generate {num_samples} NEW MOF names that could have even higher surface areas.

THINKING PROCESS for iteration {current_iteration}:
1. Review what worked: Which patterns in successful MOFs above led to high surface areas?
2. Learn from failures: What patterns in failed attempts should be avoided?
3. Identify trends: Are certain prefixes, metals, or name structures more successful?
4. Strategic planning: How can you build on successes while exploring new variations?
5. Database patterns: What naming conventions from the examples seem most promising?

Key insights from successful patterns:
- Look for naming patterns that led to successful discoveries
- Consider variations of successful MOF families
- Try different CSD-style codes following successful patterns
- Explore systematic name variations

Requirements:
1. Generate completely NEW names not in the history above
2. Follow realistic MOF naming conventions
3. Focus on names likely to exist in materials databases
4. Think through your strategy before generating

After your thinking process, write "BELOW ARE GENERATED MOFS:" and then provide ONLY MOF names, one per line:"""
        }
        
        base_templates.update(mof_name_templates)
        return base_templates
    
    def build_mof_name_prompt(
        self,
        target_surface_area: float,
        num_samples: int,
        real_mof_examples: str,
        **kwargs
    ) -> str:
        """
        Build a prompt for generating specific MOF names.
        
        Args:
            target_surface_area: Target surface area threshold
            num_samples: Number of MOF names to generate
            real_mof_examples: Examples of real MOFs from database
            **kwargs: Additional template variables
            
        Returns:
            Formatted prompt string
        """
        # Switch to MOF name generation template
        original_template = self.template
        self.template = self.builtin_templates["mof_name_generation"]
        
        variables = {
            'target_surface_area': target_surface_area,
            'num_samples': num_samples,
            'real_mof_examples': real_mof_examples,
            **kwargs
        }
        
        prompt = self.build(variables)
        
        # Restore original template
        self.template = original_template
        
        return prompt
    
    def build_iterative_mof_names_prompt(
        self,
        current_iteration: int,
        max_iterations: int,
        target_surface_area: float,
        num_samples: int,
        successful_mofs: str,
        failed_mofs: str,
        database_examples: str,
        **kwargs
    ) -> str:
        """
        Build iterative prompt for MOF name generation.
        
        Args:
            current_iteration: Current iteration number
            max_iterations: Total iterations
            target_surface_area: Surface area target
            num_samples: Number of names to generate
            successful_mofs: String of successful MOFs found
            failed_mofs: String of failed MOF attempts
            database_examples: Examples from database
            **kwargs: Additional variables
            
        Returns:
            Formatted prompt string
        """
        # Switch to iterative template
        original_template = self.template
        self.template = self.builtin_templates["iterative_mof_names"]
        
        variables = {
            'current_iteration': current_iteration,
            'max_iterations': max_iterations,
            'target_surface_area': target_surface_area,
            'num_samples': num_samples,
            'successful_mofs': successful_mofs,
            'failed_mofs': failed_mofs,
            'database_examples': database_examples,
            **kwargs
        }
        
        prompt = self.build(variables)
        
        # Restore original template
        self.template = original_template
        
        return prompt
    
    def format_real_mof_examples(self, mof_data: List[Dict[str, Any]], max_examples: int = 15) -> str:
        """
        Format real MOF examples from database for prompts.
        
        Args:
            mof_data: List of MOF data from database
            max_examples: Maximum examples to include
            
        Returns:
            Formatted string of real MOF examples
        """
        if not mof_data:
            return "No examples available."
        
        examples = []
        for mof in mof_data[:max_examples]:
            name = mof.get('name', 'Unknown')
            surface_area = mof.get('Accessible Surface Area (m^2/g)', 'Unknown')
            metal = mof.get('Metal type', 'Unknown')
            examples.append(f"- {name}: {surface_area} m²/g (metal: {metal})")
        
        return "\n".join(examples)
    
    def format_mof_history(self, mof_results: List[Dict[str, Any]], max_recent: int = 20) -> str:
        """
        Format MOF testing history for prompts.
        
        Args:
            mof_results: List of MOF evaluation results
            max_recent: Maximum recent results to show
            
        Returns:
            Formatted string of testing history
        """
        if not mof_results:
            return "No testing history available."
        
        formatted = []
        for result in mof_results[-max_recent:]:
            name = result.get('mof_name', 'Unknown')
            found = result.get('found', False)
            surface_area = result.get('surface_area', None)
            
            if found and surface_area is not None:
                formatted.append(f"- {name}: {surface_area:.1f} m²/g ✓ FOUND")
            else:
                formatted.append(f"- {name}: Not found in database ✗")
        
        return "\n".join(formatted)
    
    def format_successful_mofs(self, successful_results: List[Dict[str, Any]], max_show: int = None) -> str:
        """
        Format successful MOF discoveries for prompts.
        
        Args:
            successful_results: List of successful MOF results
            max_show: Maximum successful MOFs to show
            
        Returns:
            Formatted string of successful MOFs
        """
        if not successful_results:
            return "No successful MOFs found yet."
        
        # Sort by surface area
        sorted_mofs = sorted(
            successful_results,
            key=lambda x: x.get('surface_area', 0),
            reverse=True
        )
        
        formatted = []
        # Show all successful MOFs if max_show is None, otherwise limit
        mofs_to_show = sorted_mofs if max_show is None else sorted_mofs[:max_show]
        
        for result in mofs_to_show:
            name = result.get('mof_name', 'Unknown')
            surface_area = result.get('surface_area', 0)
            formatted.append(f"- {name}: {surface_area:.1f} m²/g ⭐")
        
        return "\n".join(formatted)
    
    def format_failed_mofs(self, failed_results: List[Dict[str, Any]], max_show: int = None) -> str:
        """
        Format failed MOF attempts for prompts.
        
        Args:
            failed_results: List of failed MOF results
            max_show: Maximum failed MOFs to show
            
        Returns:
            Formatted string of failed MOFs
        """
        if not failed_results:
            return "No failed MOFs to learn from yet."
        
        formatted = []
        # Show all failed MOFs if max_show is None, otherwise limit to most recent
        mofs_to_show = failed_results if max_show is None else failed_results[-max_show:]
        
        for result in mofs_to_show:
            name = result.get('mof_name', 'Unknown')
            found = result.get('found', False)
            surface_area = result.get('surface_area', None)
            
            if found and surface_area is not None:
                # Found in database but below threshold
                formatted.append(f"- {name}: {surface_area:.1f} m²/g (found but too low)")
            else:
                # Not found in database
                formatted.append(f"- {name}: Not found in database")
        
        return "\n".join(formatted)
    
    def format_all_historical_mofs(self, all_results: List[Dict[str, Any]], max_show: int = 40) -> str:
        """
        Format all historical MOF attempts for prompts.
        
        Args:
            all_results: List of all MOF evaluation results
            max_show: Maximum MOFs to show
            
        Returns:
            Formatted string of all historical attempts
        """
        if not all_results:
            return "No historical attempts yet."
        
        formatted = []
        # Group by outcome type
        successful = []
        found_low = []  # Found but below threshold
        not_found = []
        
        for result in all_results[-max_show:]:
            name = result.get('mof_name', 'Unknown')
            found = result.get('found', False)
            surface_area = result.get('surface_area', None)
            above_threshold = result.get('above_threshold', False)
            
            if above_threshold:
                successful.append(f"✓ {name}: {surface_area:.1f} m²/g")
            elif found and surface_area is not None:
                found_low.append(f"~ {name}: {surface_area:.1f} m²/g (too low)")
            else:
                not_found.append(f"✗ {name}: Not in database")
        
        if successful:
            formatted.append("SUCCESSFUL (high surface area):")
            formatted.extend(successful[:10])  # Limit successful to 10
        
        if found_low:
            formatted.append("\nFOUND BUT LOW SURFACE AREA:")
            formatted.extend(found_low[:15])  # Show more low SA for learning
        
        if not_found:
            formatted.append("\nNOT FOUND IN DATABASE:")
            formatted.extend(not_found[:15])  # Show recent not found
        
        return "\n".join(formatted)