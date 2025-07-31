"""Tool functions for BioDiscoveryAgent."""
import os
import re
import json
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from .gene_utils import *
from .llm_interface import complete_text, complete_text_claude


class BioDiscoveryTools:
    """Collection of tools for biological discovery."""
    
    @staticmethod
    def parse_action_input(s: str, entries: List[str]) -> List[str]:
        """Parse action input string."""
        s = s.split("{")[1].split("}")[0].strip()
        pattern = ""
        for e in entries:
            pattern += f'"{e}":([\s\S]*),\s*'
        pattern = pattern[:-4]
        result = re.search(pattern, s, re.MULTILINE)
        if result is None:
            raise Exception("Invalid: " + s)
        return [r.strip().strip('\"') for r in result.groups()]
    
    @staticmethod
    def research_log(action_input: str, folder_name: str = ".") -> str:
        """Write to research log."""
        try:
            content, = BioDiscoveryTools.parse_action_input(action_input, ["content"])
        except:
            return "Invalid action input"
        
        log_path = os.path.join(folder_name, "research_log.log")
        with open(log_path, "a") as f:
            f.write(content + "\n")
        
        return open(log_path).read()
    
    @staticmethod
    def literature_search(gene_name: str, research_problem: str, model: str = "anthropic/claude-3-5-sonnet-20240620") -> str:
        """Perform literature search for a gene."""
        # Import get_lit_review function
        try:
            from ..tools.get_lit_review import get_lit_review
            prompt = f"Research problem: {research_problem}\n\nProvide a literature review for the gene {gene_name} in the context of this research."
            return get_lit_review(prompt, model, max_number=5)
        except ImportError:
            # Fallback implementation
            return f"Literature review for {gene_name}: [Literature search not available]"
    
    @staticmethod
    def gene_search(gene_name: str, csv_path: str, k: int = 10) -> List[str]:
        """Search for similar genes based on features."""
        try:
            # Import from achilles if needed
            from ..tools.achilles import download_csv
            
            # Ensure CSV exists
            csv_file = os.path.join(csv_path, "achilles.csv")
            if not os.path.exists(csv_file):
                download_csv(csv_path)
            
            # Load data
            df = pd.read_csv(csv_file)
            
            # Find similar genes (simplified version)
            if gene_name in df.columns:
                gene_data = df[gene_name]
                correlations = {}
                
                for col in df.columns:
                    if col != gene_name and col != 'Unnamed: 0':
                        try:
                            corr = gene_data.corr(df[col])
                            if not np.isnan(corr):
                                correlations[col] = abs(corr)
                        except:
                            continue
                
                # Sort by correlation
                sorted_genes = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                return [gene for gene, _ in sorted_genes[:k]]
            else:
                return []
        except Exception as e:
            print(f"Error in gene search: {e}")
            return []
    
    @staticmethod
    def get_correlated_genes(gene_name: str, dataset: str, k: int = 10) -> List[str]:
        """Get top k correlated genes."""
        try:
            # This would use the actual correlation data
            # For now, returning placeholder
            return [f"CORR_{gene_name}_{i}" for i in range(k)]
        except Exception as e:
            print(f"Error getting correlated genes: {e}")
            return []
    
    @staticmethod
    def get_active_tissues(gene_name: str, k: int = 10) -> List[str]:
        """Get top k tissues where gene is active."""
        try:
            # This would use actual RNA expression data
            # For now, returning placeholder
            tissues = ["Brain", "Liver", "Heart", "Kidney", "Lung", 
                      "Spleen", "Muscle", "Skin", "Intestine", "Pancreas"]
            return tissues[:k]
        except Exception as e:
            print(f"Error getting active tissues: {e}")
            return []
    
    @staticmethod
    def get_reactome_pathways(gene_name: str) -> List[str]:
        """Get Reactome pathways for a gene."""
        try:
            # This would query Reactome database
            # For now, returning placeholder
            return [f"Pathway_{gene_name}_1", f"Pathway_{gene_name}_2"]
        except Exception as e:
            print(f"Error getting pathways: {e}")
            return []
    
    
    @staticmethod
    def critique_solution(solution: str, research_problem: str, model: str) -> str:
        """Critique a proposed solution using LLM."""
        critique_prompt = f"""You are a critical but constructive scientific reviewer.

Research Problem: {research_problem}

Proposed Solution:
{solution}

Please provide a brief critique of this solution, highlighting:
1. Strengths of the approach
2. Potential weaknesses or concerns
3. Suggestions for improvement

Keep your critique concise and actionable."""
        
        return complete_text(critique_prompt, model=model, temperature=0.1)


# Tool registry for backward compatibility
ALL_TOOLS = {
    "Research Log": {
        "function": BioDiscoveryTools.research_log,
        "description": "Log research findings and observations"
    },
    "Literature Search": {
        "function": BioDiscoveryTools.literature_search,
        "description": "Search literature for gene information"
    },
    "Gene Search": {
        "function": BioDiscoveryTools.gene_search,
        "description": "Find similar genes based on features"
    },
    "Correlated Genes": {
        "function": BioDiscoveryTools.get_correlated_genes,
        "description": "Find correlated genes"
    },
    "Active Tissues": {
        "function": BioDiscoveryTools.get_active_tissues,
        "description": "Find tissues where gene is active"
    },
    "Reactome Pathways": {
        "function": BioDiscoveryTools.get_reactome_pathways,
        "description": "Find biological pathways for gene"
    }
}


def agent_loop(*args, **kwargs):
    """Placeholder for agent loop - will be implemented in modes."""
    raise NotImplementedError("agent_loop should be called from modes module")