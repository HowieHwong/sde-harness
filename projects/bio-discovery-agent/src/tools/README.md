# Tool Modules

This directory contains tool modules used by BioDiscoveryAgent for biological analysis and research.

## Files

### Core Tool Files
- **`get_lit_review.py`** - Literature review functionality using PubMed, bioRxiv, and arXiv
- **`achilles.py`** - Gene feature analysis using Achilles dataset
- **`gene.py`** - Gene utilities, pathway analysis (KEGG, Reactome), and RNA-seq data
- **`LLM.py`** - Legacy LLM interface wrapper (for backward compatibility)

### Python Module
- **`__init__.py`** - Module initialization and exports

## Usage

These tools are primarily used through the `BioDiscoveryTools` class in `src/utils/tools.py`:

```python
from src.utils.tools import BioDiscoveryTools

# Literature search
lit_review = BioDiscoveryTools.literature_search(gene_name, research_problem)

# Gene similarity search  
similar_genes = BioDiscoveryTools.gene_search(gene_name, csv_path)

# Pathway analysis
pathways = BioDiscoveryTools.get_reactome_pathways(gene_name)
```

## Migration Notes

These files were moved from the project root directory to improve organization:
- Date: 2025-07-31
- Previous location: Project root (`/`)
- Current location: `/src/tools/`
- All imports have been updated accordingly