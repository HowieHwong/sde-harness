# Archive Summary

This folder contains Python files that were identified as unused or replaced in the current implementation.

## Date Archived
2025-07-31

## Files Moved

### Backup/Old Versions (src/modes/)
- `perturb_genes_backup.py` - Backup version of perturb_genes mode
- `perturb_genes_fixed.py` - Another backup/fixed version
- `perturb_genes_with_scores.py` - Version with scores implementation (merged into main perturb_genes.py)

### Unused Standalone Scripts
- `analyze_predictions.py` - Not imported anywhere in current codebase
- `get_gene_summary.py` - Not imported anywhere in current codebase  
- `test_refactor.py` - Test file for refactoring

### Legacy Main Scripts and Compatibility Wrappers (Added 2025-07-31)
- `research_assistant.py` - Original main script, uses legacy tools.py and LLM modules
- `research_assistant_new.py` - Backward compatibility wrapper that translates old CLI args to new format
- `analyze.py` - Original analysis script, replaced by src/modes/analyze.py
- `analyze_new.py` - Backward compatibility wrapper for old analyze.py interface

## Files NOT Moved (Still Potentially Used)

### Legacy Files Still Referenced
- `baseline.py` - Legacy version (replaced by src/modes/baseline.py)
- `tools.py` - Legacy version (replaced by src/utils/tools.py)
- `gene.py` - Still imported by legacy files
- `LLM.py` - Still imported by legacy files
- `achilles.py` - Still imported by legacy files
- `get_lit_review.py` - Still imported by legacy tools.py

These were kept in place as they may still be referenced by external scripts or for backward compatibility.

## Current Active Implementation
The project now uses:
- `cli.py` as the main entry point
- `src/modes/perturb_genes.py` as the active perturb genes implementation
- Modular structure under `src/` directory