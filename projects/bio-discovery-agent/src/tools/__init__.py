"""Tool modules for BioDiscoveryAgent."""
from .get_lit_review import get_lit_review
from .achilles import download_csv, parse_genes, search_genes
from .gene import (
    get_topk_genes_in_pathways,
    get_enrichment_KEGG_pathways,
    get_enrichment,
    get_gene_to_reactome_pathways,
    get_topk_genes_in_reactome,
    get_rna_seq,
    get_top_k_correlated_genes
)
from .LLM import complete_text as legacy_complete_text

__all__ = [
    'get_lit_review',
    'download_csv',
    'parse_genes', 
    'search_genes',
    'get_topk_genes_in_pathways',
    'get_enrichment_KEGG_pathways',
    'get_enrichment',
    'get_gene_to_reactome_pathways',
    'get_topk_genes_in_reactome',
    'get_rna_seq',
    'get_top_k_correlated_genes',
    'legacy_complete_text'
]