"""Prompt templates for BioDiscoveryAgent."""
from sde_harness.core import Prompt

# Base prompt for gene perturbation
PERTURB_GENES_PROMPT = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Reflection: Thoughts on previous results and next steps. 
2. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
3. Solution: Propose a list of predicted genes to test separated by commas in this format: 1. <Gene name 1>, 2. <Gene name 2> ...
Do not include any genes from this prompt (since they're already tested).
"""

# Prompt with gene search tool
PERTURB_GENES_WITH_GENE_SEARCH_PROMPT = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Reflection: Thoughts on previous results and next steps. 
2. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
3. Gene Search: Name a gene to search for 10 most similar genes based on features. Only include the gene name itself after "2. Gene Search:".
4. Solution: Propose a list of predicted genes to test separated by commas in this format: 1. <Gene name 1>, 2. <Gene name 2> ...
Do not include any genes from this prompt (since they're already tested).
"""

# Prompt with correlation tool
PERTURB_GENES_WITH_CORRELATION_PROMPT = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Reflection: Thoughts on previous results and next steps. 
2. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
3. Correlated Genes: Name a gene to search for 10 most correlated genes based on Pearson's correlation. Only include the gene name itself after "2. Correlated Genes:".
4. Solution: Propose a list of predicted genes to test separated by commas in this format: 1. <Gene name 1>, 2. <Gene name 2> ...
Do not include any genes from this prompt (since they're already tested).
"""

# Prompt with RNA expression tool
PERTURB_GENES_WITH_RNA_PROMPT = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Reflection: Thoughts on previous results and next steps. 
2. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
3. Active Tissues: Name a gene to search for the top 10 tissues where this gene is active, based on transcripts per million. Only include the gene name itself after "2. Active Tissues:".
4. Solution: Propose a list of predicted genes to test separated by commas in this format: 1. <Gene name 1>, 2. <Gene name 2> ...
Do not include any genes from this prompt (since they're already tested).
"""

# Prompt with Reactome pathways tool
PERTURB_GENES_WITH_PATHWAYS_PROMPT = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Reflection: Thoughts on previous results and next steps. 
2. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
3. Reactome Pathways: Name a gene to search for the associated biological pathways. Only include the gene name itself after "2. Reactome Pathways:".
4. Solution: Propose a list of predicted genes to test separated by commas in this format: 1. <Gene name 1>, 2. <Gene name 2> ...
Do not include any genes from this prompt (since they're already tested).
"""

# Follow-up prompt template
FOLLOW_UP_PROMPT = """Research problem: {research_problem}

You previously recommended genes that were tested. {observed}

The measured fitness for genes in the previous round was {measurement}: {result}

{instructions}

Always respond in this format exactly:

1. Reflection: Thoughts on previous results and next steps. 
2. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
3. Solution: Propose a list of predicted genes to test separated by commas in this format: 1. <Gene name 1>, 2. <Gene name 2> ...
Do not include any genes from this prompt (since they're already tested).
"""

# Tool-specific follow-up prompt template
FOLLOW_UP_WITH_TOOL_PROMPT = """Research problem: {research_problem}

You previously recommended genes that were tested. {observed}

The measured fitness for genes in the previous round was {measurement}: {result}

{instructions}

Always respond in this format exactly:

1. Reflection: Thoughts on previous results and next steps. 
2. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
3. {tool_section}: Name a gene to search for {tool_description}. Only include the gene name itself after "3. {tool_section}:".
4. Solution: Propose a list of predicted genes to test separated by commas in this format: 1. <Gene name 1>, 2. <Gene name 2> ...
Do not include any genes from this prompt (since they're already tested).
"""


# Norman GI specific prompt
PERTURB_GENES_PAIRS_NORMAN_PROMPT = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
2. Solution: Propose a list of predicted pairs of genes to test separated by commas in this format: 1. <Gene name 1> + <Gene name 2>, 2. <Gene name 3> + <Gene name 4>, 3... 
ONLY CHOOSE FROM this gene list ['AHR', 'ARID1A', 'ARRDC3', 'ATL1', 'BCORL1', 'BPGM', 'CBARP', 'CBFA2T3', 'CBL', 'CDKN1A', 'CDKN1B', 'CDKN1C', 'CEBPA', 'CEBPB', 'CEBPE', 'CELF2', 'CITED1', 'CKS1B', 'CLDN6', 'CNN1', 'CNNM4', 'COL1A1', 'COL2A1', 'CSRNP1', 'DLX2', 'DUSP9', 'EGR1', 'ELMSAN1', 'ETS2', 'FEV', 'FOSB', 'FOXA1', 'FOXA3', 'FOXF1', 'FOXL2', 'FOXL2NB', 'FOXO4', 'GLB1L2','HES7', 'HK2', 'HNF4A', 'HOXA13', 'HOXB9', 'HOXC13', 'IER5L', 'IGDCC3','IKZF3', 'IRF1', 'JUN', 'KIF18B', 'KLF1', 'LHX1', 'LYL1', 'MAML2','MAP2K3', 'MAP2K6', 'MAPK1', 'MEIS1', 'MIDN', 'NIT1', 'OSR2', 'POU3F2','PRDM1', 'PRTG', 'PTPN1', 'PTPN12', 'PTPN13', 'PTPN9', 'RHOXF2B','RP5-862P8.2', 'RREB1', 'S1PR2', 'SAMD1', 'SET', 'SGK1', 'SLC38A2','SLC4A1', 'SLC6A9', 'SNAI1', 'SPI1', 'TBX2', 'TBX3', 'TMSB4X', 'TP73','TSC22D1', 'UBASH3A', 'UBASH3B', 'ZBTB1', 'ZBTB10', 'ZBTB25', 'ZC3HAV1','ZNF318']
"""

# Horlbeck specific prompt
PERTURB_GENES_PAIRS_HORLBECK_PROMPT = """You are a scientist working on problems in drug discovery.

Research Problem: {research_problem}

Always respond in this format exactly:

1. Research Plan: The full high level research plan, with current status and reasoning behind each proposed approach. It should be at most 5 sentences.
2. Reasoning: Explanations of the reasoning behind all the proposed combinations.
3. Solution: Propose a list of predicted pairs of genes to test separated by commas in this format: 1. <Gene name 1> + <Gene name 2>, 2. <Gene name 3> + <Gene name 4>, 3... 
ONLY CHOOSE FROM THIS GENE LIST: ['TNFRSF9', 'ZAP70', 'LHX6', 'EMP3', 'CD27', 'EBF2', 'GRAP2', 'VPS29', 'CBLB', 'IL2RG', 'PLCG2', 'CD3E', 'FOXQ1', 'OTUD7A', 'LIME1', 'DEF6', 'RPL26', 'NMT1', 'NFKB2', 'SLC16A1', 'ZEB2', 'PIK3AP1', 'PI4KB', 'ITPKB', 'MUC21', 'RELA', 'IL9R', 'EIF3K', 'RIPK3', 'PSTPIP1', 'CD28', 'IL2', 'TRIM21', 'PLCG1', 'RNF40', 'MAP3K12', 'CPSF4', 'LAT2', 'CD247', 'IL1R1', 'FOXL2', 'FOSB', 'WT1', 'ARHGAP15', 'AKAP12', 'TRAF3IP2', 'CD3G', 'RPL35', 'VAV1', 'RAC2', 'MYB', 'IFNGR2', 'TSC1', 'MAP3K7', 'TNFRSF1B', 'GRAP', 'SHOC2', 'HELZ2', 'FOXL2NB', 'IRX4', 'FPR2', 'IL2RB', 'SNRPC', 'KIDINS220', 'EP400', 'RPL38', 'PSMD4', 'JAK1', 'INPPL1', 'PTPRC', 'RNF20', 'LCK', 'SPTLC2', 'CD2', 'IFNG', 'RPL19', 'MAP4K1', 'FOXF1', 'ARHGDIB', 'APOBEC3D', 'GCSAML', 'SLAMF6', 'LAT', 'FOXO4', 'EOMES', 'FOSL1', 'LTBR', 'STAT3', 'TRAF6', 'ANXA2R', 'OTUD7B', 'SRP68', 'TBX21', 'ITPKA', 'PDGFRA', 'BICDL2', 'CEACAM1', 'MCM2', 'APOL2', 'SRP19', 'RPS7', 'TAF13', 'GATA3', 'TNFRSF1A', 'EIF3D', 'CD5', 'MCM3AP', 'JMJD1C', 'CAD', 'SLA2', 'WAS', 'CDKN2C', 'MUC1', 'ITK', 'CD3D', 'EMP1', 'DGKZ', 'IKZF3', 'BRD9', 'DEPDC7', 'NRF1', 'HGS', 'MAK16', 'LCP2']
"""


def get_prompt_template(prompt_type: str, **kwargs) -> Prompt:
    """Get a prompt template by type."""
    templates = {
        "perturb_genes": PERTURB_GENES_PROMPT,
        "perturb_genes_gene_search": PERTURB_GENES_WITH_GENE_SEARCH_PROMPT,
        "perturb_genes_correlation": PERTURB_GENES_WITH_CORRELATION_PROMPT,
        "perturb_genes_rna": PERTURB_GENES_WITH_RNA_PROMPT,
        "perturb_genes_pathways": PERTURB_GENES_WITH_PATHWAYS_PROMPT,
        "perturb_genes_pairs_norman": PERTURB_GENES_PAIRS_NORMAN_PROMPT,
        "perturb_genes_pairs_horlbeck": PERTURB_GENES_PAIRS_HORLBECK_PROMPT,
        "follow_up": FOLLOW_UP_PROMPT,
        "follow_up_with_tool": FOLLOW_UP_WITH_TOOL_PROMPT,
    }
    
    if prompt_type not in templates:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    return Prompt(
        custom_template=templates[prompt_type],
        default_vars=kwargs
    )