import os

# Environment Variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['VLLM_ALLOW_RUNTIME_LORA_UPDATING'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['ENFORCE_EAGER'] = 'true'


# Crystal Structure Generation
PROMPT_PATTERN_CSG_ZEROSHOT = """You are an expert material scientist. Your task is to propose a new material with valid stable structures and compositions. No isolated or overlapped atoms are allowed.

Format requirements:
1. Each proposed structure must be formatted in JSON with the following structure:
{{
    "i": {{
        "formula": "composition_formula",
        "{fmt}": "{fmt}_format_string"
    }}
}}
2. Use proper JSON escaping for newlines (\\n) and other special characters

Example for {fmt} format:
{{
    "0": {{
        "formula": "Na4Cl4O8",
        "{fmt}": "Na4Cl4O8\\n1.0\\n   0.0000    6.6353    0.0000\\n   6.5898    0.0000    0.0000\\n   0.0000   -2.9146   -5.8520\\nNa Cl O\\n4 4 8\\ndirect\\n   0.0000    0.3671    0.2500 Na\\n   0.0000    0.6329    0.7500 Na\\n   0.5000    0.8671    0.2500 Na\\n   0.5000    0.1329    0.7500 Na\\n   0.0000    0.1555    0.7500 Cl\\n   0.0000    0.8445    0.2500 Cl\\n   0.5000    0.6555    0.7500 Cl\\n   0.5000    0.3445    0.2500 Cl\\n   0.1357    0.2949    0.6564 O\\n   0.8643    0.2949    0.8436 O\\n   0.8643    0.7051    0.3436 O\\n   0.1357    0.7051    0.1564 O\\n   0.6357    0.7949    0.6564 O\\n   0.3643    0.7949    0.8436 O\\n   0.3643    0.2051    0.3436 O\\n   0.6357    0.2051    0.1564 O"
    }}
}}

Note: Try to generate diverse structures with different elements, lattice parameters, and atomic arrangements. Create a novel material structure using seed {random_seed}.

IMPORTANT: Please output only the structure data in clean JSON format with no additional text, code blocks, or explanations.

Output your hypotheses below:
"""

PROMPT_PATTERN_CSG = """You are an expert material scientist. Your task is to propose {rep_size} new materials with valid stable structures and compositions. No isolated or overlapped atoms are allowed.

The proposed new materials can be a modification or combination of the base materials given below.

Format requirements:
1. Each proposed structure must be formatted in JSON with the following structure:
{{
    "i": {{
        "formula": "composition_formula",
        "{fmt}": "{fmt}_format_string"
    }}
}}
2. Use proper JSON escaping for newlines (\\n) and other special characters

Base material structure for reference:
{input}

Your task:
1. Generate {rep_size} new structure hypotheses
2. Each structure should be stable and physically reasonable
3. Format each structure exactly as shown in the input

IMPORTANT: Please output only the structure data in clean JSON format with no additional text, code blocks, or explanations.

Output your hypotheses below:
"""

# PROMPT_CRYSTALLLM = PROMPT_PATTERN_CSG
PROMPT_CRYSTALLLM = """Below is a description of a bulk material. 
{input}
Design a novel, thermodynamically stable variants of the input crystal by filling the [MASK]. Generate a description of the lengths and angles of a new lattice vectors and then the element type and coordinates for each atom within the lattice:\n"
"""
# PROMPT_CRYSTALLLM = """Below is a description of a bulk material. 
# {input}
# Design a novel, thermodynamically stable variants of the input crystal. Generate a description of the lengths and angles of a new lattice vectors and then the element type and coordinates for each atom within the lattice:\n"
# """

# Crystal Structure Prediction
# PROMPT_PATTERN_CSP = """You are an expert material scientist. Your task is to design {rep_size} novel, thermodynamically stable variants of sodium aluminum chloride with the general formula (Na3AlCl6)*n, where n represents different multiples of the base composition. No isolated or overlapped atoms are allowed. You may refer to the reference structures provided below as inspiration, but ensure you propose novel atomic arrangements beyond simple atomic substitution.

# Crystal structures for reference:
# {input}

# Format requirements:
# 1. Each proposed structure must be formatted in JSON with the following structure:
# {{
#     "i": {{
#         "formula": "Na3AlCl6",
#         "{fmt}": "{fmt}_format_string"
#     }}
# }}
# 2. Use proper JSON escaping for newlines (\\n) and other special characters.

# Output your hypotheses below:
# """

# 
# Crystal Structure Prediction
# PROMPT_PATTERN_CSP = """You are an expert material scientist. Your task is to find optimal positions for intercalant atom Zn within the host structure MnO2 and propose hypotheses for {rep_size} intercalated structures. No isolated or overlapped atoms are allowed. The Ag6O2 framework must be strictly preserved as the host structure throughout any modifications. 

# The parent structures only suggest the valid structure motif; please use the reference host structure as guidelines for your proposed materials.

# Crystal structures for reference:
# {input}

# Format requirements:
# 1. Each proposed structure must be formatted in JSON with the following structure:
# {{
#     "i": {{
#         "formula": "Li2ZrCl6",
#         "{fmt}": "{fmt}_format_string"
#     }}
# }}
# 2. Use proper JSON escaping for newlines (\\n) and other special characters.

# Output your hypotheses below:
# """
PROMPT_PATTERN_CSP = """You are an expert computational materials scientist specializing in crystal structure prediction. Your task is to predict the ground state crystal structure of {compound}.

Propose {rep_size} distinct, physically realistic polymorphs that might represent the ground state (lowest energy) structure for {compound}. Consider different space groups, coordination environments, and bonding motifs that would minimize the total energy.

Key considerations for each structure:
1. Ensure proper atomic coordination based on chemical principles
2. Maintain reasonable bond lengths and angles
3. Avoid isolated atoms, overlapping atoms, or unrealistic coordination environments
4. Consider common structure types observed in similar compounds
5. Apply knowledge of electronegativity and ionic radius to predict reasonable structures

Crystal structures for reference:
{input}

Format requirements:
Each proposed structure must be formatted in JSON with the following structure:
{{
    "i": {{
        "formula": "{compound}",
        "{fmt}": "{fmt}_format_string"
    }}
}}

Use proper JSON escaping for newlines (\\n) and other special characters.

IMPORTANT: Please output only the structure data in clean JSON format with no additional text, code blocks, or explanations.

Output your ground state polymorph prediction below:
"""

# You are an expert material scientist. Your task is to design {rep_size} novel, thermodynamically stable variants of Lithium Zirconium Chloride structures with the general formula (Li2ZrCl6)*n, where n represents different multiples of the base composition. No isolated or overlapped atoms are allowed. You may refer to the reference structures provided below as inspiration, but ensure you propose novel atomic arrangements beyond simple atomic substitution.

# You are an expert material scientist. Your task is to design {rep_size} novel, thermodynamically stable variants of sodium aluminum chloride with the general formula (Na3AlCl6)*n, where n represents different multiples of the base composition. No isolated or overlapped atoms are allowed. The proposed new structure can be a modification or combination of the reference structures given below.

# You are an expert material scientist. Your task is to design {rep_size} novel, thermodynamically stable variants of sodium aluminum chloride with the general formula (Na3AlCl6)*n, where n represents different multiples of the base composition. No isolated or overlapped atoms are allowed. You may refer to the reference structures provided below as inspiration, but ensure you propose novel atomic arrangements beyond simple atomic substitution.

# You may refer to the reference structures provided below as inspiration, but ensure you propose novel atomic arrangements beyond simple atomic substitution.

# Your task:
# 1. Generate {rep_size} new structure hypotheses
# 2. Each structure should be stable and physically reasonable
# 3. Each structure must maintain the composition where Na:Ta:Cl = 3:1:6
# 4. Format each structure exactly as shown in the input

# Crystal Structure Inpainting
# BATCH_PATTERN = """You are an expert material scientist. Your task is to design {rep_size} novel, thermodynamically stable intercalation compounds with Zn ions in MnO2 structure. The composition should follow the formula Zn + MnO2, where the number of Zn atoms must be less than Mn atoms. Please generate structures for different compositions in the reduced formula of ZnxMnO2. 

# Place Zn atoms in physically reasonable intercalation sites. No isolated or overlapped atoms are allowed. Do not use simple element substitution but instead propose new structure prototypes based on the structure motif from the parent structures.

# Crystal structures for reference:
# {input}

# Format requirements:
# 1. Each proposed structure must be formatted in JSON with the following structure:
# {{
#     "i": {{
#         "formula": "Zn + MnO2",
#         "{fmt}": "{fmt}_format_string"
#     }}
# }}
# 2. Use proper JSON escaping for newlines (\\n) and other special characters

# Output your hypotheses below:
# """

# Place Zn atoms in physically reasonable intercalation sites. No isolated or overlapped atoms are allowed. You may use the provided reference structures as templates for inspiration, modification, or combination, while ensuring your proposed structures represent novel configurations rather than simple atomic substitutions. 

# BATCH_PATTERN = """You are an expert material scientist. Your task is to propose hypotheses for {rep_size} new materials with valid stable structures and compositions that optimize the material property towards band gap of 3eV calculated with PBE-DFT. No isolated or overlapped atoms are allowed.
# The proposed new materials can be a modification or combination of the existing materials given below.
# Format the structure of each material proposal into a JSON string which should be of the exact same format as the input material structure. 
# ---
# Output your hypotheses in the following HTML tags, each hypothesis as an item:
# <ul>
#     <li>
#         <structure>
#         </structure>
#     </li>
# </ul>

# ---
# The base crystal structures are: ```{input}```
# ---
# Output Hypothesis:
# """
# BATCH_PATTERN = """You are an expert material scientist. Your task is to propose hypotheses for {rep_size} new materials with valid stable structures and compositions that optimize the material property towards band gap of 3eV calculated with PBE-DFT. No isolated or overlapped atoms are allowed.
# The proposed new materials can be a modification of the existing material given below.
# Format the structure of each material proposal into a JSON string which should be of the exact same format as the input material structure. 
# ---
# Output your hypotheses in the following HTML tags, each hypothesis as an item:
# <ul>
#     <li>
#         <structure>
#         </structure>
#     </li>
# </ul>

# ---
# The structure of the base material: ```{input}```
# ---
# Output Hypothesis:
# """


# Model Settings
MODEL_SETTINGS = {
    "70b": {
        "max_token_length": 7904,
    },
    "mistral": {
        "max_token_length": 32000,
    }
}

# Stability Thresholds
STABILITY_THRESHOLDS = [0.03, 0.06]


LANTHANIDES = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
ACTINIDES = ['Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
METAL_OXIDATION_STATES= {
    "Li": [1],
    "Na": [1],
    "K": [1],
    "Rb": [1],
    "Cs": [1],
    "Be": [2],
    "Mg": [2],
    "Ca": [2],
    "Sr": [2],
    "Ba": [2],
    "Sc": [3],
    "Ti": [4, 3],
    "V": [5, 4, 3, 2],
    "Cr": [2, 3, 4, 6],
    "Mn": [2, 3, 4, 7],
    "Fe": [2, 3, 4],
    "Co": [2, 3, 4],
    "Ni": [2, 3, 4],
    "Cu": [2, 1],
    "Zn": [2],
    "Y": [3],
    "Zr": [4],
    "Nb": [5],
    "Mo": [6, 4],
    "Tc": [7],
    "Ru": [4, 3],
    "Rh": [3],
    "Pd": [2, 4],
    "Ag": [1],
    "Cd": [2],
    "La": [3],
    "Hf": [4],
    "Ta": [5],
    "W": [6],
    "Re": [7],
    "Os": [4, 8],
    "Ir": [3, 4],
    "Pt": [2, 4],
    "Au": [3, 1],
    "Hg": [2, 1],
    "Al": [3],
    "Ga": [3],
    "In": [3],
    "Tl": [1, 3],
    "Sn": [4, 2],
    "Pb": [2, 4],
    "Bi": [3],
}