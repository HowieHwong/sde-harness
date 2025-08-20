from typing import List, Dict, Any
import random

from src.utils.chemistry_utils import (
    retrieve_routes,
    process_reaction_routes,
    get_feedback,
    redundant_reaction
)
from syntheseus.search.mol_inventory import SmilesListInventory


GENERAL_PROMPT = """
You are a professional chemist specializing in synthesis analysis. Your task is to propose a retrosynthesis route for a target molecule provided in SMILES format.

Definition:
A retrosynthesis route is a sequence of backward reactions that starts from the target molecules and ends with commercially purchasable building blocks.

Key concepts:
- Molecule set: The working set of molecules at any given step. Initially, it contains only the target molecule.
- Commercially purchasable: Molecules that can be directly bought from suppliers (permitted building blocks).
- Non-purchasable: Molecules that must be further decomposed via retrosynthesis steps.
- Reaction source: All reactions must be derived from the USPTO dataset, and stereochemistry (e.g., E/Z isomers, chiral centers) must be preserved.

Process:
1. Initialization: Start with the molecule set = [target molecule].
2. Iteration:
    - Select one non-purchasable molecule from the molecule set (the product).
    - Apply a valid backward reaction from the USPTO dataset to decompose it into reactants.
    - Remove the product molecule from the set.
    - Add the reactants to the set.
3. Termination: Continue until all molecules in the set are commercially purchasable.
"""


# --------------
# INITIALIZATION
# --------------
INITIALIZATION_TASK_DESCRIPTION = """
My target molecule is: {}

To assist you for the format, an example retrosynthesis route is provided.
{}

Please propose a retrosynthesis route for my target molecule. The provided reference routes may be helpful. You can also design a synthetic route based on your own knowledge.
"""

INITIALIZATION_REQUIREMENTS = """
You need to analyze the target molecule and make a retrosynthesis plan in the <PLAN></PLAN> before proposing the route. After making the plan, you should explain the plan in the <EXPLANATION></EXPLANATION>. The route should be a list of steps wrapped in <ROUTE></ROUTE>. Each step in the list should be a dictionary.
At the first step, the molecule set should be the target molecules set given by the user. Here is an example:

<PLAN>: Analyze the target molecule and plan for each step in the route. </PLAN>
<EXPLANATION>: Explain the plan. </EXPLANATION>
<ROUTE>
[   
    {
        "Molecule set": "[Target Molecule]",
        "Rational": "Step analysis",
        "Product": "[Product molecule]",
        "Reaction": "[Reaction template]",
        "Reactants": "[Reactant1, Reactant2]",
        "Updated molecule set": "[Reactant1, Reactant2]"
    },
    {
        "Molecule set": "[Reactant1, Reactant2]",
        "Rational": "Step analysis",
        "Product": "[Product molecule]",
        "Reaction": "[Reaction template]",
        "Reactants": "[subReactant1, subReactant2]",
        "Updated molecule set": "[Reactant1, subReactant1, subReactant2]"
    }
]
</ROUTE>


Requirements: 
1. The "Molecule set" contains molecules we need to synthesize at this stage. In the first step, it should be the target molecule. In the following steps, it should be the "Updated molecule set" from the previous step.
2. The "Rational" part in each step should be your analysis for syhthesis planning in this step. It should be in the string format wrapped with ""
3. "Product" is the molecule we plan to synthesize in this step. It should be from the "Molecule set". The molecule should be a molecule from the "Molecule set" in a list. The molecule smiles should be wrapped with "".
4. "Reaction" is a backward reaction which can decompose the product molecule into its reactants. The reaction should be in a list. All the molecules in the reaction template should be in SMILES format. For example, ["Product>>Reactant1.Reactant2"].
5. "Reactants" are the reactants of the reaction. It should be in a list. The molecule smiles should be wrapped with "".
6. The "Updated molecule set" should be molecules we need to purchase or synthesize after taking this reaction. To get the "Updated molecule set", you need to remove the product molecule from the "Molecule set" and then add the reactants in this step into it. In the last step, all the molecules in the "Updated molecule set" should be purchasable.
7. In the <PLAN>, you should analyze the target molecule and plan for the whole route.
8. In the <EXPLANATION>, you should analyze the plan.
"""

def construct_initialization_prompt(
    target_smiles: str,
    example_routes: str
) -> str:
    """Create the INITIALIZATION prompt for querying the LLM for initial routes based on example reference routes."""
    return GENERAL_PROMPT + INITIALIZATION_REQUIREMENTS + INITIALIZATION_TASK_DESCRIPTION.format(target_smiles, example_routes)


# --------------
# MUTATION
# --------------
MUTATION_REQUIREMENTS = """
You need to analyze the target molecule and make a retrosynthesis plan in the <PLAN></PLAN> before proposing the route. After making the plan, you should explain the plan in the <EXPLANATION></EXPLANATION>. The route should be a list of steps wrapped in <ROUTE></ROUTE>. Each step in the list should be a dictionary. You need to keep a molecule set in which are the molecules we need to synthesize or purchase. In each step, you need to select a molecule from the "Molecule set" as the product molecule in this step and use a reaction to synthesize it. Usually, the reactants are easier to synthesize or can be purchased from the market. After proposing the reaction in this step, you need to remove the product molecule from the molecule set and add the reactants in this reaction into the molecule set and then name this updated set as the "Updated molecule set" in this step. In the next step, the starting molecule set should be the "Updated molecule set" from the previous step. In the last step, all the molecules in the "Updated molecule set" should be purchasable. Here is an example: corresponds to a set of molecules that are commercially available. Here is an example:

<PLAN>: Analyze the target molecule set and plan for each step in the route. </PLAN>
<EXPLANATION>: Explanation for the whole route. </EXPLANATION>
<ROUTE>
[   
    {
        "Molecule set": "[Target molecules]",
        "Rational": "Step analysis",
        "Product": "[Product molecule]",
        "Reaction": "[Reaction template]",
        "Reactants": "[Reactant1, Reactant2]",
        "Updated molecule set": "[Reactant1, Reactant2]"
    },
    {
        "Molecule set": "[Reactant1, Reactant2]",
        "Rational": "Step analysis",
        "Product": "[Product molecule]",
        "Reaction": "[Reaction template]",
        "Reactants": "[subReactant1, subReactant2]",
        "Updated molecule set": "[Reactant1, subReactant1, subReactant2]"
    }
]
</ROUTE>

Requirements: 
1. The "Molecule set" contains all of the molecules we need to synthesize. In the first step, it should be the list of target molecules given by the user. In the following steps, it should be the "Updated molecule set" from the previous step.
2. The "Rational" part in each step should be your analysis for synthesis planning in this step. It should be in the string format wrapped with "".
3. "Product" is the molecule we plan to synthesize in this step. It should be from the "Molecule set". The molecule should be a molecule from the "Molecule set" in a list. The molecule smiles should be wrapped with "".
4. "Reaction" is a backward reaction which can decompose the product molecule into its reactants. The reaction should be in a list. All the molecules in the reaction template should be in SMILES format. For example, ["Product>>Reactant1.Reactant2"].
5. "Reactants" are the reactants of the reaction. It should be in a list. The molecule smiles should be wrapped with "".
6. The "Updated molecule set" should be molecules we need to purchase or synthesize after taking this reaction. To get the "Updated molecule set", you need to remove the product molecule from the "Molecule set" and then add the reactants in this step into it. In the last step, all the molecules in the "Updated molecule set" should be purchasable.
7. In the <PLAN>, you should make a plan to synthesize the target molecules.
8. In the <EXPLANATION>, you should explain the plan.
"""

def modification_hints(
    nonpurchasable_molecule: List[str],
    all_fps: List[int],
    route_list: List[List[str]]
) -> str:
    """Generate modification hints based on similar routes for non-purchasable molecules."""
    
    fb = """\n
    """
    examples = []
    for smi in nonpurchasable_molecule:
        examples = examples + retrieve_routes(
            target_smi=smi,
            all_fps=all_fps,
            route_list=route_list,
            number=3
        )
    random.shuffle(examples)
    for i in examples:
        fb = fb + str(process_reaction_routes(i)) + "\n"

    return fb

def construct_mutation_prompt(
    molecule_smi_list: List[str],
    examples: str,
    evaluation: Dict[str, Any],
    inventory: SmilesListInventory,
    reaction_cache: Dict[str, List[str]]
) -> str:
    """Construct a mutation prompt for modifying synthetic routes based on feedback."""

    previous_route, feedback = get_feedback(
        molecule_set=molecule_smi_list,
        evaluation=evaluation,
        inventory=inventory
    )

    if not evaluation["product_inside"]:
        mutation_task_description = """
        My target molecule set is: {}

        To assist you for the format, an example retrosynthesis route is provided.\n {}

        Please propose a retrosynthesis route for the target molecule set. The provided reference routes may be helpful. You can also design a synthetic route based on your own knowledge.
        """.format(str(molecule_smi_list), examples)
    else:
        mutation_task_description = """
        My target molecule set is: {}

        In the previous attempt, the first step is: 
        <ROUTE>
        {}
        </ROUTE>
        The feedback for this step is: {}
        To assist you for the format, an example retrosynthesis route is provided.\n {}
        Please propose a retrosynthesis route for the starting molecule set. The provided reference routes may be helpful. You can also design a synthetic route based on your own knowledge. All the molecules should be in SMILES format. For example, Cl2 should be ClCl in SMILES format. Br2 should be BrBr in SMILES format. H2O should be O in SMILES format. HBr should be [H]Br in SMILES format. NH3 should be N in SMILES format. Hydrogen atoms are implicitly understood unless explicitly needed for clarity.
        """.format(str(molecule_smi_list), previous_route, feedback, examples)

    mutation_prompt = GENERAL_PROMPT + MUTATION_REQUIREMENTS + mutation_task_description
    redundant_checking = False

    for smi in molecule_smi_list:
        if smi in reaction_cache:
            redundant_checking = True
            break

    if redundant_checking:
        mutation_prompt += redundant_reaction(
            smi_list=molecule_smi_list,
            reaction_cache=reaction_cache
        )

    return mutation_prompt
