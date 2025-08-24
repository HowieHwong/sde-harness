from typing import List, Dict, Tuple, Any
import copy
import ast
from itertools import permutations
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions, DataStructs
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction

from src.oracle.oracle import Oracle
from syntheseus import Molecule
from syntheseus.search.mol_inventory import SmilesListInventory

# Fingerprint parameters
RADIUS = 2
# For molecules
MORGAN_FP_GENERATOR = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), radius=RADIUS, useFeatures=False)
# For reactions
REACTION_FP_GENERATOR = rdChemReactions.CreateDifferenceFingerprintForReaction
SIMILARITY_METRIC = DataStructs.BulkTanimotoSimilarity


# ------------------------
# Molecule-level functions
# ------------------------
def check_availability(
    smi_list: List[str],
    inventory: SmilesListInventory
) -> List[str]:
    """Check if the molecules in the list are purchasable (i.e., in the inventory)."""
    unavailable_list = []
    for smi in smi_list:
        signal = inventory.is_purchasable(Molecule(smi))
        if not signal: unavailable_list.append(smi)
    
    return unavailable_list

def check_validity(smi_list: List[str]) -> List[bool]:
    """Check if the molecules in the list are valid."""
    validity_signal = [False] * len(smi_list)
    for idx, smi in enumerate(smi_list):
        signal = sanitize_smiles(smi)
        if signal != None:
            validity_signal[idx] = True
    
    return validity_signal

def check_purchasable(
    smi_list: List[str],
    validity_signals: List[bool],
    inventory: SmilesListInventory
) -> List[bool]:
    """Check if valid molecules in the list are purchasable."""
    availability_signals = [False] * len(smi_list)
    for idx, smi in enumerate(smi_list):
        if validity_signals[idx] == True: 
            signal = inventory.is_purchasable(Molecule(smi))
            availability_signals[idx] = signal
    
    return availability_signals

def sanitize_smiles(smiles: str) -> str:
    """Check SMILES validity and return the sanitized SMILES."""
    if smiles == "": return None

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, canonical=True)
        return smi_canon
    
    except Exception: return None

def rank_reactants(
    reactants_list: List[str],
    inventory: SmilesListInventory,
    oracle: Oracle
) -> List[str]:
    """Rank reactants based on the reward."""
    dead_molecules = dict()
    visited_molecules = dict()
    non_empty_reactant_list = [item for item in reactants_list if item != []]
    scores = [oracle.reward(inventory, reactant, visited_molecules, dead_molecules) for reactant in non_empty_reactant_list]
    sorted_list = [x for _, x in sorted(zip(scores, non_empty_reactant_list), key=lambda pair: pair[0], reverse=True)]
    return sorted_list

def extract_molecules_from_output(output: Any) -> List[str]:
    """Extract molecule SMILES from LLM output."""
    try:
        parsed_output = ast.literal_eval(output)
        if isinstance(parsed_output, list): return parsed_output
        elif isinstance(parsed_output, str): return [parsed_output]
        else: return []
        
    except (ValueError, SyntaxError): return []


# ------------------------
# Reaction-level functions
# ------------------------
def get_reaction_fps(template_dict: Dict[str, str]) -> Tuple[List[str], List[int]]:
    """Get reaction fingerprints from template dictionary."""
    reaction_list = list(template_dict.values())
    all_reaction_fps = [REACTION_FP_GENERATOR(rdChemReactions.ReactionFromSmarts(reaction)) for reaction in reaction_list]
    
    return reaction_list, all_reaction_fps
    
def change_to_forward_reaction(reaction: str) -> str:
    """Convert a reaction from backward to forward direction."""
    products, reactants = reaction.split(">>")
    return reactants + ">>" + products

def preprocess_reaction_dict(reaction_dict: Dict[str, str]) -> Dict[str, Tuple[List[Any], List[Any]]]:
    """Preprocess reaction dictionary to compile SMARTS into RDKit Mol objects."""
    preprocessed_dict = {}
    for key, smarts in reaction_dict.items():
        try:
            products, reactants = smarts.split(">>")
            reactant_mols = [Chem.MolFromSmarts(r) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmarts(p) for p in products.split(".")]
            preprocessed_dict[key] = (reactant_mols, product_mols)
        except Exception:
            continue
    return preprocessed_dict

def process_reaction_routes(route: List[str]) -> List[Dict[str, str]]:
    """Process reaction routes into structured format for LLM prompt."""
    json_list = []
    for idx,i in enumerate(route):
        products, reactants = i.split(">>")
        reaction = i
        reactants_list = reactants.split(".")

        if idx == 0:
            step = {
                "Molecule set": str([products]),
                "Product": str([products]),
                "Reaction": str([reaction]),
                "Reactants": str(reactants_list),
                "Updated molecule set": str(reactants_list)
            }
            json_list.append(step)
        else:
            original_set = copy.deepcopy(ast.literal_eval(json_list[idx-1]["Updated molecule set"]))
            update_set = copy.deepcopy(ast.literal_eval(json_list[idx-1]["Updated molecule set"]))
            try: update_set.remove(products)
            except Exception: pass
            update_set = update_set + reactants_list
            step = {
                "Molecule set": str(original_set),
                "Product": str([reaction]),
                "Reaction": str([reaction]),
                "Reactants": str(reactants_list),
                "Updated molecule set": str(update_set)
            }
            json_list.append(step)
    
    return json_list

def retrieve_routes(
    target_smi: str,
    all_fps: List[int],
    route_list: List[List[str]],
    number: int
) -> List[List[str]]:
    """Retrieve similar routes based on molecular fingerprint similarity."""
    fp = MORGAN_FP_GENERATOR(target_smi)
    sim_score = SIMILARITY_METRIC(fp, [fp_ for fp_ in all_fps])

    rag_tuples = list(zip(sim_score, route_list))
    rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)[:50]
    sims_list, route_list = zip(*rag_tuples)

    sum_scores = sum(sims_list)
    population_probs = [p / sum_scores for p in sims_list]
    sampled_index = np.random.choice(len(route_list), p=population_probs, size=number, replace=False)
    sampled_routes = [route_list[i] for i in sampled_index]

    return sampled_routes

def verify_reaction_step(
    molecule_set: List[str],
    updated_molecule_set: List[str],
    reaction: str,
    product: List[str],
    reactants: List[str],
    inventory: SmilesListInventory,
    oracle: Oracle
) -> Tuple[bool, bool]:
    """Verify if a reaction step is valid and the molecule sets are correctly updated."""

    results = {
        "reaction_valid": False,
        "updated_set_valid": False
    }

    # Parse the reaction, reactants, and products
    try:
        reaction_smiles = reaction

        reactants = [Chem.MolFromSmiles(smi) for smi in reactants]
        products_expected = [Chem.MolFromSmiles(smi) for smi in product]
        updated_molecule_set = [Chem.MolFromSmiles(smi) for smi in updated_molecule_set]
        original_molecule_set = [Chem.MolFromSmiles(smi) for smi in molecule_set]

    except Exception:
        pass

    # Check if the products are generated by the reaction
    try:
        target_rd = rdchiralReactants(product[0])
        reaction_outputs = run_retro(target_rd, reaction_smiles)
        # Rank the reactants by the number of products generated
        if len(reaction_outputs) > 1:
            reaction_outputs = rank_reactants(reaction_outputs, inventory, oracle)
        reactants_generated = [reactant for reactant in reaction_outputs[0]]
        reactants_generated = [sanitize_smiles(smi) for smi in reactants_generated]


        if None in reactants or None in reactants_generated:
            results["reaction_valid"] = False
        elif reactants_generated == []:
            results["reaction_valid"] = False
        else:
            results["reaction_valid"] = True

    except Exception:
        pass

    # Verify if the updated molecule set includes reactants and products
    # In check_route function, the reactants are proposed by LLM
    # In check_route_extra function, the reactants are generated by the reaction
    try:
        updated_smiles = {Chem.MolToSmiles(mol) for mol in updated_molecule_set if mol is not None}
        reactants_smiles = {Chem.MolToSmiles(mol) for mol in reactants if mol is not None}
        products_smiles = {Chem.MolToSmiles(mol) for mol in products_expected}
        original_smiles = {Chem.MolToSmiles(mol) for mol in original_molecule_set if mol is not None}

        expected_updated_sets = (original_smiles | reactants_smiles) - products_smiles

        if expected_updated_sets == updated_smiles and products_smiles.issubset(original_smiles):
            results["updated_set_valid"] = True
        if None in updated_molecule_set: results["updated_set_valid"] = False
        elif None in original_molecule_set: results["updated_set_valid"] = False
        elif None in reactants: results["updated_set_valid"] = False
        elif None in products_expected: results["updated_set_valid"] = False

        common_elements = products_smiles & reactants_smiles
        if common_elements: results["updated_set_valid"] = False

    except Exception:
        pass

    return results["reaction_valid"], results["updated_set_valid"]

def is_reaction_in_dict(
    reaction_smiles: str, 
    preprocessed_dict: Dict[str, Tuple[List[Any], List[Any]]]
) -> Tuple[bool, str]:
    """Check if a reaction SMILES string matches any reaction template in the preprocessed dictionary."""
    reaction_key = None
    try:
        products, reactants = reaction_smiles.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
    except Exception:
        return False, reaction_key

    if None in reactant_mols or None in product_mols:
        return False, reaction_key
    
    for key, (smarts_reactant_mols, smarts_product_mols) in preprocessed_dict.items():
        try:
            # Check if all reactants and products match
            if len(smarts_reactant_mols) != len(reactant_mols):
                continue
            if len(smarts_product_mols) != len(product_mols):
                continue

            reactant_match = is_one_to_one_match(smarts_reactant_mols, reactant_mols)
            product_match = is_one_to_one_match(smarts_product_mols, product_mols)

            if reactant_match and product_match:
                reaction_key = key
                return True, reaction_key
            
        except Exception:
            pass
    
    return False, reaction_key

def is_reaction_match(args: Tuple[List[Any], List[Any], List[Any], List[Any]]) -> bool:
    """Check if reaction components match SMARTS patterns."""
    reactant_mols, product_mols, smarts_reactant_mols, smarts_product_mols = args

    if len(smarts_reactant_mols) != len(reactant_mols): return False
    elif not is_one_to_one_match(smarts_reactant_mols, reactant_mols): return False
    elif len(smarts_product_mols) != len(product_mols): return False
    elif not is_one_to_one_match(smarts_product_mols, product_mols): return False
    else: return True

def is_one_to_one_match(
    smarts_mols: List[Any], 
    target_mols: List[Any]
) -> bool:
    """Check if there's a one-to-one substructure match between SMARTS and target molecules."""
    for perm in permutations(target_mols, len(smarts_mols)):
        if all(target.HasSubstructMatch(smarts) for smarts, target in zip(smarts_mols, perm)):
            return True
    return False

def sanitize_reaction(reaction_smiles: str) -> str:
    """Process a reaction SMILES, removing invalid molecules in reactants and products."""
    try:
        reactants, products = reaction_smiles.split(">>")
    
        reactants_list = reactants.split(".")
        products_list = products.split(".")

        sanitized_reactants = [sanitize_smiles(smiles) for smiles in reactants_list]
        sanitized_products = [sanitize_smiles(smiles) for smiles in products_list]

        sanitized_reactants = [s for s in sanitized_reactants if s is not None]
        sanitized_products = [s for s in sanitized_products if s is not None]

        sanitized_reaction = ".".join(sanitized_reactants) + ">>" + ".".join(sanitized_products)

        return sanitized_reaction

    except Exception: 
        return reaction_smiles

def redundant_reaction(
    smi_list: List[str],
    reaction_cache: Dict[str, List[str]]
) -> str:
    """Generate a redundant reaction prompt for the LLM."""
    text = ""
    for idx, smi in enumerate(smi_list):
        if smi in reaction_cache:
            smi_description = "For molecule {}, these reactions have been tried previously:\n".format(smi)
            for reaction in reaction_cache[smi]: smi_description += reaction + "\n" 
            text += smi_description + "Please do not use them again.\n"
    return text

def run_retro(
    product: str,
    template: str
) -> List[str]:
    """Run a reaction given the product and the template."""
    reactants = template.split(">>")[0].split(".")
    if len(reactants) > 1:
        template = "(" + template.replace(">>", ")>>")
    template = rdchiralReaction(template)

    try:
        outputs = rdchiralRun(template, product)

    except Exception: 
        return []

    return [output.split(".") for output in outputs]

def smiles_to_reaction(smiles: str) -> Any:
    """Convert SMILES string to RDKit reaction object."""
    try:
        reactants, products = smiles.split(">>")
        reactant_list = reactants.split(".")
        product_list = products.split(".")
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_list]
        product_mols = [Chem.MolFromSmiles(p) for p in product_list]
        reaction_smarts = f"{'.'.join([Chem.MolToSmarts(mol) for mol in reactant_mols])}>>{'.'.join([Chem.MolToSmarts(mol) for mol in product_mols])}"
        return rdChemReactions.ReactionFromSmarts(reaction_smarts)
    
    except Exception: 
        return None

def check_and_update_routes(
    routes: List[Dict[str, Any]],
    target_list: List[str]
) -> List[Dict[str, Any]]:
    """Update routes to ensure molecule sets are consistent across steps."""

    routes[0]["Molecule set"] = str(target_list)
    for i in range(1, len(routes)):
        current_updated_set = ast.literal_eval(routes[i]["Molecule set"])
        previous_molecule_set = ast.literal_eval(routes[i - 1]["Updated molecule set"])
        if set(current_updated_set) != set(previous_molecule_set): routes[i]["Molecule set"] = str(previous_molecule_set)

    return routes

def map_reaction(
    routes: List[Dict[str, Any]], 
    stepwise_results: List[Tuple[int, bool, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Use found reactions to substitute reactions proposed by the LLM."""
    for i in range(len(routes)):
        step_id, is_valid, data = stepwise_results[i]
        reaction_smiles = data["reaction"]

        routes[i]["Reaction"] = str([reaction_smiles])

    return routes


# ------------------------
# Route feedback functions
# ------------------------
def get_feedback(
    molecule_set: List[str],
    evaluation: Dict[str, Any],
    inventory: SmilesListInventory
) -> Tuple[str, str]:
    original_set = copy.deepcopy(molecule_set)
    update_set = copy.deepcopy(original_set)

    try: update_set.remove(evaluation["product"][0])
    except Exception: update_set = []
    
    update_set += evaluation["reactants"]
    reaction = evaluation["reaction"]
    step = [{
            "Molecule set": str(molecule_set),
            "Product": str(evaluation["product"]),
            "Reaction": str([reaction]),
            "Reactants": str(evaluation["reactants"]),
            "Updated molecule set": str(update_set)
        }]
    
    # Feedbacks
    feedback = """\n"""
    if evaluation["reaction_existence"] == False: feedback += reaction_unavailable_feedback(evaluation)                    
    if len(evaluation["invalid_updated_mol_id"]) != 0: feedback += molecule_invalid_feedback(evaluation)
    if (evaluation["reaction_existence"] == True) and (evaluation["reaction_valid"] == False): feedback += reaction_cannot_happen_feedback(evaluation)
    if (evaluation["reaction_existence"] == True) and (evaluation["reaction_valid"] == True) and (evaluation["updated_set_valid"] == False): feedback += updated_set_mismatch_feedback(evaluation)
    if (evaluation["check_availability"] == True) and (len(evaluation["unavailable_mol_id"]) != 0) and (len(evaluation["invalid_updated_mol_id"]) == 0): feedback += molecule_unavailable_feedback(evaluation, inventory)

    if feedback == """\n""": feedback = "The provided route is valid. Please try to propose a new synthetic route for this target molecule."

    # Return the feedback and the step
    return str(step), feedback

def reaction_unavailable_feedback(evaluation: Dict[str, Any]) -> str:
    """Generate feedback when a reaction is not available in the database."""
    reaction = evaluation["reaction"]
    fb = """\nThe reaction {} does not exist in the USPTO dataset. Please make sure all the molecules in the reaction are in SMILES format.\n""".format(reaction)

    return fb

def molecule_invalid_feedback(evaluation: Dict[str, Any]) -> str:
    """Generate feedback when molecules are invalid."""
    invalid_molecule_id = evaluation["invalid_updated_mol_id"]
    updated_molecule_set = evaluation["updated_molecule_set"]

    fb = """\nIn the 'Updated molecule set',"""

    for i in range(len(invalid_molecule_id)):
        fb = fb + """
        the molecule {} is not a valid molecule SMILES. Please make sure all the molecules are in the SMILES format.
        """.format(updated_molecule_set[invalid_molecule_id[i]])

    return fb

def reaction_cannot_happen_feedback(evaluation: Dict[str, Any]) -> str:
    """Generate feedback when a reaction cannot happen with the product molecule."""
    reaction = evaluation["reaction"]
    #forward_reaction = change_to_forward_reaction(reaction)    
    fb = """\nThe reaction {} cannot happen with the product molecule. \n""".format(reaction)

    return fb

def updated_set_mismatch_feedback(evaluation: Dict[str, Any]) -> str:
    """Generate feedback when molecule sets are not aligned."""
    reaction = evaluation["reaction"]
    product = evaluation["product"][0]

    fb = """\nThe molecule set and the updated molecule set are not aligned. In each step, you need to keep a molecule set in which are the molecules we need. After taking the backward reaction in this step, you need to remove the products from the molecule set and add the reactants to the molecule set and then store
this set as 'Updated molecule set' in this step. In the last step, all the molecules in the 'Updated molecule set' should be purchasable. Please also check whether the product of this reaction is in the molecule set."""

    return fb

def molecule_unavailable_feedback(
    evaluation: Dict[str, Any],
    inventory: SmilesListInventory
) -> str:
    """Generate feedback when molecules are unavailable."""
    updated_molecule_set = evaluation["updated_molecule_set"]
    non_purchasable_molecule = check_availability(
        smi_list=updated_molecule_set,
        inventory=inventory
    )
    fb = """\nIn the 'Updated molecule set', the molecule {} cannot be purchased from the market.\n""".format(str(non_purchasable_molecule))
    return fb


def starting_invalid_feedback(evaluation: Dict[str, Any]) -> str:
    """Generate feedback when the starting molecule is invalid."""
    fb = """\nIn the first step, the molecule in the molecule set should be the target molecule"""
    return fb


def product_not_inside_feedback(evaluation: Dict[str, Any]) -> str:
    """Generate feedback when the product is not in the molecule set."""
    product = evaluation["product"][0]
    fb = """\nThe product molecule {} is not in the molecule set. Please make sure the product molecule is in the molecule set.\n""".format(product)
    return fb
