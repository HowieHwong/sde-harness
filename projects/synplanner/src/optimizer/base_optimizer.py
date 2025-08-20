from typing import List, Dict, Any, Tuple
import os
import logging
import yaml
import json
import pickle
import ast
import heapq
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdchiral.initialization import rdchiralReactants

from src.utils.chemistry_utils import (
    sanitize_smiles, sanitize_reaction, run_retro, check_validity, check_purchasable, smiles_to_reaction, is_reaction_in_dict, verify_reaction_step,
    get_reaction_fps, preprocess_reaction_dict, check_and_update_routes, map_reaction, extract_molecules_from_output,
    MORGAN_FP_GENERATOR, SIMILARITY_METRIC
)
from src.oracle.oracle import Oracle
from src.rag.sim_based_rag import get_data_df, split_data_df, do_one

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "idx2template_retro.json")
INVENTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "inventory.pkl")
RULE_BASED_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "data_processed.csv")
RULE_BASED_FPS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "rule_based_fps.pkl")



class BaseOptimizer:
    def __init__(
        self, 
        args: Any = None
    ) -> None:
        """Initialize the base optimizer with configuration parameters."""
        self.args = args

        self.oracle = Oracle(args=self.args)

        args.template_path = TEMPLATE_PATH
        args.inventory_path = INVENTORY_PATH
        args.rule_based_set_path = RULE_BASED_SET_PATH

        self.original_template_dict = self.load_template(args.template_path)
        self.template_dict =  preprocess_reaction_dict(self.original_template_dict)
        self.inventory = self.load_inventory(args.inventory_path)
        self.reaction_list, self.all_reaction_fps = get_reaction_fps(self.original_template_dict)
        self.datasub = self.load_rule_based_set(args.rule_based_set_path)

        self.explored_reaction = set()
        self.visited_molecules = dict() # SMILES: visit number
        self.dead_molecules = dict()
        self.jx_cache = {}
        self.template_to_key = {v: k for k, v in self.original_template_dict.items()}

    def load_template(
        self, 
        template_path: str
    ) -> Dict[str, str]:
        """Load template dictionary from JSON file."""
        with open(template_path, "r") as f: template_dict = json.load(f)
        return template_dict

    def load_inventory(
        self, 
        inventory_path: str
    ) -> List[str]:
        """Load inventory from pickle file."""
        with open(inventory_path, "rb") as file:
            inventory = pickle.load(file)
        
        return inventory
    
    def load_rule_based_set(
        self, 
        rule_based_set_path: str
    ) -> pd.DataFrame:
        """Load rule-based set from CSV file."""
        data = get_data_df(rule_based_set_path)
        split_data_df(data)

        if os.path.exists(RULE_BASED_FPS_FILE):
            logging.info("Loading rule-based set's fingerprints from file...")
            all_fps = pickle.load(open(RULE_BASED_FPS_FILE, "rb"))
        else:
            logging.info(f"Computing Morgan fingerprints of rule-based set... will be stored in ./dataset/rule_based_fps.pkl")
            all_fps = [MORGAN_FP_GENERATOR(smi) for smi in data["prod_smiles"]]
            data["prod_fp"] = all_fps
            pickle.dump(all_fps, open(RULE_BASED_FPS_FILE, "wb"))

        return data.loc[data["dataset"] == "val"]

    def update_visited_molecules(
        self, 
        updated_molecule_set: List[str]
    ) -> None:
        """Update the count of visited molecules."""
        for smi in updated_molecule_set:
            if smi in self.visited_molecules:
                self.visited_molecules[smi] += 1
            else:
                self.visited_molecules[smi] = 1

    def update_dead_molecules(self, dead_molecule: str) -> None:
        """Update the count of dead molecules that cannot be synthesized."""
        smi = dead_molecule
        assert isinstance(smi, str)
        if smi in self.dead_molecules:
            self.dead_molecules[smi] += 1
        else:
            self.dead_molecules[smi] = 1

    def rule_based_search(
        self, 
        product_smiles: str, 
        reaction_smiles: str
    ) -> Tuple[bool, str, str]:
        """Search for reaction templates based on similarity to known reactions."""
        template_list, self.jx_cache = do_one(product_smiles, self.datasub, self.jx_cache)

        if template_list == []:
            return False, None, reaction_smiles
        else:
            templates = [t[0] for t in template_list]
            scores = [t[1] for t in template_list]
            weights = np.array(scores) 
            probabilities = weights / weights.sum()  # Normalize to get probabilities
            sampled_index = np.random.choice(len(templates), p=probabilities, size=len(templates), replace=False)
            sorted_templates = [templates[i] for i in sampled_index]
            for template in sorted_templates:
                raw_template = template[1:].replace(")>>", ">>")
                if (product_smiles, raw_template) in self.explored_reaction:
                    continue
                else:
                    key = "99999999999"
                    return True, key, raw_template
            return False, None, reaction_smiles

    def blurry_search(
        self, 
        reaction_smiles: str, 
        product_smiles: str, 
        exploration_signal: bool
    ) -> Tuple[bool, str, str]:
        """Search for similar reactions using fingerprint similarity and test them on the given product."""
        if exploration_signal == True:
            reaction_number = 1000
        else:
            reaction_number = 100

        try:
            # Remove invalid molecules in the reaction to perform similarity search
            sanitized_reaction = sanitize_reaction(reaction_smiles)
            rxn_obj = smiles_to_reaction(sanitized_reaction)
            fp_re = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn_obj)

            sims = SIMILARITY_METRIC(fp_re, self.all_reaction_fps)
            top_indices = heapq.nlargest(reaction_number, range(len(sims)), key=lambda i: sims[i])
            sorted_reaction_list = [self.reaction_list[i] for i in top_indices]
            
        except Exception:
            return self.rule_based_search(product_smiles, reaction_smiles)

        try: 
            target_rd = rdchiralReactants(product_smiles)
        except Exception: 
            return False, None, reaction_smiles

        for reaction_smarts in sorted_reaction_list:
            try:
                reaction_outputs = run_retro(target_rd, reaction_smarts)
                if len(reaction_outputs) > 1:
                    reaction_outputs = self.rank_reactants(reaction_outputs)
                if len(reaction_outputs) == 0:
                    continue
                reactants_generated = [reactant for reactant in reaction_outputs[0]]
                if reactants_generated == []:
                    continue
                elif len(reactants_generated) > 0:
                    key = self.template_to_key.get(reaction_smarts)
                    if (product_smiles, reaction_smarts) in self.explored_reaction:
                        continue
                    return True, key, reaction_smarts
            except Exception:
                continue

        return self.rule_based_search(product_smiles, reaction_smiles)

    def sanitize(
        self,
        starting_list: List[str],
        route: List[Dict[str, Any]],
        exploration_signal: bool
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, bool, Dict[str, Any]]]]:
        """Sanitize and validate a synthetic route."""
        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        # Checks that each subsequent step starts from the previous step's product
        new_route = check_and_update_routes(
            routes=route, 
            target_list=starting_list
        )
        first_evaluation = self.check_route(
            target_smi=starting_list, 
            route=new_route, 
            exploration_signal=exploration_signal
        )
        # Update the route based on the LLM-proposed reactions (E.g., with exact match templates from the reference database)
        new_route = map_reaction(
            routes=new_route, 
            stepwise_results=first_evaluation
        )
        new_route = self.fix_reaction_error(
            routes=new_route, 
            stepwise_results=first_evaluation
        )
        new_route = check_and_update_routes(
            routes=new_route, 
            target_list=starting_list
        )
        # Final_evaluation is a list of tuples, each containing:
        #   - Step index (int)
        #   - Whether the step is valid (bool)
        #   - Step information (dictionary)
        final_evaluation = self.check_route_extra(
            target_smi=starting_list, 
            route=new_route, 
            first_evaluation=first_evaluation
        )

        return new_route, final_evaluation

    def sort_buffer(self) -> None:
        """Sort the oracle route buffer."""
        self.oracle.sort_buffer()
    
    def log_intermediate(
        self, 
        finish: bool = False
    ) -> None:
        """Log intermediate results during optimization."""
        self.oracle.log_intermediate(finish=finish)
      
    def save_result(
        self, 
        suffix: str = None
    ) -> None:
        """Save optimization results to a YAML file."""
        
        if suffix is None:  output_file_path = os.path.join(self.args.output_dir, "results.yaml")
        else:
            suffix = suffix.replace("/", "")
            output_file_path = os.path.join(self.args.output_dir, "results_" + suffix + ".yaml")

        self.sort_buffer()
        with open(output_file_path, "w") as f:
            yaml.dump(self.route_buffer, f, sort_keys=False)
    
    def check_route(
        self, 
        target_smi: List[str], 
        route: List[Dict[str, Any]], 
        exploration_signal: bool
    ) -> List[Tuple[int, bool, Dict[str, Any]]]:
        """Check if the route is valid."""
        results = []
        for i in range(len(route)):
            current_step_index = i
            current_step = route[current_step_index]

            step_validity = False
            molecule_set = ast.literal_eval(current_step["Molecule set"])
            updated_molecule_set = ast.literal_eval(current_step["Updated molecule set"])
            reaction = ast.literal_eval(current_step["Reaction"])[0]
            product = extract_molecules_from_output(current_step["Product"])
            reactants = ast.literal_eval(current_step["Reactants"])

            # -----------------------------
            # Step 1: Molecule-level Checks
            # -----------------------------
            starting_signal = True
            # If first step, make sure the route starts from the target molecule
            if current_step_index == 0:
                mdd = set(molecule_set).issubset(set(target_smi))
                if not mdd:
                    starting_signal = False

            # Product in molecule set
            product_inside = False
            if product[0] in molecule_set:
                product_inside = True
            invalid_molset_mol_id = []
            invalid_updated_mol_id = []
            
            # Check that the mols are RDKit-parsable
            updated_set_signals = check_validity(updated_molecule_set)
            if False in updated_set_signals:
                invalid_updated_mol_id = [index for index, value in enumerate(updated_set_signals) if not value]
    
            mol_set_signals = check_validity(molecule_set)
            if False in mol_set_signals:
                invalid_molset_mol_id = [index for index, value in enumerate(mol_set_signals) if not value]

            # Check purchasability
            check_availability = False
            unavailable_mol_id = []
            if i == len(route) - 1:
                availabilities = check_purchasable(updated_molecule_set, updated_set_signals, self.inventory)
                check_availability = True
            if check_availability == True:
                if False in availabilities:
                    unavailable_mol_id = [index for index, value in enumerate(availabilities) if not value]
    
            # -----------------------------
            # Step 2: Reaction-level Checks
            # -----------------------------
            reaction_valid, updated_set_valid, reaction_existence = False, False, False
            # ":" is a delimiter for reaction templates
            if ":" in reaction:
                # Check if the LLM proposed reaction has an exact match template
                keys = [key for key, value in self.original_template_dict.items() if value == reaction]
                if len(keys) == 1:
                    reaction_existence = True
                    reaction_key = keys[0]
                else:
                    reaction_existence = False
                    reaction_key = None
            else:
                if current_step_index == 0:
                    # Check if the LLM proposed reaction has an exact match template
                    reaction_existence, reaction_key = is_reaction_in_dict(reaction, self.template_dict)
                # To save time, we only check the step after a valid step (i.e. the previous step is valid)
                elif results[-1][1] == True:
                    reaction_existence, reaction_key = is_reaction_in_dict(reaction, self.template_dict)

                else:
                    reaction_existence = False
                    reaction_key = None

            if reaction_key == None:
                new_reaction = reaction
            else:
                new_reaction = self.original_template_dict[reaction_key]
                
            # Verify the reaction step
            # E.g., checks that applying the template yields the expected reactants
            if reaction_existence == True:
                reaction_valid, updated_set_valid = verify_reaction_step(
                    molecule_set=molecule_set,
                    updated_molecule_set=updated_molecule_set,
                    reaction=new_reaction,
                    product=product,
                    reactants=reactants,
                    inventory=self.inventory,
                    oracle=self.oracle)
            
            # If the first reaction step did not yield an exact match template, then run a "blurry search"
            # which takes the LLM-proposed reaction and runs a similarity search on the reaction templates database
            # It then ranks the templates (out of 100 or 1000, if exploration is turned on) by the one that gives 
            # reactants with the highest reward. Reactions previously explored are not considered. 
            # If by the end, no valid reaction is found, then the LLM proposed product is used to search for closest molecule neighbour
            # again by Tanimoto similarity of Morgan fingerprints
            if current_step_index == 0:
                if reaction_key == None: 
                    reaction_existence, reaction_key, new_reaction = self.blurry_search(reaction, product[0], exploration_signal)
                elif (product[0], new_reaction) in self.explored_reaction: 
                    reaction_existence, reaction_key, new_reaction = self.blurry_search(reaction, product[0], exploration_signal)
                elif reaction_existence and not reaction_valid: 
                    reaction_existence, reaction_key, new_reaction = self.blurry_search(reaction, product[0], exploration_signal)  
            
                if reaction_key == None: new_reaction = reaction
                else: reaction_valid, updated_set_valid = verify_reaction_step(
                    molecule_set=molecule_set,
                    updated_molecule_set=updated_molecule_set,
                    reaction=new_reaction,
                    product=product,
                    reactants=reactants,
                    inventory=self.inventory,
                    oracle=self.oracle)
               
            if (
                len(invalid_molset_mol_id) == 0 and
                len(invalid_updated_mol_id) == 0 and
                reaction_valid and
                updated_set_valid and
                starting_signal and
                product_inside
            ):
                step_validity = True

            # Construct the dictionary
            step_info = {
                "target_smi": target_smi,
                "starting_signal": starting_signal,
                "product_inside": product_inside,
                "molecule_set": molecule_set,
                "updated_molecule_set": updated_molecule_set,
                "reaction": new_reaction,
                "reaction_key": reaction_key,
                "product": product,
                "reactants": reactants,
                "updated_set_signals": updated_set_signals,
                "invalid_updated_mol_id": invalid_updated_mol_id,
                "mol_set_signals": mol_set_signals,
                "invalid_molset_mol_id": invalid_molset_mol_id,
                "check_availability": check_availability,
                "unavailable_mol_id": unavailable_mol_id,
                "reaction_existence": reaction_existence,
                "reaction_valid": reaction_valid,
                "updated_set_valid": updated_set_valid
            }

            # Store the tuple in the results list
            results.append((current_step_index, step_validity, step_info))

        return results

    def check_route_extra(
        self, 
        target_smi: List[str], 
        route: List[Dict[str, Any]], 
        first_evaluation: List[Tuple[int, bool, Dict[str, Any]]]
    ) -> List[Tuple[int, bool, Dict[str, Any]]]:
        """Check if the route is valid again."""
        results = []
        for i in range(len(route)):
            current_step_index = i
            current_step = route[current_step_index]
            step_id, is_valid, current_evaluation = first_evaluation[current_step_index]

            step_validity = False
            molecule_set = ast.literal_eval(current_step["Molecule set"])
            updated_molecule_set = ast.literal_eval(current_step["Updated molecule set"])
            reaction = ast.literal_eval(current_step["Reaction"])[0]
            product = extract_molecules_from_output(current_step["Product"])
            reactants = ast.literal_eval(current_step["Reactants"])

            # Step 1: Check molecules' validity
            starting_signal = True
            if current_step_index == 0:
                mmd = set(molecule_set).issubset(set(target_smi))
                if not mmd:
                    starting_signal = False

            # Product in molecule set
            product_inside = False
            if product[0] in molecule_set:
                product_inside = True

            invalid_molset_mol_id = []
            invalid_updated_mol_id = []
            
            updated_set_signals = check_validity(updated_molecule_set)
            if False in updated_set_signals:
                invalid_updated_mol_id = [index for index, value in enumerate(updated_set_signals) if not value]
    
            mol_set_signals = check_validity(molecule_set)
            if False in mol_set_signals:
                invalid_molset_mol_id = [index for index, value in enumerate(mol_set_signals) if not value]


            check_availability = False
            unavailable_mol_id = []
            if i == len(route) - 1:
                availabilities = check_purchasable(updated_molecule_set, updated_set_signals, self.inventory)
                check_availability = True
            if check_availability == True:
                if False in availabilities:
                    unavailable_mol_id = [index for index, value in enumerate(availabilities) if not value]
    
            # Step 2
            reaction_valid, updated_set_valid = False, False
            reaction_existence, reaction_key = current_evaluation["reaction_existence"], current_evaluation["reaction_key"]
            
            new_reaction = current_evaluation["reaction"]

            if reaction_existence == True:
                reaction_valid, updated_set_valid = verify_reaction_step(molecule_set, updated_molecule_set, new_reaction, product, reactants, self.inventory, self.oracle)
    
    
            if (
                len(invalid_molset_mol_id) == 0 and
                len(invalid_updated_mol_id) == 0 and
                reaction_valid and
                updated_set_valid and
                starting_signal and
                product_inside
            ):
                step_validity = True
            if step_validity == True:
                self.explored_reaction.add((product[0], new_reaction))
            # Construct the dictionary
            step_info = {
                "target_smi": target_smi,
                "starting_signal": starting_signal,
                "product_inside": product_inside,
                "molecule_set": molecule_set,
                "updated_molecule_set": updated_molecule_set,
                "reaction": new_reaction,
                "reaction_key": reaction_key,
                "product": product,
                "reactants": reactants,
                "updated_set_signals": updated_set_signals,
                "invalid_updated_mol_id": invalid_updated_mol_id,
                "mol_set_signals": mol_set_signals,
                "invalid_molset_mol_id": invalid_molset_mol_id,
                "check_availability": check_availability,
                "unavailable_mol_id": unavailable_mol_id,
                "reaction_existence": reaction_existence,
                "reaction_valid": reaction_valid,
                "updated_set_valid": updated_set_valid
            }

            # Store the tuple in the results list
            results.append((current_step_index, step_validity, step_info))

        return results

    def reset(self) -> None:
        """Reset the optimizer to initial state."""
        del self.oracle
        self.oracle = Oracle(args=self.args)
        self.oracle.route_buffer = {}
        self.oracle.reaction_cache = dict()
        self.explored_reaction = set()
        self.visited_molecules = dict()

    @property
    def route_buffer(self) -> Dict[str, Any]:
        """Get the oracle route buffer."""
        return self.oracle.route_buffer

    @property
    def finish(self) -> bool:
        """Check if optimization is finished."""
        return self.oracle.finish
        
    def _optimize(self, oracle: Oracle, config: Dict[str, Any]) -> None:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError
            
    def rewards(self, route_evaluation: List[Tuple[int, bool, Dict[str, Any]]]) -> float:
        """Calculate reward for a route evaluation."""
        return self.oracle(
            inventory=self.inventory, 
            route_evaluation=route_evaluation, 
            visited_molecules=self.visited_molecules, 
            dead_molecules=self.dead_molecules
        )
    
    def update_cache(
        self, 
        mol_smiles: str, 
        reaction: str
    ) -> None:
        """Update the reaction cache with a new molecule-reaction pair."""
        try:
            s = rdChemReactions.CreateDifferenceFingerprintForReaction(smiles_to_reaction(reaction))
            self.oracle.store_cache(mol_smiles, reaction)
        except:
            pass

    def optimize(
        self, 
        target: str, 
        route_list: List[str], 
        all_fps: List[int], 
        config: Dict[str, Any], 
        seed: int=0
    ) -> None:
        self.reset()
        self.seed = seed 
        self.oracle.task_label = f"{target}_{seed}"
        self._optimize(
            target=target,
            route_list=route_list,
            all_fps=all_fps,
            config=config
        )
        self.save_result(f"{target}_{seed}")

    def fix_reaction_error(
        self, 
        routes: List[Dict[str, Any]], 
        stepwise_results: List[Tuple[int, bool, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Update routes with corrected reactants and molecule sets for valid reactions using template-based retrosynthesis."""
        updated_routes = []
        for i in range(len(routes)):
            step_id, is_valid, data = stepwise_results[i]
            if data["reaction_existence"] == True and data["reaction_valid"] == True:
                reaction_smiles = data["reaction"]

                target_rd = rdchiralReactants(data["product"][0])
                reaction_outputs = run_retro(target_rd, reaction_smiles)
                if len(reaction_outputs) > 1:
                    reaction_outputs = self.rank_reactants(reaction_outputs)
                reactants_generated = [reactant for reactant in reaction_outputs[0]]
                reactants_generated = [sanitize_smiles(smi) for smi in reactants_generated]
                reactants_smiles = set(reactants_generated)

                products_smiles = {smi for smi in data["product"]}
                original_molecule_set = [Chem.MolFromSmiles(smi) for smi in data["molecule_set"]]
                original_set = {Chem.MolToSmiles(mol) for mol in original_molecule_set if mol is not None}

                updated_mol_set = (original_set | reactants_smiles) - products_smiles
                data["Updated molecule set"] = list(updated_mol_set)
                routes[i]["Reactants"] = str(list(reactants_smiles))
                routes[i]["Updated molecule set"] = str(list(updated_mol_set))

        return routes
    
    def rank_reactants(
        self, 
        reactants_list: List[List[str]]
    ) -> List[List[str]]:
        """Rank reactants based on the number of products generated."""
        non_empty_reactant_list = [item for item in reactants_list if item != []]
        scores = [self.oracle.reward(self.inventory, reactant, self.visited_molecules, self.dead_molecules) for reactant in non_empty_reactant_list]
        sorted_list = [x for _, x in sorted(zip(scores, non_empty_reactant_list), key=lambda pair: pair[0], reverse=True)]
        return sorted_list
   