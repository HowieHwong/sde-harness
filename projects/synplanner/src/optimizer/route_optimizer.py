from __future__ import print_function
from typing import List, Tuple, Dict, Any
import re 
import copy
import logging
import ast
import random
import math
import numpy as np

from rdkit import rdBase
rdBase.DisableLog("rdApp.error")

from src.optimizer.base_optimizer import BaseOptimizer, extract_molecules_from_output
from concurrent.futures import ThreadPoolExecutor, wait
from src.optimizer.route import Route, check_distinct_route
from src.utils.prompt import construct_initialization_prompt, modification_hints, construct_mutation_prompt
from src.utils.utils import query_LLM
from src.utils.chemistry_utils import process_reaction_routes, sanitize_smiles, check_availability, MORGAN_FP_GENERATOR, SIMILARITY_METRIC
from syntheseus.search.mol_inventory import SmilesListInventory

MINIMUM = 1e-10


def make_mating_pool(
    combined_list: List[Tuple[float, Route]],
    population_scores: List[float],
    visited_cache: Dict[str, int],
    inventory: SmilesListInventory,
    offspring_size: int
) -> List[Tuple[float, Route]]:
    """
    Create a mating pool by selecting routes based on probability distribution derived from their node values.
    Probabilities are computed by exponentially inverting node values that combine population scores with visit penalties.
    """
    # Scores -> probs 
    population_scores = [
        node_value(
            combined_list=combined_list[i],
            population_score=population_scores[i],
            visited_cache=visited_cache,
            inventory=inventory
        ) for i in range(len(population_scores))
    ]
    weights = np.exp(-np.array(population_scores))  # Exponentially invert scores
    probabilities = weights / weights.sum()  # Normalize to get probabilities
    while True:
        sampled_index = np.random.choice(len(combined_list), p=probabilities, size=offspring_size, replace=True)
        if len(set(sampled_index)) > 1:
            break
    mating_pool = [combined_list[i] for i in sampled_index]
    return mating_pool

def node_value(
    combined_list: Tuple[float, Route],
    population_score: float,
    visited_cache: Dict[str, int],
    inventory: SmilesListInventory,
    C: float = 0.3
) -> float:
    """
    Calculate node value combining population score with UCB exploration bonus.
    Returns 1 if all molecules are purchasable, otherwise penalizes over-visited nodes.
    """
    smi_list = combined_list[1].validated_route[-1]["Updated molecule set"]
    unpurchasable_list = check_availability(
        smi_list=smi_list,
        inventory=inventory
    )

    if len(unpurchasable_list) == 0: return 1
    total_visits = sum(visited_cache.values())
    node_visits = 1

    for molecule in unpurchasable_list:
        if molecule in visited_cache:
            node_visits += visited_cache[molecule]
            if visited_cache[molecule] > 25:
                population_score = -10

    if total_visits == 0: final_score = - population_score
    else: final_score = - population_score - C * math.sqrt(math.log(total_visits) / node_visits)

    return final_score

class RouteOptimizer(BaseOptimizer):
    """Optimizer for finding synthetic routes using LLM-guided search."""
    def __init__(
        self,
        args: Dict[str, Any]
    ) -> None:
        super().__init__(args)
        self.llm_calls = 0

    def initialization(
        self,
        target_smi: str, 
        rag_tuples: List[Tuple[float, List[str]]],
        temperature: float
    ) -> Route:
        """
        Initializes synthetic route design process by sampling 3 reference routes and querying the LLM for an initial route.
        """
        # 1. Sample 3 reference routes from the population with probability proportional to similarity scores
        sims_list, route_list = zip(*rag_tuples)
        sum_scores = sum(sims_list)
        population_probs = [p / sum_scores for p in sims_list]

        while True:
            try:
                sampled_index = np.random.choice(len(route_list), p=population_probs, size=3, replace=False)
                sampled_routes = [route_list[i] for i in sampled_index]
                example_routes = "\n".join(
                    f"<ROUTE>\n{str(process_reaction_routes(route))}\n</ROUTE>\n"
                    for route in sampled_routes
                )

                # Create the INITIALIZATION prompt (First step of route design)
                initialization_prompt = construct_initialization_prompt(
                    target_smiles=target_smi,
                    example_routes=example_routes
                )

                # Query the LLM for the initial route
                _, answer = query_LLM(
                    query=initialization_prompt,
                    temperature=temperature,
                )
                self.llm_calls += 1

                # Extract the route from the LLM response
                match = re.search(r"<ROUTE>(.*?)<ROUTE>", answer, re.DOTALL)
                if match == None: match = re.search(r"<ROUTE>(.*?)</ROUTE>", answer, re.DOTALL)

                route_content = match.group(1)

                route = ast.literal_eval(route_content)
                comp1 = ast.literal_eval(route[-1]["Updated molecule set"])
                comp2 = ast.literal_eval(route[-2]["Updated molecule set"])
                last_step_reactants = route[-1]["Reactants"]

                if (
                    set(comp1) == set(comp2) or 
                    last_step_reactants == "" or 
                    last_step_reactants == "[]" or 
                    last_step_reactants == "None" or 
                    last_step_reactants == "[None]"
                ):
                    route = route[:-1]

                for step in route:
                    temp = ast.literal_eval(step["Molecule set"])
                    temp = ast.literal_eval(step["Reaction"])[0]
                    products, reactants = temp.split(">>")
                    temp = extract_molecules_from_output(step["Product"])[0]
                    temp = ast.literal_eval(step["Reactants"])[0]
                    temp = ast.literal_eval(step["Updated molecule set"])

                # After parsing the LLM output, sanitize the route
                route_class_item = Route(target_smi)
                logging.info(f"INITIALIZATION phase: Sanitizing (running checks) the LLM generated route...")
                checked_route, final_evaluation = self.sanitize(
                    starting_list=[target_smi],
                    route=route,
                    exploration_signal=True  # Search and rank 1000 reference reactions instead of 100
                )

                if final_evaluation[0][2]["reaction_existence"] == False and final_evaluation[0][2]["product_inside"] == True:
                    self.dead_molecules.append(final_evaluation[0][2]["product"][0])
                    continue
                if final_evaluation[0][1] == False: continue

                # Calculate the reward score for the route
                score = self.rewards(final_evaluation)
                route_class_item.add_route(checked_route, final_evaluation)
                route_class_item.update_reward(score)
                route_class_item.update_evaluation(final_evaluation)
                break

            except Exception as e:
                logging.error(f"INITIALIZATION ERROR (usually LLM proposed route error and is benign): {e} - Retrying...")
                continue

        return route_class_item

    def modification(
        self,
        combined_list: List[Tuple[float, Route]],
        population_routes: List[Route],
        all_fps: List[int],
        route_list: List[List[str]], 
        inventory: SmilesListInventory,  
        temperature: float
    ) -> Route:
        """Modify existing routes to generate new candidate routes."""
        # Randomly select a parent route from the mating pool
        parent_a = random.choice(combined_list)
        sampled_route = parent_a[1]
        count = 0
        final_route_item = None
        for _ in range(5):
            try:
                count = count + 1
                # If we've failed 5 times, select a new parent route
                if count >= 5:
                    count = 0
                    parent_a = random.choice(combined_list)
                    sampled_route = parent_a[1]
                route = sampled_route.validated_route
                new_route_item = copy.deepcopy(sampled_route)
                evaluation = new_route_item.evaluation
                smi_list = route[-1]["Updated molecule set"]
                # Find which molecules are not available for purchase
                nonpurchasable_list = check_availability(
                    smi_list=smi_list,
                    inventory=inventory
                )

                # Retrieve relevant routes based on similarity to non-purchasable molecules
                retrieved_routes = modification_hints(
                    nonpurchasable_molecule=nonpurchasable_list,
                    all_fps=all_fps,
                    route_list=route_list
                )
                # Construct modification prompt with context from non-purchasable molecules and relevant routes
                mutation_prompt = construct_mutation_prompt(
                    molecule_smi_list=nonpurchasable_list,
                    examples=retrieved_routes,
                    evaluation=evaluation,
                    inventory=inventory,
                    reaction_cache=self.oracle.reaction_cache
                )
                # Query the LLM to generate a modified route
                _, new_a = query_LLM(
                    query=mutation_prompt,
                    temperature=temperature
                )
                self.llm_calls += 1

                # Extract the modified route from LLM response
                match = re.search(r"<ROUTE>(.*?)<ROUTE>", new_a, re.DOTALL)
                if match == None: match = re.search(r'<ROUTE>(.*?)</ROUTE>', new_a, re.DOTALL)

                route_content = match.group(1)

                new_route = ast.literal_eval(route_content)

                # Clean up the route by removing redundant last steps
                comp1 = ast.literal_eval(new_route[-1]["Updated molecule set"])
                comp2 = ast.literal_eval(new_route[-2]["Updated molecule set"])
                last_step_reactants = new_route[-1]["Reactants"]

                if (
                    set(comp1) == set(comp2) or 
                    last_step_reactants == "" or 
                    last_step_reactants == "[]" or 
                    last_step_reactants == "None" or 
                    last_step_reactants == "[None]"
                ):
                    new_route = new_route[:-1]

                # Validate the route format and extract key components
                for idx, step in enumerate(new_route):
                    temp = ast.literal_eval(step["Molecule set"])
                    reaction = ast.literal_eval(step["Reaction"])[0]
                    products, reactants = reaction.split(">>")
                    product = extract_molecules_from_output(step["Product"])[0]
                    temp = ast.literal_eval(step["Reactants"])[0]
                    temp = ast.literal_eval(step["Updated molecule set"])
                    # For the first step, update reaction cache if product matches
                    if idx == 0:
                        if products == product:
                            self.update_cache(product, reaction)

                # Sanitize the modified route to ensure chemical validity
                logging.info(f"MUTATION phase: Sanitizing (running checks) the LLM generated route...")
                checked_route, final_evaluation = self.sanitize(
                    starting_list=smi_list,
                    route=new_route,
                    exploration_signal=False  # During MUTATION, explore less reactions
                )
                # Update visited molecules cache
                if final_evaluation[0][2]["product_inside"] == True:
                    self.update_visited_molecules(final_evaluation[0][2]["product"])

                # Check if the reaction exists in the database
                if final_evaluation[0][2]["reaction_existence"] == False: 
                    if final_evaluation[0][2]["product_inside"] == True: self.update_dead_molecules(final_evaluation[0][2]["product"][0])
                    continue
                # Update the route with validated information
                new_route_item.update_route(checked_route, final_evaluation)
                # Check if this route is distinct from existing population
                if not check_distinct_route(population_routes, new_route_item): continue
                # Calculate reward score for the new route
                score = self.rewards(final_evaluation)
                
                new_route_item.update_reward(score)
                new_route_item.update_evaluation(final_evaluation)
                final_route_item = new_route_item
                break

            except Exception as e:
                logging.error(f"MUTATION ERROR (usually LLM proposed route error and is benign): {e} - Retrying...")
                continue

        return final_route_item

    def _optimize(
        self,
        target: str,
        route_list: List[List[str]],
        all_fps: List[int],
        config: Dict[str, Any]
    ) -> None:
        """Main optimization loop for finding synthetic routes."""
        
        target = sanitize_smiles(target)
        fp = MORGAN_FP_GENERATOR(target)
        sims = SIMILARITY_METRIC(fp, [fp_ for fp_ in all_fps])

        # ----------------------------------------------------
        # INITIALIZATION
        # ----------------------------------------------------
        logging.info("--- INITIALIZATION ---")
        logging.info("Retrieving reference routes...")
        
        # 1. Retrieve most similar routes to the target molecule as reference routes
        rag_tuples = list(zip(sims, route_list))
        rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)
        assert len(self.oracle.route_buffer) == 0, f"route_buffer not empty: {self.oracle.route_buffer}"

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.initialization, target, rag_tuples, float(config["temperature"])) for _ in range(config["population_size"])]
            starting_population = [future.result() for future in futures]

        # Initial population
        population_routes = starting_population
        population_scores = [route_class_item.get_reward() for route_class_item in population_routes]
        combined_list = list(zip(population_scores, population_routes))
        all_routes = copy.deepcopy(population_routes)

        # ----------------------------------------------------
        # MUTATION
        # ----------------------------------------------------
        logging.info("--- MUTATION ---")
        logging.info("Modifying routes to find a solved synthetic route...")
        while True:
            if len(self.oracle) > 5:
                self.sort_buffer()
                if 0 in population_scores:
                    self.log_intermediate(finish=True) 
                    logging.info(f"Solved a route in {self.llm_calls} LLM calls, ending...")
                    for route in population_routes:
                        reward = route.get_reward()
                        if reward == 0:
                            route.save_result(
                                results_dir=self.args.output_dir,
                                suffix=target
                            )
                            
                    break

            mating_pool = make_mating_pool(
                combined_list=combined_list,
                population_scores=population_scores,
                visited_cache=self.visited_molecules,
                inventory=self.inventory,
                offspring_size=config["population_size"]
            )

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.modification, mating_pool, all_routes, all_fps, route_list, self.inventory, float(config["temperature"])) for _ in range(config["offspring_size"])]
                done, not_done = wait(futures, timeout=300)
                returned_routes = [future.result() for future in done]
                offspring_routes = []
                for i in range(len(returned_routes)):
                    if returned_routes[i] is not None:
                        if check_distinct_route(offspring_routes, returned_routes[i]): offspring_routes.append(returned_routes[i])

            # Add new population
            population_routes += offspring_routes
            all_routes = all_routes + offspring_routes

            # ----------------------------------------------------
            # SELECTION
            # ----------------------------------------------------
            all_scores = [self.oracle.reward(
                inventory=self.inventory,
                updated_molecule_set=route_class_item.validated_route[-1]["Updated molecule set"],
                visited_molecules=self.visited_molecules,
                dead_molecules=self.dead_molecules
            ) for route_class_item in all_routes]

            combined_list = list(zip(all_scores, all_routes))
            combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_routes = [t[1] for t in combined_list]
            population_scores = [t[0] for t in combined_list]

            # Stop criterion
            if len(self.oracle) > 5:
                if 0 in population_scores:
                    self.log_intermediate(finish=True)  
                    logging.info(f"Found a route in {self.llm_calls} LLM calls, ending...")
                    for route in population_routes:
                        reward = route.get_reward()
                        if reward == 0:
                            route.save_result(
                                results_dir=self.args.output_dir,
                                suffix=target
                            )
                            
                    break
  
            if self.finish:   
                logging.info(f"Finished in {self.llm_calls} LLM calls. Route not found, ending...")              
                break
