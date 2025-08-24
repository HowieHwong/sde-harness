from typing import Dict, List, Tuple, Any, Optional
import os
import json
import yaml

from rdkit import Chem

from syntheseus import Molecule
from syntheseus.search.mol_inventory import SmilesListInventory
from src.scscore.scscore.standalone_model_numpy import SCScorer
from src.sascore.sascorer import calculateScore



class Oracle:
    """Computes the reward using SCScore and SAScore for a synthetic route."""
    def __init__(
        self, 
        args: Optional[Any] = None, 
        route_buffer: Optional[Dict[str, List[float]]] = None
    ) -> None:
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.route_buffer = {} if route_buffer is None else route_buffer
        self.reaction_cache = dict() # mol_smiles: [reaction]

        self.last_log = 0
        self.sc_Oracle = SCScorer()
        self.sc_Oracle.restore()  # Load the SCScore model

    @property
    def budget(self) -> int:
        """Get the maximum number of oracle calls allowed."""
        return self.max_oracle_calls
    
    @property
    def finish(self) -> bool:
        """Check if the maximum number of oracle calls has been reached."""
        return len(self.route_buffer) >= self.max_oracle_calls
    
    def __len__(self) -> int:
        """Return the number of routes in the buffer."""
        return len(self.route_buffer) 
    
    def __call__(
        self, 
        inventory: SmilesListInventory, 
        route_evaluation: List[Tuple[int, bool, Dict[str, Any]]], 
        visited_molecules: Dict[str, int], 
        dead_molecules: Dict[str, int]
    ) -> float:
        """Callable interface to score a route and handle logging."""
        score_list = self.score_route(
            inventory=inventory, 
            route_evaluation=route_evaluation, 
            visited_molecules=visited_molecules, 
            dead_molecules=dead_molecules
        )
        if len(self.route_buffer) % self.freq_log == 0 and len(self.route_buffer) > self.last_log:
            self.sort_buffer()
            self.last_log = len(self.route_buffer)
            self.save_result(self.task_label)
        return score_list
    
    def reward(
        self, 
        inventory: SmilesListInventory, 
        updated_molecule_set: List[str], 
        visited_molecules: Dict[str, int], 
        dead_molecules: Dict[str, int]
    ) -> float:
        """Calculate reward score for a set of molecules."""
        score_list = []
        for smi in updated_molecule_set:
            if smi in dead_molecules:
                if dead_molecules[smi] >= 1:
                    score_list.append(100)
                    continue
            try:
                signal = inventory.is_purchasable(Molecule(smi))
                if not signal:
                    score = self.get_oracle_score(smi)
                    if smi in visited_molecules:
                        if visited_molecules[smi] > 15:
                            score = (visited_molecules[smi]/15) * score
                    score_list.append(score)
            except Exception:
                score_list.append(5)

        score_mean = sum(score_list) / len(score_list) if len(score_list) != 0 else 0
        combined_score = score_mean + sum(score_list)
        final_score = - combined_score
        return final_score
    
    def store_cache(
        self, 
        mol_smiles: str, 
        reaction: str
    ) -> None:
        """Store a reaction in the cache for a given molecule."""
        if mol_smiles in self.reaction_cache:
            if reaction not in self.reaction_cache[mol_smiles]:
                self.reaction_cache[mol_smiles].append(reaction)
        else:

            self.reaction_cache[mol_smiles] = [reaction]

    def get_oracle_score(
        self, 
        mol_smiles: str
    ) -> float:
        """Get the oracle score for a molecule."""
        smi, SC_score = self.sc_Oracle.get_score_from_smi(mol_smiles)
        sa_score = calculateScore(Chem.MolFromSmiles(mol_smiles))
        length = len(self.route_buffer)

        if length <= 150:
            overall_score = SC_score
        elif length > 150 and length < 220:
            alpha = (length/self.max_oracle_calls)
            overall_score = (1 - alpha) * SC_score + 0.5 * alpha * sa_score            
        else:
            overall_score = 0.5 * sa_score 

        return overall_score
    
    def evaluate(
        self, 
        inventory: SmilesListInventory, 
        route_evaluation: List[Tuple[int, bool, Dict[str, Any]]], 
        visited_molecules: Dict[str, int], 
        dead_molecules: Dict[str, int]
    ) -> float:
        """Evaluate a synthetic route and return its score."""
        for idx, step in enumerate(route_evaluation):
            if step[1] == False:
                score = self.reward(inventory, step[2]["molecule_set"], visited_molecules, dead_molecules)
                return score
            elif step[1] == True:
                continue
        
        # Last step
        if route_evaluation[-1][2]["check_availability"] == True and len(route_evaluation[-1][2]["unavailable_mol_id"]) == 0:
            score = 0
            return score
        else:
            score = self.reward(inventory, route_evaluation[-1][2]["updated_molecule_set"], visited_molecules, dead_molecules)
            return score

    def sort_buffer(self) -> None:
        """Sort the route buffer by scores in descending order."""
        self.route_buffer = dict(sorted(self.route_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix: Optional[str] = None) -> None:
        """Save the route buffer to a YAML file."""
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, "results.yaml")
        else:
            suffix = suffix.replace("/", "")
            output_file_path = os.path.join(self.args.output_dir, "results_" + suffix + ".yaml")

        self.sort_buffer()
        with open(output_file_path, "w") as f:
            yaml.dump(self.route_buffer, f, sort_keys=False)

    def log_intermediate(self, finish: bool = False) -> None:
        """Log intermediate results during optimization."""
        if finish: self.save_result(self.task_label)

    def score_route(
        self, 
        inventory: SmilesListInventory, 
        route_evaluation: List[Tuple[int, bool, Dict[str, Any]]], 
        visited_molecules: Dict[str, int], 
        dead_molecules: Dict[str, int]
    ) -> float:
        """Score a synthetic route and store it in the buffer."""
        if len(self.route_buffer) > self.max_oracle_calls: return -15
        if route_evaluation is None: return -15
        dict_key = json.dumps(route_evaluation)
        
        if dict_key in self.route_buffer: pass
        else:
            self.route_buffer[dict_key] = [float(self.evaluate(
                inventory=inventory, 
                route_evaluation=route_evaluation, 
                visited_molecules=visited_molecules, 
                dead_molecules=dead_molecules
            )), len(self.route_buffer)+1]

        return self.route_buffer[dict_key][0]
