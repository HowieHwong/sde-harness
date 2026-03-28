"""
Core elucidation engine for iterative molecular structure elucidation.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from .llm_interface import LLMInterface, ElucidationStep
from .data_utils import MolecularDataLoader, NMRProcessor
from .nmr_predictor import NMRPredictor


@dataclass
class ElucidationConfig:
    """Configuration for the elucidation process."""
    
    max_iterations: int = 10
    similarity_threshold: float = 0.8
    temperature: float = 0.7
    save_intermediate_results: bool = True
    output_dir: str = "elucidation_results"
    log_level: str = "INFO"
    use_nmr_predictor: bool = True
    nmr_tolerance: float = 0.20
    prefer_c_nmr: bool = True  # Use C-NMR for similarity if available


@dataclass
class ElucidationResult:
    """Result of the elucidation process."""
    
    target_molecule_id: str
    target_nmr: str
    target_c_nmr: Optional[str]
    final_smiles: Optional[str]
    final_similarity: float
    total_iterations: int
    steps: List[ElucidationStep]
    success: bool
    execution_time: float
    metadata: Dict[str, Any]


class ElucidationEngine:
    """Main engine for iterative molecular structure elucidation."""
    
    def __init__(self, 
                 data_loader: MolecularDataLoader,
                 llm_interface: LLMInterface,
                 config: ElucidationConfig,
                 nmr_predictor: Optional[NMRPredictor] = None):
        """
        Initialize the elucidation engine.
        
        Args:
            data_loader: Molecular data loader instance
            llm_interface: LLM interface instance
            config: Elucidation configuration
            nmr_predictor: NMR predictor instance (optional)
        """
        self.data_loader = data_loader
        self.llm_interface = llm_interface
        self.config = config
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize NMR processor
        self.nmr_processor = NMRProcessor()
        
        # Initialize NMR predictor if provided or if enabled in config
        if nmr_predictor:
            self.nmr_predictor = nmr_predictor
        elif config.use_nmr_predictor:
            # Try to get API key from LLM interface
            api_key = getattr(llm_interface, 'api_key', None)
            self.nmr_predictor = NMRPredictor(openai_api_key=api_key)
        else:
            self.nmr_predictor = None
    
    def elucidate_molecule(self, 
                          target_molecule_id: str,
                          max_iterations: Optional[int] = None) -> ElucidationResult:
        """
        Perform iterative molecular structure elucidation.
        
        Args:
            target_molecule_id: ID of the target molecule to elucidate
            max_iterations: Maximum number of iterations (overrides config)
            
        Returns:
            ElucidationResult containing the elucidation process and results
        """
        
        start_time = time.time()
        
        # Get target molecule data
        target_molecule = self.data_loader.get_molecule_by_id(target_molecule_id)
        if not target_molecule:
            raise ValueError(f"Target molecule {target_molecule_id} not found")
        
        target_h_nmr = target_molecule.get('H_NMR', '')
        target_c_nmr = target_molecule.get('C_NMR', '')
        
        if not target_h_nmr and not target_c_nmr:
            raise ValueError(f"No NMR data found for molecule {target_molecule_id}")
        
        self.logger.info(f"Starting elucidation for molecule {target_molecule_id}")
        if target_h_nmr:
            self.logger.info(f"Target H-NMR: {target_h_nmr[:100]}...")
        if target_c_nmr:
            self.logger.info(f"Target C-NMR: {target_c_nmr[:100]}...")
        
        # Initialize elucidation process
        iteration = 1
        max_iter = max_iterations or self.config.max_iterations
        steps = []
        best_similarity = 0.0
        best_smiles = None
        
        while iteration <= max_iter:
            self.logger.info(f"Starting iteration {iteration}/{max_iter}")
            
            # Create prompt for current iteration
            prompt = self.llm_interface.create_elucidation_prompt(
                target_nmr=target_h_nmr or target_c_nmr,
                iteration=iteration,
                history=steps,
                target_molecule_info=target_molecule
            )
            
            # Query LLM
            self.logger.info("Querying LLM...")
            response = self.llm_interface.query_llm(
                prompt, 
                temperature=self.config.temperature
            )
            
            # Extract SMILES from response
            generated_smiles = self.llm_interface.extract_smiles_from_response(response)
            
            # Calculate NMR similarity if SMILES was generated
            nmr_similarity = 0.0
            if generated_smiles and self.llm_interface.validate_smiles(generated_smiles):
                nmr_similarity = self._calculate_nmr_similarity(
                    target_h_nmr, target_c_nmr, generated_smiles
                )
            
            # Create step record
            step = ElucidationStep(
                iteration=iteration,
                prompt=prompt,
                response=response,
                generated_smiles=generated_smiles,
                nmr_similarity=nmr_similarity,
                timestamp=time.time(),
                metadata={
                    'target_molecule_id': target_molecule_id,
                    'temperature': self.config.temperature,
                    'nmr_prediction_method': 'web_scraping' if self.nmr_predictor else 'database_lookup'
                }
            )
            
            steps.append(step)
            
            # Log progress
            self.logger.info(f"Iteration {iteration}: SMILES={generated_smiles}, Similarity={nmr_similarity:.3f}")
            
            # Check if we've reached the similarity threshold
            if nmr_similarity >= self.config.similarity_threshold:
                self.logger.info(f"Target similarity threshold reached: {nmr_similarity:.3f}")
                best_similarity = nmr_similarity
                best_smiles = generated_smiles
                break
            
            # Update best result if this iteration is better
            if nmr_similarity > best_similarity:
                best_similarity = nmr_similarity
                best_smiles = generated_smiles
                self.logger.info(f"New best similarity: {best_similarity:.3f}")
            
            # Save intermediate results if configured
            if self.config.save_intermediate_results:
                self._save_intermediate_results(target_molecule_id, iteration, step)
            
            iteration += 1
            
            # Add delay between iterations to avoid rate limiting
            time.sleep(1)
        
        # Create final result
        execution_time = time.time() - start_time
        success = best_similarity >= self.config.similarity_threshold
        
        result = ElucidationResult(
            target_molecule_id=target_molecule_id,
            target_nmr=target_h_nmr or target_c_nmr,
            target_c_nmr=target_c_nmr,
            final_smiles=best_smiles,
            final_similarity=best_similarity,
            total_iterations=len(steps),
            steps=steps,
            success=success,
            execution_time=execution_time,
            metadata={
                'config': asdict(self.config),
                'target_molecule': target_molecule,
                'nmr_prediction_used': self.nmr_predictor is not None
            }
        )
        
        # Save final results
        self._save_final_results(result)
        
        self.logger.info(f"Elucidation completed: Success={success}, "
                        f"Final Similarity={best_similarity:.3f}, "
                        f"Total Time={execution_time:.2f}s")
        
        return result
    
    def _calculate_nmr_similarity(self, 
                                 target_h_nmr: str, 
                                 target_c_nmr: str, 
                                 generated_smiles: str) -> float:
        """
        Calculate NMR similarity between target and generated molecule.
        
        Args:
            target_h_nmr: Target H-NMR spectrum
            target_c_nmr: Target C-NMR spectrum
            generated_smiles: Generated SMILES structure
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            self.logger.debug(f"Calculating NMR similarity for SMILES: {generated_smiles}")
            self.logger.debug(f"Target H-NMR: {target_h_nmr[:100]}...")
            self.logger.debug(f"Target C-NMR: {target_c_nmr[:100]}...")
            
            # First try to find the generated molecule in our database
            generated_molecule = self.data_loader.search_by_smiles(generated_smiles)
            if generated_molecule:
                self.logger.debug(f"Found generated molecule in database")
                generated_h_nmr = generated_molecule.get('H_NMR', '')
                generated_c_nmr = generated_molecule.get('C_NMR', '')
                
                self.logger.debug(f"Generated H-NMR: {generated_h_nmr[:100]}...")
                self.logger.debug(f"Generated C-NMR: {generated_c_nmr[:100]}...")
                
                # Calculate similarity using database data
                similarities = []
                
                if target_h_nmr and generated_h_nmr:
                    self.logger.debug("Calculating H-NMR similarity...")
                    h_sim = self.nmr_processor.calculate_nmr_similarity(
                        target_h_nmr, generated_h_nmr, tolerance=0.05  # Use tighter tolerance for H-NMR
                    )
                    similarities.append(h_sim)
                    self.logger.debug(f"H-NMR similarity: {h_sim:.3f}")
                
                if target_c_nmr and generated_c_nmr:
                    self.logger.debug("Calculating C-NMR similarity...")
                    c_sim = self.nmr_processor.calculate_nmr_similarity(
                        target_c_nmr, generated_c_nmr, tolerance=0.05  # Use tighter tolerance for C-NMR
                    )
                    similarities.append(c_sim)
                    self.logger.debug(f"C-NMR similarity: {c_sim:.3f}")
                
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    self.logger.debug(f"Average similarity from database: {avg_similarity:.3f}")
                    return avg_similarity
                else:
                    self.logger.warning("No NMR data available for similarity calculation")
            else:
                self.logger.debug(f"Generated molecule not found in database")
            
            # If not found in database and NMR predictor is available, use it
            if self.nmr_predictor:
                self.logger.info(f"Predicting NMR for generated SMILES: {generated_smiles}")
                
                # Get NMR prediction
                c_nmr_records, h_nmr_records = self.nmr_predictor.get_nmr_prediction(
                    generated_smiles, fallback_to_llm=True
                )
                
                similarities = []
                
                # Calculate H-NMR similarity if available
                if target_h_nmr and h_nmr_records:
                    predicted_h_nmr = self.nmr_predictor.format_nmr_for_comparison(h_nmr_records, "H")
                    h_sim = self.nmr_predictor.calculate_nmr_similarity(
                        target_h_nmr, predicted_h_nmr, "H", self.config.nmr_tolerance
                    )
                    similarities.append(h_sim)
                
                # Calculate C-NMR similarity if available (preferred for accuracy)
                if target_c_nmr and c_nmr_records:
                    predicted_c_nmr = self.nmr_predictor.format_nmr_for_comparison(c_nmr_records, "C")
                    c_sim = self.nmr_predictor.calculate_nmr_similarity(
                        target_c_nmr, predicted_c_nmr, "C", self.config.nmr_tolerance
                    )
                    similarities.append(c_sim)
                
                if similarities:
                    # Weight C-NMR more heavily if preferred
                    if self.config.prefer_c_nmr and len(similarities) > 1:
                        if target_c_nmr and c_nmr_records:
                            # C-NMR gets higher weight
                            weighted_sim = (c_sim * 0.7 + h_sim * 0.3) if 'h_sim' in locals() else c_sim
                            self.logger.debug(f"Weighted similarity from NMR predictor: {weighted_sim:.3f}")
                            return weighted_sim
                        else:
                            self.logger.debug(f"H-NMR similarity from NMR predictor: {h_sim:.3f}")
                            return h_sim
                    else:
                        avg_sim = sum(similarities) / len(similarities)
                        self.logger.debug(f"Average similarity from NMR predictor: {avg_sim:.3f}")
                        return avg_sim
            
            # Fallback: return 0 if no similarity can be calculated
            self.logger.warning(f"Could not calculate NMR similarity for {generated_smiles}")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating NMR similarity: {e}")
            return 0.0
    
    def elucidate_random_molecule(self, max_iterations: Optional[int] = None) -> ElucidationResult:
        """
        Elucidate a random molecule from the dataset.
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            ElucidationResult for the random molecule
        """
        random_molecule = self.data_loader.get_random_molecule()
        if not random_molecule:
            raise ValueError("No molecules available in dataset")
        
        molecule_id = random_molecule['molecule_id']
        self.logger.info(f"Selected random molecule: {molecule_id}")
        
        return self.elucidate_molecule(molecule_id, max_iterations)
    
    def batch_elucidation(self, 
                         molecule_ids: List[str], 
                         max_iterations: Optional[int] = None) -> List[ElucidationResult]:
        """
        Perform elucidation on multiple molecules.
        
        Args:
            molecule_ids: List of molecule IDs to elucidate
            max_iterations: Maximum iterations per molecule
            
        Returns:
            List of ElucidationResult objects
        """
        results = []
        
        for i, molecule_id in enumerate(molecule_ids):
            self.logger.info(f"Processing molecule {i+1}/{len(molecule_ids)}: {molecule_id}")
            
            try:
                result = self.elucidate_molecule(molecule_id, max_iterations)
                results.append(result)
                
                # Save batch results
                self._save_batch_result(result, i+1, len(molecule_ids))
                
            except Exception as e:
                self.logger.error(f"Error elucidating molecule {molecule_id}: {e}")
                # Create error result
                error_result = ElucidationResult(
                    target_molecule_id=molecule_id,
                    target_nmr="",
                    target_c_nmr="",
                    final_smiles=None,
                    final_similarity=0.0,
                    total_iterations=0,
                    steps=[],
                    success=False,
                    execution_time=0.0,
                    metadata={'error': str(e)}
                )
                results.append(error_result)
        
        return results
    
    def _save_intermediate_results(self, molecule_id: str, iteration: int, step: ElucidationStep):
        """Save intermediate results for a single iteration."""
        output_file = self.output_dir / f"{molecule_id}_iter_{iteration}.json"
        
        step_data = asdict(step)
        step_data['molecule_id'] = molecule_id
        
        with open(output_file, 'w') as f:
            json.dump(step_data, f, indent=2, default=str)
    
    def _save_final_results(self, result: ElucidationResult):
        """Save final elucidation results."""
        output_file = self.output_dir / f"{result.target_molecule_id}_final.json"
        
        result_data = asdict(result)
        
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
    
    def _save_batch_result(self, result: ElucidationResult, current: int, total: int):
        """Save batch elucidation result."""
        batch_dir = self.output_dir / "batch_results"
        batch_dir.mkdir(exist_ok=True)
        
        output_file = batch_dir / f"batch_{current:03d}_of_{total:03d}_{result.target_molecule_id}.json"
        
        result_data = asdict(result)
        
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
    
    def get_elucidation_summary(self, result: ElucidationResult) -> Dict[str, Any]:
        """
        Generate a summary of the elucidation process.
        
        Args:
            result: ElucidationResult to summarize
            
        Returns:
            Dictionary containing summary information
        """
        summary = {
            'target_molecule_id': result.target_molecule_id,
            'success': result.success,
            'final_similarity': result.final_similarity,
            'best_similarity': result.final_similarity,  # Add this for compatibility
            'total_iterations': result.total_iterations,
            'execution_time': result.execution_time,
            'best_smiles': result.final_smiles,
            'nmr_prediction_used': result.metadata.get('nmr_prediction_used', False),
            'similarity_progression': [
                {
                    'iteration': step.iteration,
                    'similarity': step.nmr_similarity,
                    'smiles': step.generated_smiles,
                    'prediction_method': step.metadata.get('nmr_prediction_method', 'unknown')
                }
                for step in result.steps
            ],
            'improvement_rate': self._calculate_improvement_rate(result.steps)
        }
        
        return summary
    
    def _calculate_improvement_rate(self, steps: List[ElucidationStep]) -> float:
        """Calculate the rate of improvement across iterations."""
        if len(steps) < 2:
            return 0.0
        
        similarities = [step.nmr_similarity for step in steps]
        improvements = [similarities[i] - similarities[i-1] for i in range(1, len(similarities))]
        
        if not improvements:
            return 0.0
        
        return sum(improvements) / len(improvements)
