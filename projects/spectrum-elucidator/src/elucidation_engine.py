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
from .similarity import get_1H_13C_peaks, compare_c_nmr_strings, parse_c_nmr


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
    max_per_iter_retries: int = 2  # Try alternate SMILES within the same iteration if NMR not found


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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Track last predicted/fetched NMR strings for step-level persistence
        self._last_predicted_h_nmr: Optional[str] = None
        self._last_predicted_c_nmr: Optional[str] = None
        self._last_db_match_id: Optional[str] = None
        self._last_similarity_method: str = 'unknown'
        self._current_target_smiles: Optional[str] = None
        self._last_db_matched_smiles: Optional[str] = None
    
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
        # No non-NMR similarity fallbacks; rely solely on NMR
        
        while iteration <= max_iter:
            self.logger.info(f"Starting iteration {iteration}/{max_iter}")

            attempts = 0
            generated_smiles = None
            response = ""
            prompt = ""
            nmr_similarity = 0.0

            while attempts <= self.config.max_per_iter_retries:
                # Build prompt: initial or refinement
                if attempts == 0:
                    prompt = self.llm_interface.create_elucidation_prompt(
                        target_nmr=target_h_nmr or target_c_nmr,
                        iteration=iteration,
                        history=steps,
                        target_molecule_info=target_molecule
                    )
                else:
                    # Use refinement prompt to avoid repeating invalid candidates
                    prev_smiles = generated_smiles or ""
                    prev_sim = nmr_similarity
                    prompt = self.llm_interface.create_refinement_prompt(
                        target_nmr=target_h_nmr or target_c_nmr,
                        current_smiles=prev_smiles,
                        current_similarity=prev_sim,
                        iteration=iteration,
                    )

                # Query LLM
                self.logger.info("Querying LLM...")
                response = self.llm_interface.query_llm(
                    prompt, temperature=self.config.temperature
                )

                # Extract and validate SMILES
                candidate = self.llm_interface.extract_smiles_from_response(response)
                if not candidate or not self.llm_interface.validate_smiles(candidate):
                    attempts += 1
                    continue
                # Canonicalize before scoring/storage
                candidate = self._canonicalize_smiles(candidate)

                # Already canonicalized above

                # Reset per-attempt predicted strings
                self._last_predicted_h_nmr = None
                self._last_predicted_c_nmr = None
                self._last_db_match_id = None
                self._last_db_matched_smiles = None

                # Calculate similarity; accept only if we could fetch NMR (database or web)
                nmr_similarity = self._calculate_nmr_similarity(
                    target_h_nmr, target_c_nmr, candidate
                )
                fetch_status = getattr(self, '_last_nmr_fetch_status', 'unknown')

                if fetch_status in ('database', 'web'):
                    # If matched to DB, prefer the dataset SMILES string
                    if fetch_status == 'database' and getattr(self, '_last_db_matched_smiles', None):
                        generated_smiles = self._last_db_matched_smiles
                    else:
                        generated_smiles = candidate
                    break

                # Otherwise retry with refinement
                attempts += 1

            # If still no valid fetch, keep the last candidate (may be None)
            if generated_smiles is None:
                generated_smiles = candidate if 'candidate' in locals() else None

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
                    'nmr_prediction_method': getattr(self, '_last_similarity_method', 'unknown'),
                    'nmr_fetch_status': getattr(self, '_last_nmr_fetch_status', 'unknown'),
                    'predicted_h_nmr': getattr(self, '_last_predicted_h_nmr', None),
                    'predicted_c_nmr': getattr(self, '_last_predicted_c_nmr', None),
                    'db_match_id': getattr(self, '_last_db_match_id', None),
                    'per_iter_retry_count': attempts,
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

    def elucidate_from_nmr(self,
                           target_molecule_id: str,
                           target_h_nmr: Optional[str] = "",
                           target_c_nmr: Optional[str] = "",
                           max_iterations: Optional[int] = None) -> ElucidationResult:
        """
        Perform iterative elucidation starting from provided target NMR strings
        instead of loading a molecule from the dataset.

        Args:
            target_molecule_id: Identifier for this custom target (used for saving files)
            target_h_nmr: Target 1H NMR spectrum string
            target_c_nmr: Target 13C NMR spectrum string
            max_iterations: Maximum number of iterations (overrides config)

        Returns:
            ElucidationResult with the elucidation process and results
        """
        start_time = time.time()

        if not (target_h_nmr or target_c_nmr):
            raise ValueError("At least one of target_h_nmr or target_c_nmr must be provided")

        self.logger.info(f"Starting elucidation from provided NMR for {target_molecule_id}")
        if target_h_nmr:
            self.logger.info(f"Target H-NMR: {target_h_nmr[:100]}...")
        if target_c_nmr:
            self.logger.info(f"Target C-NMR: {target_c_nmr[:100]}...")

        iteration = 1
        max_iter = max_iterations or self.config.max_iterations
        steps: List[ElucidationStep] = []

        best_similarity = 0.0
        best_smiles: Optional[str] = None

        while iteration <= max_iter:
            # Build a minimal info dict for prompt context
            target_info = {
                'molecule_id': target_molecule_id,
                'H_NMR': target_h_nmr,
                'C_NMR': target_c_nmr,
            }

            attempts = 0
            response = ""
            prompt = ""
            generated_smiles = None
            nmr_similarity = 0.0

            while attempts <= self.config.max_per_iter_retries:
                if attempts == 0:
                    prompt = self.llm_interface.create_elucidation_prompt(
                        target_nmr=target_h_nmr or target_c_nmr,
                        iteration=iteration,
                        history=steps,
                        target_molecule_info=target_info,
                    )
                else:
                    prev_smiles = generated_smiles or ""
                    prev_sim = nmr_similarity
                    prompt = self.llm_interface.create_refinement_prompt(
                        target_nmr=target_h_nmr or target_c_nmr,
                        current_smiles=prev_smiles,
                        current_similarity=prev_sim,
                        iteration=iteration,
                    )

                self.logger.info("Querying LLM...")
                response = self.llm_interface.query_llm(
                    prompt,
                    temperature=self.config.temperature,
                )

                candidate = self.llm_interface.extract_smiles_from_response(response)
                if not candidate or not self.llm_interface.validate_smiles(candidate):
                    attempts += 1
                    continue

                self._last_predicted_h_nmr = None
                self._last_predicted_c_nmr = None
                self._last_db_match_id = None
                self._last_db_matched_smiles = None

                nmr_similarity = self._calculate_nmr_similarity(
                    target_h_nmr or "",
                    target_c_nmr or "",
                    candidate,
                )

                fetch_status = getattr(self, '_last_nmr_fetch_status', 'unknown')
                if fetch_status in ('database', 'web'):
                    if fetch_status == 'database' and getattr(self, '_last_db_matched_smiles', None):
                        generated_smiles = self._last_db_matched_smiles
                    else:
                        generated_smiles = candidate
                    break
                attempts += 1

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
                    'nmr_prediction_method': getattr(self, '_last_similarity_method', 'unknown'),
                    'nmr_fetch_status': getattr(self, '_last_nmr_fetch_status', 'unknown'),
                    'predicted_h_nmr': getattr(self, '_last_predicted_h_nmr', None),
                    'predicted_c_nmr': getattr(self, '_last_predicted_c_nmr', None),
                    'db_match_id': getattr(self, '_last_db_match_id', None),
                    'per_iter_retry_count': attempts,
                },
            )

            steps.append(step)
            self.logger.info(f"Iteration {iteration}: SMILES={generated_smiles}, Similarity={nmr_similarity:.3f}")

            if nmr_similarity >= self.config.similarity_threshold:
                self.logger.info(f"Target similarity threshold reached: {nmr_similarity:.3f}")
                best_similarity = nmr_similarity
                best_smiles = generated_smiles
                break

            if nmr_similarity > best_similarity:
                best_similarity = nmr_similarity
                best_smiles = generated_smiles
                self.logger.info(f"New best similarity: {best_similarity:.3f}")

            if self.config.save_intermediate_results:
                self._save_intermediate_results(target_molecule_id, iteration, step)

            iteration += 1
            time.sleep(1)

        execution_time = time.time() - start_time
        success = best_similarity >= self.config.similarity_threshold

        result = ElucidationResult(
            target_molecule_id=target_molecule_id,
            target_nmr=target_h_nmr or target_c_nmr,
            target_c_nmr=target_c_nmr or "",
            final_smiles=best_smiles,
            final_similarity=best_similarity,
            total_iterations=len(steps),
            steps=steps,
            success=success,
            execution_time=execution_time,
            metadata={
                'config': asdict(self.config),
                'target_molecule': target_info,
                'nmr_prediction_used': self.nmr_predictor is not None,
            },
        )

        self._save_final_results(result)
        self.logger.info(
            f"Elucidation (from NMR) completed: Success={success}, Final Similarity={best_similarity:.3f}, Total Time={execution_time:.2f}s"
        )

        return result
    
    def _canonicalize_smiles(self, smiles: str) -> str:
        """Return canonical SMILES using RDKit if available; else return input.

        This normalizes LLM-generated candidates to a canonical form so that
        DB lookups and comparisons are consistent across iterations.
        """
        try:
            from rdkit import Chem  # type: ignore
        except Exception:
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return smiles

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
            # Default: unknown fetch status (used to inform prompts)
            self._last_nmr_fetch_status = 'unknown'
            self._last_similarity_method = 'unknown'
            self.logger.debug(f"Calculating NMR similarity for SMILES: {generated_smiles}")
            self.logger.debug(f"Target H-NMR: {target_h_nmr[:100]}...")
            self.logger.debug(f"Target C-NMR: {target_c_nmr[:100]}...")
            
            # First try to find the generated molecule in our database (if available)
            generated_molecule = None
            if getattr(self, 'data_loader', None) is not None:
                generated_molecule = self.data_loader.search_by_smiles(generated_smiles)
            if generated_molecule:
                self.logger.debug(f"Found generated molecule in database")
                generated_h_nmr = generated_molecule.get('H_NMR', '')
                generated_c_nmr = generated_molecule.get('C_NMR', '')
                
                self.logger.debug(f"Generated H-NMR: {generated_h_nmr[:100]}...")
                self.logger.debug(f"Generated C-NMR: {generated_c_nmr[:100]}...")
                # Persist last fetched NMR strings and match id for this iteration
                self._last_predicted_h_nmr = generated_h_nmr or None
                self._last_predicted_c_nmr = generated_c_nmr or None
                self._last_db_match_id = generated_molecule.get('molecule_id') or None
                # Keep the dataset SMILES to report canonical dataset form
                self._last_db_matched_smiles = generated_molecule.get('SMILES') or None
                
                # Calculate similarity using database data
                similarities = []
                
                if target_h_nmr and generated_h_nmr:
                    self.logger.debug("Calculating H-NMR similarity (database)...")
                    t_peaks = parse_c_nmr(target_h_nmr, keep_ranges_as_center=True)
                    g_peaks = parse_c_nmr(generated_h_nmr, keep_ranges_as_center=True)
                    h_sim = 0.0
                    if t_peaks and g_peaks:
                        matched = 0
                        total = max(len(t_peaks), len(g_peaks))
                        for tp in t_peaks:
                            for gp in g_peaks:
                                if abs(tp - gp) <= 0.05:
                                    matched += 1
                                    break
                        h_sim = matched / total if total else 0.0
                    similarities.append(h_sim)
                    self.logger.debug(f"H-NMR similarity: {h_sim:.3f}")
                
                if target_c_nmr and generated_c_nmr:
                    self.logger.debug("Calculating C-NMR similarity (database)...")
                    c_res = compare_c_nmr_strings(
                        target_c_nmr, generated_c_nmr, tol_ppm=0.05, keep_ranges_as_center=True
                    )
                    c_sim = c_res["metrics"]["f1"]
                    similarities.append(c_sim)
                    self.logger.debug(f"C-NMR similarity: {c_sim:.3f}")
                
                if similarities:
                    self._last_nmr_fetch_status = 'database'
                    self._last_similarity_method = 'database_lookup'
                    avg_similarity = sum(similarities) / len(similarities)
                    self.logger.debug(f"Average similarity from database: {avg_similarity:.3f}")
                    return avg_similarity
                else:
                    self.logger.warning("No NMR data available for similarity calculation from database")
                    self._last_nmr_fetch_status = 'not_found'
            else:
                self.logger.debug(f"Generated molecule not found in database")
            # Prefer NMRShiftDB automation interface (CML) as primary prediction source
            try:
                self.logger.info(f"Fetching NMR predictions from NMRShiftDB automation for: {generated_smiles}")
                peak_sets = get_1H_13C_peaks(generated_smiles)
                similarities = []
                h_sim = None
                c_sim = None

                # Format and persist predicted 1H peaks as a simple δ list
                if target_h_nmr and peak_sets.get("1H"):
                    predicted_h_nmr = "δ " + ", ".join(f"{p['ppm']:.2f}" for p in peak_sets["1H"])
                    self._last_predicted_h_nmr = predicted_h_nmr
                    # Simple H-NMR matching similar to predictor's method
                    t_peaks = parse_c_nmr(target_h_nmr, keep_ranges_as_center=True)
                    p_peaks = parse_c_nmr(predicted_h_nmr, keep_ranges_as_center=True)
                    if t_peaks and p_peaks:
                        matched = 0
                        total = max(len(t_peaks), len(p_peaks))
                        for tp in t_peaks:
                            for pp in p_peaks:
                                if abs(tp - pp) <= self.config.nmr_tolerance:
                                    matched += 1
                                    break
                        h_sim = matched / total if total else 0.0
                        similarities.append(h_sim)

                # Format and persist predicted 13C peaks
                if target_c_nmr and peak_sets.get("13C"):
                    predicted_c_nmr = "δ " + ", ".join(f"{p['ppm']:.2f}" for p in peak_sets["13C"])
                    self._last_predicted_c_nmr = predicted_c_nmr
                    c_res = compare_c_nmr_strings(
                        target_c_nmr,
                        predicted_c_nmr,
                        tol_ppm=self.config.nmr_tolerance,
                        keep_ranges_as_center=True,
                    )
                    c_sim = c_res["metrics"]["f1"]
                    similarities.append(c_sim)

                if similarities:
                    self._last_nmr_fetch_status = 'web'
                    self._last_similarity_method = 'nmrshiftdb'
                    if self.config.prefer_c_nmr and (c_sim is not None):
                        weighted = (c_sim * 0.7 + (h_sim or 0.0) * 0.3) if (h_sim is not None) else c_sim
                        self.logger.debug(f"Weighted similarity from NMRShiftDB: {weighted:.3f}")
                        return weighted
                    avg = sum(similarities) / len(similarities)
                    self.logger.debug(f"Average similarity from NMRShiftDB: {avg:.3f}")
                    return avg
            except Exception as ee:
                self.logger.warning(f"NMRShiftDB automation fetch failed: {ee}")
            else:
                # If NMRShiftDB did not raise but produced no usable data
                self._last_nmr_fetch_status = 'not_found'
            
            # If not found via NMRShiftDB and NMR predictor is available, use it
            if self.nmr_predictor:
                self.logger.info(f"Predicting NMR for generated SMILES: {generated_smiles}")
                
                # Get NMR prediction
                c_nmr_records, h_nmr_records = self.nmr_predictor.get_nmr_prediction(
                    generated_smiles, fallback_to_llm=False  # Do NOT use LLM fallback per user request
                )
                
                similarities = []
                
                # Calculate H-NMR similarity if available
                if target_h_nmr and h_nmr_records:
                    predicted_h_nmr = self.nmr_predictor.format_nmr_for_comparison(h_nmr_records, "H")
                    # Persist last predicted H-NMR string
                    self._last_predicted_h_nmr = predicted_h_nmr or None
                    h_sim = self.nmr_predictor.calculate_nmr_similarity(
                        target_h_nmr, predicted_h_nmr, "H", self.config.nmr_tolerance
                    )
                    similarities.append(h_sim)
                
                # Calculate C-NMR similarity if available (preferred for accuracy)
                if target_c_nmr and c_nmr_records:
                    predicted_c_nmr = self.nmr_predictor.format_nmr_for_comparison(c_nmr_records, "C")
                    # Persist last predicted C-NMR string
                    self._last_predicted_c_nmr = predicted_c_nmr or None
                    c_sim = self.nmr_predictor.calculate_nmr_similarity(
                        target_c_nmr, predicted_c_nmr, "C", self.config.nmr_tolerance
                    )
                    similarities.append(c_sim)
                
                if similarities:
                    self._last_nmr_fetch_status = 'web'
                    self._last_similarity_method = 'nmr_predictor'
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
                else:
                    # No NMR records available from web (and LLM fallback disabled)
                    self._last_nmr_fetch_status = 'not_found'
            
            # No non-NMR fallback; if NMR not available, treat as not found
            if getattr(self, '_last_nmr_fetch_status', None) not in ('database', 'web'):
                self._last_nmr_fetch_status = 'not_found'
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
        batch_dir.mkdir(parents=True, exist_ok=True)
        
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
