"""
Data loading utilities for X-ray transient classification.

Handles loading transient observations and formatting data for LLM prompts.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory path."""
    return get_project_root() / "data"


def load_transient(path: str) -> Dict[str, Any]:
    """
    Load a single transient observation from JSON file.
    
    Args:
        path: Path to the transient JSON file
        
    Returns:
        Dictionary containing observation data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    with open(path, 'r') as f:
        return json.load(f)


def list_transients(data_dir: Optional[str] = None) -> List[str]:
    """
    List available transient observation files.
    
    Args:
        data_dir: Optional data directory path. If None, uses default.
        
    Returns:
        List of transient JSON file paths
    """
    if data_dir is None:
        data_dir = get_data_dir() / "transients"
    else:
        data_dir = Path(data_dir)
    
    if not data_dir.exists():
        return []
    
    return sorted([str(f) for f in data_dir.glob("*.json")])


def validate_transient(observation: Dict[str, Any]) -> List[str]:
    """
    Validate that a transient observation has required fields.
    
    Args:
        observation: Observation dictionary to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    required_top_level = ['transient_id', 'detection', 'temporal', 'spectral', 'energetics']
    for field in required_top_level:
        if field not in observation:
            errors.append(f"Missing required field: {field}")
    
    if 'detection' in observation:
        detection = observation['detection']
        if 'band_keV' not in detection:
            errors.append("Missing detection.band_keV")
        if 'instrument' not in detection:
            errors.append("Missing detection.instrument")
    
    if 'energetics' in observation:
        energetics = observation['energetics']
        if 'peak_flux_erg_s_cm2' not in energetics:
            errors.append("Missing energetics.peak_flux_erg_s_cm2")
    
    if 'spectral' in observation:
        spectral = observation['spectral']
        if 'fits' not in spectral:
            errors.append("Missing spectral.fits")
    
    return errors


def validate_data_files(data_dir: Optional[str] = None) -> bool:
    """
    Validate that required data files exist.
    
    Args:
        data_dir: Optional data directory path
        
    Returns:
        True if all required files exist
    """
    if data_dir is None:
        data_dir = get_data_dir()
    else:
        data_dir = Path(data_dir)
    
    transients_dir = data_dir / "transients"
    
    if not transients_dir.exists() or not list(transients_dir.glob("*.json")):
        return False
    
    return True


def format_observation_for_prompt(
    observation: Dict[str, Any],
    include_ground_truth: bool = False
) -> str:
    """
    Format observation data into a human-readable string for LLM prompts.
    
    Args:
        observation: Observation dictionary
        include_ground_truth: Whether to include ground truth (for debugging)
        
    Returns:
        Formatted string representation
    """
    lines = []
    
    lines.append(f"=== X-Ray Transient Observation: {observation.get('transient_id', 'Unknown')} ===")
    lines.append("")
    
    if 'metadata' in observation:
        meta = observation['metadata']
        lines.append(f"Description: {meta.get('description', 'N/A')}")
        lines.append(f"Discovery Instrument: {meta.get('discovery_instrument', 'N/A')}")
        lines.append("")
    
    if 'detection' in observation:
        det = observation['detection']
        lines.append("--- Detection Parameters ---")
        lines.append(f"Instrument: {det.get('instrument', 'N/A')}")
        band = det.get('band_keV', [])
        if band:
            lines.append(f"Energy Band: {band[0]}-{band[1]} keV")
        lines.append(f"Exposure: {det.get('exposure_ks', 'N/A')} ks")
        lines.append(f"Net Counts: {det.get('net_counts', 'N/A')}")
        lines.append(f"SNR: {det.get('snr', 'N/A')}")
        lines.append(f"Localization: {det.get('localization_arcsec', 'N/A')} arcsec ({det.get('localization_confidence', 'N/A')*100:.0f}% confidence)")
        lines.append("")
    
    if 'persistent_constraints' in observation:
        pers = observation['persistent_constraints']
        lines.append("--- Persistent Emission Constraints ---")
        lines.append(f"Pre-flare limit: {pers.get('pre_flare_limit_erg_s_cm2', 'N/A'):.1e} erg/s/cm²")
        if pers.get('pre_flare_limit_note'):
            lines.append(f"  Note: {pers['pre_flare_limit_note']}")
        lines.append(f"Stacked limit: {pers.get('stacked_limit_erg_s_cm2', 'N/A'):.1e} erg/s/cm² ({pers.get('stacked_exposure_ks', 'N/A')} ks)")
        lines.append(f"Prior detections: {pers.get('prior_detections', 'N/A')}")
        lines.append("")
    
    if 'temporal' in observation:
        temp = observation['temporal']
        lines.append("--- Temporal Properties ---")
        lines.append(f"Overall morphology: {temp.get('overall_morphology', 'N/A')}")
        
        if 'spike' in temp:
            spike = temp['spike']
            lines.append(f"Spike duration: {spike.get('duration_s', 'N/A')} s")
            lines.append(f"Rise time: {spike.get('rise_time_s', 'N/A')} s")
            lines.append(f"Peak count rate: {spike.get('peak_count_rate_cts_s', 'N/A')} cts/s")
            if spike.get('peak_count_rate_note'):
                lines.append(f"  Note: {spike['peak_count_rate_note']}")
        
        if 'tail' in temp:
            tail = temp['tail']
            lines.append(f"Tail duration: {tail.get('duration_s', 'N/A')} s")
            lines.append(f"Tail morphology: {tail.get('morphology', 'N/A')}")
        
        lines.append(f"Precursor detected: {temp.get('precursor_detected', 'N/A')}")
        lines.append("")
    
    if 'spectral' in observation:
        spec = observation['spectral']
        lines.append("--- Spectral Properties ---")
        lines.append(f"Spectral evolution: {spec.get('evolution', 'N/A')}")
        
        if 'hardness_ratio' in spec:
            hr = spec['hardness_ratio']
            lines.append(f"Hardness ratio (spike): {hr.get('spike', 'N/A')}")
            if hr.get('spike_note'):
                lines.append(f"  Note: {hr['spike_note']}")
            lines.append(f"Hardness ratio (tail): {hr.get('tail_range', 'N/A')}")
        
        if 'fits' in spec:
            fits = spec['fits']
            lines.append(f"Fitting method: {fits.get('method', 'N/A')}")
            
            if 'power_law' in fits:
                pl = fits['power_law']
                lines.append(f"Power-law fit: Γ = {pl.get('photon_index', 'N/A')} ± {pl.get('photon_index_err', 'N/A')}")
            
            if 'blackbody' in fits:
                bb = fits['blackbody']
                lines.append(f"Blackbody fit: kT = {bb.get('kT_keV', 'N/A')} ± {bb.get('kT_keV_err', 'N/A')} keV")
            
            if fits.get('fit_comparison'):
                lines.append(f"Fit comparison: {fits['fit_comparison']}")
        lines.append("")
    
    if 'energetics' in observation:
        ener = observation['energetics']
        lines.append("--- Energetics ---")
        
        if 'peak_flux_powerlaw_erg_s_cm2' in ener:
            lines.append(f"Peak flux (power law model): {ener['peak_flux_powerlaw_erg_s_cm2']:.1e} erg/s/cm²")
        if 'peak_flux_blackbody_erg_s_cm2' in ener:
            lines.append(f"Peak flux (blackbody model): {ener['peak_flux_blackbody_erg_s_cm2']:.1e} erg/s/cm²")
        
        flux = ener.get('peak_flux_erg_s_cm2', [])
        if flux:
            lines.append(f"Peak flux: ({flux[0]:.1e} - {flux[1]:.1e}) erg/s/cm²")
        
        band = ener.get('peak_flux_band_keV', [])
        if band:
            lines.append(f"  Band: {band[0]}-{band[1]} keV")
        
        if 'fluence_powerlaw_erg_cm2' in ener:
            lines.append(f"Fluence (power law model): {ener['fluence_powerlaw_erg_cm2']:.1e} erg/cm²")
        if 'fluence_blackbody_erg_cm2' in ener:
            lines.append(f"Fluence (blackbody model): {ener['fluence_blackbody_erg_cm2']:.1e} erg/cm²")
        
        fluence = ener.get('fluence_erg_cm2', [])
        if fluence:
            lines.append(f"Fluence: ({fluence[0]:.1e} - {fluence[1]:.1e}) erg/cm²")
        
        if ener.get('energetics_note'):
            lines.append(f"Note: {ener['energetics_note']}")
        lines.append("")
    
    if 'counterparts' in observation:
        cp = observation['counterparts']
        lines.append("--- Multi-wavelength Counterparts ---")
        
        if 'gamma_ray' in cp and cp['gamma_ray']:
            gr = cp['gamma_ray']
            lines.append(f"Gamma-ray detected: {gr.get('detected', 'N/A')}")
            if gr.get('note'):
                lines.append(f"  Note: {gr['note']}")
        
        if 'optical' in cp and cp['optical']:
            opt = cp['optical']
            lines.append(f"Optical counterpart in error circle: {opt.get('counterpart_in_error_circle', 'N/A')}")
            if 'limits_5sigma_mag' in opt:
                lim = opt['limits_5sigma_mag']
                lines.append(f"  5σ limits: g={lim.get('g', 'N/A')}, u={lim.get('u', 'N/A')}, i={lim.get('i', 'N/A')} mag")
            if opt.get('extended_galaxy_detected') is not None:
                lines.append(f"Extended galaxy detected: {opt['extended_galaxy_detected']}")
        
        if 'radio' in cp:
            lines.append(f"Radio: {cp['radio'] if cp['radio'] else 'No data'}")
        lines.append("")
    
    if 'context' in observation:
        ctx = observation['context']
        lines.append("--- Context ---")
        lines.append(f"Field: {ctx.get('field', 'N/A')}")
        
        if 'possible_associations' in ctx:
            lines.append("Possible associations:")
            for assoc_name, assoc_data in ctx['possible_associations'].items():
                if 'distance_kpc' in assoc_data:
                    lines.append(f"  - {assoc_name}: {assoc_data['distance_kpc']} kpc")
                elif 'distance_kpc_range' in assoc_data:
                    r = assoc_data['distance_kpc_range']
                    lines.append(f"  - {assoc_name}: {r[0]}-{r[1]} kpc")
                elif 'distance_kpc_min' in assoc_data:
                    lines.append(f"  - {assoc_name}: >{assoc_data['distance_kpc_min']} kpc")
                if assoc_data.get('note'):
                    lines.append(f"      ({assoc_data['note']})")
        
        if ctx.get('stellar_population_note'):
            lines.append(f"Stellar population: {ctx['stellar_population_note']}")
        lines.append("")
    
    if include_ground_truth and 'ground_truth' in observation:
        gt = observation['ground_truth']
        lines.append("--- Ground Truth (DEBUG) ---")
        lines.append(f"Classification: {gt.get('classification', 'N/A')}")
        lines.append(f"Confidence: {gt.get('confidence', 'N/A')}")
        lines.append(f"Alternatives: {gt.get('alternatives', [])}")
        if gt.get('reasoning_notes'):
            lines.append("Reasoning:")
            for note in gt['reasoning_notes']:
                lines.append(f"  - {note}")
    
    return "\n".join(lines)


def get_ground_truth(observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract ground truth from observation.
    
    Args:
        observation: Observation dictionary
        
    Returns:
        Ground truth dictionary or empty dict if not present
    """
    return observation.get('ground_truth', {})


