"""
NMR prediction helpers for the Spectrum Elucidator Toolkit.

This module provides a lightweight, dependency-minimal API to fetch/predict
NMR spectra for a SMILES using the NMRShiftDB automation (CML) endpoints
exposed via utilities in `src/similarity.py`. It does not use Selenium/NMRDB
web-scraping or an LLM fallback and is safe to import in restricted runtimes.
"""

import time
import re
from typing import List, Tuple, Optional, Dict, Any
import logging
import pandas as pd

from .similarity import (
    compare_c_nmr_strings,
    parse_c_nmr,
    get_nmr_peaks,
    parse_peaks_from_cml,
    get_1H_13C_peaks,
)
from .data_utils import NMRProcessor

# Optional RDKit for molecular formula (best-effort)
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except Exception:
    Chem = None  # type: ignore
    RDKIT_AVAILABLE = False


class NMRPredictor:
    """Predict NMR spectra for SMILES using NMRShiftDB automation endpoints."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        headless: bool = True,
        timeout: int = 10,
        use_web_scraping: bool = True,
    ):
        """Initialize the predictor (keeps legacy signature for compatibility)."""
        self.openai_api_key = openai_api_key
        self.headless = headless
        self.timeout = timeout
        # This flag is a no-op now; kept for compatibility with examples
        self.use_web_scraping = True
        self.logger = logging.getLogger(__name__)

    # Thin wrappers to reuse module-level utilities and avoid duplication
    def collect_nmr_records(self, text: str) -> List[str]:
        return collect_nmr_records(text)

    def get_molecular_formula(self, smiles: str) -> str:
        return get_molecular_formula(smiles)

    # High-level API used by the engine (returns C first for historical reasons)
    def get_nmr_prediction(self, smiles: str, fallback_to_llm: bool = False) -> Tuple[List[Any], List[Any]]:
        """Return (C_records, H_records) for the SMILES using NMRShiftDB automation.

        Records are raw peak dicts from NMRShiftDB (each has at minimum 'ppm'),
        suitable for downstream `format_nmr_for_comparison`.
        """
        return self.get_nmr_from_web(smiles)

    def get_nmr_from_web(self, smiles: str) -> Tuple[List[Any], List[Any]]:
        """Return (C_records, H_records) for the SMILES via NMRShiftDB automation."""
        try:
            peaks = get_1H_13C_peaks(smiles)
        except Exception as e:
            self.logger.warning(f"NMRShiftDB fetch failed for {smiles}: {e}")
            return [], []
        c = peaks.get("13C", []) or []
        h = peaks.get("1H", []) or []
        return c, h

    def format_nmr_for_comparison(self, records: List[Any], nucleus: str) -> str:
        """Format peak records into a compact string for similarity functions.

        Accepts:
          - list of dicts with 'ppm' keys (from NMRShiftDB helpers)
          - list of numbers/strings that can be parsed as floats
          - already-formatted strings (returned unchanged)
        Output: a simple string like "δ 170.2, 151.3, ..." suitable for parsing.
        """
        if not records:
            return ""
        if isinstance(records, str):
            return records
        # If it looks like a list of dicts (NMRShiftDB), extract ppm values
        vals: List[float] = []
        try:
            if isinstance(records, list) and records and isinstance(records[0], dict):
                vals = [float(p["ppm"]) for p in records if p.get("ppm") is not None]
            else:
                # Parse numbers from mixed list
                vals = [float(x) for x in records]
        except Exception:
            # Last resort: join as-is
            return "δ " + ", ".join(str(x) for x in records)
        if not vals:
            return ""
        vals = sorted(vals, reverse=(nucleus.upper() == "C"))
        return "δ " + ", ".join(f"{v:.2f}" for v in vals)

    def calculate_nmr_similarity(self, nmr1: str, nmr2: str, nucleus: str, tolerance: float = 0.20) -> float:
        """Compute similarity between two formatted NMR strings.

        - For 13C, use tolerant peak matching (F1) via `compare_c_nmr_strings`.
        - For 1H, use the advanced parser in `NMRProcessor`.
        """
        nuc = nucleus.upper()
        if nuc == "C":
            try:
                res = compare_c_nmr_strings(nmr1, nmr2, tol_ppm=tolerance, keep_ranges_as_center=True)
                return float(res["metrics"]["f1"])  # type: ignore
            except Exception:
                return 0.0
        # Default to 1H logic
        try:
            return float(NMRProcessor.calculate_nmr_similarity_advanced(nmr1, nmr2, "H", tolerance))
        except Exception:
            return 0.0

# --- Module-level convenience helpers matching requested API ---

def collect_nmr_records(text: str) -> List[str]:
    """Extract NMR records from plain text, matching 'NMR: ...' snippets.
    Mirrors the regex used by NMRPredictor.collect_nmr_records.
    """
    nmr_pattern = r'NMR:\s*(.*?)(?=\.\s| loading|\n|$)'
    recs = re.findall(nmr_pattern, text, re.DOTALL)
    return [r.strip() for r in recs]

def get_molecular_formula(smiles: str) -> str:
    """Compute a molecular formula via RDKit if available."""
    if not RDKIT_AVAILABLE:
        return "RDKit not available"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        from rdkit.Chem import rdMolDescriptors
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return "Error"

def get_NMR(smiles: str) -> Tuple[List[Any], List[Any]]:
    """Return (C_records, H_records) using the NMRShiftDB path (compat)."""
    predictor = NMRPredictor()
    return predictor.get_nmr_from_web(smiles)

# --- Lightweight NMRShiftDB-based fetchers (no Selenium) ---

def get_c_nmr_sequence_from_smiles(
    smiles: str,
    *,
    round_decimals: int = 1,
    merge_tol: float = 0.05,
) -> Optional[str]:
    """
    Fetch predicted 13C NMR for a SMILES using NMRShiftDB (CML service),
    then format as: "NMR: 19.2 (2C, s), 21.0 (1C, s), ...".

    - Groups peaks within ±merge_tol ppm (after rounding) as one entry with count nC.
    - Uses 's' as multiplicity placeholder for 13C (broadband decoupled typical report).
    Returns None if no peaks can be fetched.
    """
    # Try robust NMRShiftDB servlet first (works behind JS predictor),
    # then fall back to the similarity.py helper.
    peaks: List[Dict[str, Any]] = []
    try:
        import requests  # local import to keep module import light
        from urllib.parse import quote
        base = "https://nmrshiftdb.nmr.uni-koeln.de/NmrshiftdbServlet/nmrshiftdbaction/searchorpredict"
        url = f"{base}/smiles/{quote(smiles)}/spectrumtype/13C"
        # SSL chain on the host is sometimes incomplete; allow unverified for read-only fetch
        try:
            # Suppress noisy insecure request warnings for this specific call
            requests.packages.urllib3.disable_warnings()  # type: ignore
        except Exception:
            pass
        r = requests.get(url, timeout=30, verify=False)
        if r.ok and r.text.strip():
            peaks = parse_peaks_from_cml(r.text)
    except Exception:
        peaks = []
    # Fallback to helper if needed
    if not peaks:
        try:
            peaks = get_nmr_peaks(smiles, nucleus="13C")  # [{'ppm': float, ...}]
        except Exception:
            peaks = []

    # If still empty, parse attribute-style CML peakList directly (xValue on <peak>)
    if not peaks:
        try:
            import xml.etree.ElementTree as ET
            from urllib.parse import quote
            base = "https://nmrshiftdb.nmr.uni-koeln.de/NmrshiftdbServlet/nmrshiftdbaction/searchorpredict"
            url = f"{base}/smiles/{quote(smiles)}/spectrumtype/13C"
            import requests
            r = requests.get(url, timeout=30, verify=False)
            if r.ok and r.text:
                root = ET.fromstring(r.text)
                tmp: List[Dict[str, Any]] = []
                for peak in root.findall('.//{*}peak'):
                    x = peak.attrib.get('xValue') or peak.attrib.get('xvalue') or peak.attrib.get('x')
                    mult = peak.attrib.get('peakMultiplicity') or peak.attrib.get('multiplicity') or peak.attrib.get('mult')
                    arefs = (peak.attrib.get('atomRefs') or '').strip()
                    count = len([t for t in arefs.split() if t]) if arefs else 1
                    try:
                        ppm = float(x) if x is not None else None
                    except Exception:
                        ppm = None
                    if ppm is not None:
                        tmp.append({'ppm': ppm, 'mult': mult, 'count': count})
                # Expand by count so downstream grouping works uniformly
                peaks = []
                for p in tmp:
                    for _ in range(max(1, int(p.get('count', 1)))):
                        peaks.append({'ppm': p['ppm'], 'mult': p.get('mult')})
        except Exception:
            peaks = []
    if not peaks:
        return None

    # Extract ppm values and sort ascending
    vals = sorted([p["ppm"] for p in peaks if isinstance(p.get("ppm"), (int, float))])
    if not vals:
        return None

    # Round to desired decimals, then cluster near-identical values
    rounded = [round(v, round_decimals) for v in vals]
    rounded.sort()

    sequences: List[Tuple[float, int]] = []  # (ppm, count)
    for v in rounded:
        if not sequences:
            sequences.append((v, 1))
        else:
            last_v, last_c = sequences[-1]
            if abs(v - last_v) <= merge_tol:
                sequences[-1] = (last_v, last_c + 1)
            else:
                sequences.append((v, 1))

    # Format: NMR: 19.2 (2C, s), 21.0 (1C, s), ...
    parts = [f"{ppm:.{round_decimals}f} ({count}C, s)" for ppm, count in sequences]
    return "NMR: " + ", ".join(parts)

def update_csv_with_nmr(
    input_csv: str,
    output_csv: str,
    smiles_column: str = 'SMILES',
    out_h_col: str = 'up_H_NMR',
    out_c_col: str = 'up_C_NMR',
    sleep_sec: float = 0.0,
) -> None:
    """Batch-fetch NMR records for each SMILES in a CSV and write to a new CSV.

    Adds/overwrites columns out_h_col and out_c_col with lists/strings of NMR records.
    """
    df = pd.read_csv(input_csv)
    predictor = NMRPredictor()

    for idx, row in df.iterrows():
        smiles = str(row.get(smiles_column, '')).strip()
        if not smiles:
            continue
        try:
            c_rec, h_rec = predictor.get_nmr_from_web(smiles)
            # Store compact strings for readability
            df.at[idx, out_h_col] = predictor.format_nmr_for_comparison(h_rec, "H")
            df.at[idx, out_c_col] = predictor.format_nmr_for_comparison(c_rec, "C")
        except Exception as e:
            # keep row but leave fields empty/NaN
            predictor.logger.warning(f"Failed NMR fetch for index {idx} SMILES {smiles}: {e}")
        if sleep_sec:
            time.sleep(sleep_sec)
    df.to_csv(output_csv, index=False)

__all__ = [
    "NMRPredictor",
    "collect_nmr_records",
    "get_molecular_formula",
    "get_c_nmr_sequence_from_smiles",
    "update_csv_with_nmr",
    "get_NMR",
]
