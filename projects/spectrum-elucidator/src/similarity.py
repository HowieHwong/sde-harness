import re
import xml.etree.ElementTree as ET
from urllib.parse import quote
from typing import List, Tuple, Dict, Optional

# Optional RDKit for fingerprints
try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    from rdkit import DataStructs  # type: ignore
    RDKIT_AVAILABLE = True
except Exception:
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    DataStructs = None  # type: ignore
    RDKIT_AVAILABLE = False

# Common solvent residual 13C peaks (approximate, ppm)
COMMON_SOLVENTS = {
    "CDCl3": [77.16],              # (triplet center; sidebands often at ~76.8/77.5)
    "DMSO-d6": [39.52],
    "acetone-d6": [29.84],
    "methanol-d4": [49.00],
    "acetonitrile-d3": [118.69, 1.39],  # includes 13C of CD3CN and potential impurity
    "THF-d8": [67.21, 25.31],
    "toluene-d8": [137.9, 128.9, 125.9, 20.4],
}

# ---- NMRShiftDB automation client (CML-based) ----

def _fetch_cml_from_nmrshiftdb(smiles: str, nucleus: str = "13C", timeout: int = 30) -> str:
    """
    Call NMRShiftDB 'search or predict' automation endpoint, return CML (XML) text.
    nucleus: "13C", "1H", or others ("15N","31P","19F","11B","29Si","17O",...)
    """
    primary = "https://www.nmrshiftdb.org/NmrshiftdbServlet/nmrshiftdbaction/searchorpredict"
    fallback = "https://nmrshiftdb.nmr.uni-koeln.de/NmrshiftdbServlet/nmrshiftdbaction/searchorpredict"
    path = f"/smiles/{quote(smiles)}/spectrumtype/{nucleus}"
    try:
        import requests  # optional dependency
    except Exception:
        return ""

    # Try primary host with verified TLS
    try:
        r = requests.get(primary + path, timeout=timeout)
        if r.ok and r.text.strip():
            return r.text
    except Exception:
        pass

    # Try fallback host; its TLS chain can be incomplete, allow verify=False
    try:
        try:
            requests.packages.urllib3.disable_warnings()  # type: ignore
        except Exception:
            pass
        r = requests.get(fallback + path, timeout=timeout, verify=False)
        if r.ok and r.text.strip():
            return r.text
    except Exception:
        pass

    return ""

def _text(el: Optional[ET.Element]) -> Optional[str]:
    return None if el is None else (el.text or "").strip() or None

def _float_safe(x: Optional[str]) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def _find_first_text(node: ET.Element, candidates: List[str]) -> Optional[str]:
    """Try multiple tag names under any namespace."""
    for name in candidates:
        el = node.find(f".//{{*}}{name}")
        if el is not None and (el.text or "").strip():
            return el.text.strip()
    return None

def parse_peaks_from_cml(cml_text: str) -> List[Dict]:
    """
    Return [{'ppm': float, 'intensity': Optional[float], 'mult': Optional[str], 'J': Optional[str], 'raw': dict}, ...]
    Notes:
      - For predicted spectra, multiplicity / J may be missing or reside in various vendor-specific tags.
      - We attempt common fields: xValue / yValue, multiplicity, peakMultiplicity, comment, label, etc.
    """
    if not cml_text or not cml_text.strip():
        return []
    root = ET.fromstring(cml_text)
    peaks: List[Dict] = []
    for peak in root.findall(".//{*}peak"):
        # First try child-text fields
        ppm = _float_safe(_find_first_text(peak, ["xValue", "xvalue", "x", "xCoord"]))
        intensity = _float_safe(_find_first_text(peak, ["yValue", "yvalue", "y", "yCoord", "peakHeight"]))
        mult = _find_first_text(peak, [
            "multiplicity", "peakMultiplicity", "mult", "pattern", "spinmult"
        ])
        J = _find_first_text(peak, [
            "J", "jCoupling", "coupling", "JValue", "Jvalue"
        ])
        comment = _find_first_text(peak, ["comment", "label", "name"])
        # If not found, fall back to attributes (NMRShiftDB often encodes peaks as attributes)
        if ppm is None:
            ppm = _float_safe(peak.attrib.get('xValue') or peak.attrib.get('xvalue') or peak.attrib.get('x') or peak.attrib.get('xCoord'))
        if intensity is None:
            intensity = _float_safe(peak.attrib.get('yValue') or peak.attrib.get('yvalue') or peak.attrib.get('y') or peak.attrib.get('yCoord') or peak.attrib.get('peakHeight'))
        if not mult:
            mult = peak.attrib.get('peakMultiplicity') or peak.attrib.get('multiplicity') or peak.attrib.get('mult') or None
        if not J:
            J = peak.attrib.get('J') or peak.attrib.get('jCoupling') or peak.attrib.get('coupling') or None
        if not comment:
            comment = peak.attrib.get('label') or peak.attrib.get('name') or None
        if not mult and comment:
            m = re.search(r"\b(s|d|t|q|quin|sext|sept|m|dd|dt|dq|td|tt|dq|ddd)\b", comment, re.I)
            if m:
                mult = m.group(1)
            mJ = re.search(r"J\s*=\s*([\d\.]+)\s*Hz", comment, re.I)
            if mJ and not J:
                J = mJ.group(1)
        peaks.append({
            "ppm": ppm,
            "intensity": intensity,
            "mult": mult,
            "J": J,
            "raw": {**{child.tag.split('}')[-1]: (child.text or "").strip() for child in peak}, **{k: v for k, v in peak.attrib.items()}},
        })
    peaks = [p for p in peaks if p["ppm"] is not None]
    return sorted(peaks, key=lambda d: d["ppm"], reverse=True)

def get_nmr_peaks(smiles: str, nucleus: str = "13C") -> List[Dict]:
    """
    Input: SMILES; Output: list of peaks for the given nucleus.
      Each peak: {'ppm': float, 'intensity': Optional[float], 'mult': Optional[str], 'J': Optional[str], 'raw': dict}
    """
    cml = _fetch_cml_from_nmrshiftdb(smiles, nucleus=nucleus)
    return parse_peaks_from_cml(cml)

def get_1H_13C_peaks(smiles: str) -> Dict[str, List[Dict]]:
    """Convenience wrapper: return both 1H and 13C peak lists."""
    return {
        "1H": get_nmr_peaks(smiles, "1H"),
        "13C": get_nmr_peaks(smiles, "13C"),
    }

# ---- Tanimoto similarity (fallback) ----

def tanimoto_smiles(smi_a: str, smi_b: str, radius: int = 2, n_bits: int = 2048) -> Optional[float]:
    """Compute Tanimoto similarity between two SMILES using Morgan fingerprints.
    Returns None if RDKit is unavailable or parsing fails.
    """
    if not RDKIT_AVAILABLE:
        return None
    try:
        ma = Chem.MolFromSmiles(smi_a)
        mb = Chem.MolFromSmiles(smi_b)
        if ma is None or mb is None:
            return None
        fa = AllChem.GetMorganFingerprintAsBitVect(ma, radius, nBits=n_bits)
        fb = AllChem.GetMorganFingerprintAsBitVect(mb, radius, nBits=n_bits)
        return float(DataStructs.TanimotoSimilarity(fa, fb))
    except Exception:
        return None

def parse_c_nmr(s: str, keep_ranges_as_center: bool = True) -> List[float]:
    """
    Extract 13C shifts (ppm) from a free-form NMR string.
    Examples it accepts:
      '13C NMR (100 MHz, CDCl3) δ 170.2, 151.3, 128.1–128.5, 77.0'
      'δ170.2 151.3 128.1-128.5'
    Returns a list of floats (ppm).
    """
    # Normalize dashes and decimals
    s = s.replace("–", "-").replace("—", "-").replace("‒", "-")
    # Grab range tokens first: e.g., 128.1-128.5 (allow no leading zero)
    range_pat = re.compile(r'(?<!\d)(\d{1,3}(?:\.\d+)?)[ \t]*-[ \t]*(\d{1,3}(?:\.\d+)?)(?!\d)')
    ranges = [(float(a), float(b)) for a, b in range_pat.findall(s)]
    # Remove ranges so their endpoints aren’t double-counted by float matcher
    s_wo_ranges = range_pat.sub(" ", s)

    # Grab standalone floats up to ~3 digits before decimal (typical 0–220 ppm)
    float_pat = re.compile(r'(?<!\d)(\d{1,3}(?:\.\d+)?)(?!\d)')
    singles = [float(x) for x in float_pat.findall(s_wo_ranges)]

    peaks: List[float] = []
    if keep_ranges_as_center:
        for a, b in ranges:
            peaks.append((a + b) / 2.0)
    else:
        # If you’d rather keep both endpoints, extend instead:
        for a, b in ranges:
            peaks.extend([a, b])

    peaks.extend(singles)
    # Valid 13C scale is roughly 0–220 ppm; filter out outliers just in case
    peaks = [p for p in peaks if 0.0 <= p <= 230.0]
    # Sort & de-duplicate very near duplicates (e.g., repeated in text)
    peaks = dedup_close(peaks, tol=0.02)
    return peaks

def dedup_close(vals: List[float], tol: float = 0.02) -> List[float]:
    vals = sorted(vals)
    out = []
    for v in vals:
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out

def remove_solvent_peaks(peaks: List[float],
                         solvents: Optional[List[str]] = None,
                         tol: float = 0.10) -> List[float]:
    """Remove peaks that match common solvent peaks within ±tol ppm."""
    if not solvents:
        return peaks
    solvent_list = []
    for name in solvents:
        solvent_list.extend(COMMON_SOLVENTS.get(name, []))
    cleaned = []
    for p in peaks:
        if any(abs(p - s) <= tol for s in solvent_list):
            continue
        cleaned.append(p)
    return cleaned

def match_peaks(a: List[float], b: List[float], tol: float = 0.20) -> Dict[str, object]:
    """
    One-to-one nearest matching: each peak in A can match at most one in B within tol.
    Greedy strategy is sufficient for typical NMR lists.
    Returns:
      - matches: list of tuples (a_ppm, b_ppm, diff)
      - unmatched_a, unmatched_b
      - metrics: dict with precision/recall/F1 and mean_abs_diff (matched)
    """
    A = sorted(a)
    B = sorted(b)
    used = [False] * len(B)
    matches: List[Tuple[float, float, float]] = []

    for x in A:
        # find closest y in B within tol that is unused
        best_j = None
        best_diff = None
        for j, y in enumerate(B):
            if used[j]:
                continue
            d = abs(x - y)
            if d <= tol and (best_diff is None or d < best_diff):
                best_diff = d
                best_j = j
        if best_j is not None:
            used[best_j] = True
            matches.append((x, B[best_j], best_diff))

    matched_a = {m[0] for m in matches}
    matched_b = {m[1] for m in matches}
    unmatched_a = [x for x in A if x not in matched_a]
    unmatched_b = [y for y in B if y not in matched_b]

    tp = len(matches)
    fp = len(unmatched_b)  # peaks in B with no match (if B is 'observed')
    fn = len(unmatched_a)  # peaks in A with no match (if A is 'expected')
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_abs_diff = sum(d for _, _, d in matches) / tp if tp else None

    return {
        "matches": matches,
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
        "metrics": {
            "tolerance_ppm": tol,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_abs_diff": mean_abs_diff
        }
    }

# ---- Convenience wrapper you can call directly ----

def compare_c_nmr_strings(
    s1: str,
    s2: str,
    tol_ppm: float = 0.20,
    ignore_solvents: Optional[List[str]] = None,
    keep_ranges_as_center: bool = True
) -> Dict[str, object]:
    """
    Parse two 13C NMR strings and compare their peak lists.
    """
    a = parse_c_nmr(s1, keep_ranges_as_center=keep_ranges_as_center)
    b = parse_c_nmr(s2, keep_ranges_as_center=keep_ranges_as_center)
    if ignore_solvents:
        a = remove_solvent_peaks(a, ignore_solvents)
        b = remove_solvent_peaks(b, ignore_solvents)
    return {
        "peaks_a": a,
        "peaks_b": b,
        **match_peaks(a, b, tol=tol_ppm)
    }

# ---- Pretty-printer ----

def format_report(result: Dict[str, object]) -> str:
    lines = []
    metrics = result["metrics"]
    lines.append(f"Tolerance: ±{metrics['tolerance_ppm']:.2f} ppm")
    lines.append(f"Matches: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    lines.append(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
    mad = metrics["mean_abs_diff"]
    if mad is not None:
        lines.append(f"Mean abs. diff (matched): {mad:.3f} ppm")
    lines.append("\nMatched peaks (A ↔ B | Δ):")
    for a_ppm, b_ppm, d in result["matches"]:
        lines.append(f"  {a_ppm:7.2f} ↔ {b_ppm:7.2f} | {d:5.3f}")
    if result["unmatched_a"]:
        lines.append("\nUnmatched A:")
        lines.append("  " + ", ".join(f"{x:.2f}" for x in result["unmatched_a"]))
    if result["unmatched_b"]:
        lines.append("\nUnmatched B:")
        lines.append("  " + ", ".join(f"{y:.2f}" for y in result["unmatched_b"]))
    return "\n".join(lines)

def __main__():
    s1 = "13C NMR (100 MHz, CDCl3) δ 170.2, 151.3, 135.0, 128.1–128.5, 77.16, 55.4"
    s2 = "δ 170.21, 151.1, 134.9, 128.3, 55.38, 29.8; solvent CDCl3"

    res = compare_c_nmr_strings(
        s1, s2,
        tol_ppm=0.20,
        ignore_solvents=["CDCl3"],      # drop the 77.16 ppm solvent line
        keep_ranges_as_center=True      # treat 128.1–128.5 as 128.3
    )
    print(format_report(res))


if __name__ == "__main__":
    __main__()
