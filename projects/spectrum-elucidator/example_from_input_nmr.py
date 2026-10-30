#!/usr/bin/env python3
"""
Example: Iterative elucidation starting from user-provided NMR sequence(s).

Workflow:
1) Provide target H-NMR and/or 13C-NMR strings below (or via CLI/env).
2) LLM proposes a SMILES.
3) System fetches/predicts NMR for that SMILES (NMRShiftDB automation; optional
   LLM fallback if enabled), then computes similarity vs target.
4) Feedback (score + history) goes back into next LLM prompt; iterate.

Note: Requires OPENAI_API_KEY for LLM usage. Selenium is not required.
"""

import os
import sys
from pathlib import Path

# Ensure local src is importable
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import ConfigManager
from src.llm_interface import LLMInterface
from src.elucidation_engine import ElucidationEngine
from src.data_utils import MolecularDataLoader
from src.nmr_predictor import NMRPredictor


def main():
    # --- 0) Configure target NMR here ---
    # Example H-NMR: match your real target string format
    target_h_nmr = "H-NMR: 7.20 (5H, m), 2.62 (2H, q, J = 7.2 Hz), 1.20 (3H, t, J = 7.6 Hz)"
    # Example 13C-NMR
    target_c_nmr = "δ 142.3, 128.6, 127.4, 126.2, 28.7, 15.7"
    target_id = "custom_target_1"

    # --- 1) Load config and LLM ---
    cfg = ConfigManager()
    cfg.update_elucidation_config(
        use_nmr_predictor=True,     # enable NMR prediction
        max_iterations=5,           # 3–5 iterations as requested
        similarity_threshold=0.7,   # adjustable
        log_level="INFO",
    )
    nmr_cfg = cfg.get_nmr_predictor_config()

    api_key = os.getenv("OPENAI_API_KEY", cfg.get_llm_config().api_key)
    if not api_key:
        print("ERROR: OPENAI_API_KEY is required for LLM prompting.")
        return

    llm = LLMInterface(api_key=api_key, model=cfg.get_llm_config().model, max_tokens=cfg.get_llm_config().max_tokens)

    # --- 2) Build predictor (uses NMRShiftDB automation) ---
    predictor = NMRPredictor(
        openai_api_key=api_key,
        headless=True,
        timeout=nmr_cfg.web_timeout,
        use_web_scraping=nmr_cfg.use_web_scraping,
    )

    # --- 3) Run elucidation loop from provided NMR ---
    data_loader = MolecularDataLoader(cfg.get_data_config().data_path)
    engine = ElucidationEngine(
        data_loader=data_loader,
        llm_interface=llm,
        config=cfg.get_elucidation_config(),
        nmr_predictor=predictor,
    )

    result = engine.elucidate_from_nmr(
        target_molecule_id=target_id,
        target_h_nmr=target_h_nmr,
        target_c_nmr=target_c_nmr,
        max_iterations=None,  # use config
    )

    print("\n=== Elucidation (from input NMR) Summary ===")
    print(f"ID: {result.target_molecule_id}")
    print(f"Success: {result.success}")
    print(f"Final similarity: {result.final_similarity:.3f}")
    print(f"Iterations: {result.total_iterations}")
    print(f"Final SMILES: {result.final_smiles}")


if __name__ == "__main__":
    main()
