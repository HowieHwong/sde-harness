# Spectrum Elucidator Toolkit

A comprehensive toolkit for iterative molecular structure elucidation from NMR spectra using Large Language Models (LLMs). This toolkit implements an iterative approach where the LLM generates molecular structures, evaluates their NMR similarity to target spectra, and refines the structures based on feedback.

## 🎯 Overview

The Spectrum Elucidator Toolkit provides:

- **Iterative Elucidation**: LLM generates molecular structures and refines them based on NMR similarity feedback
- **Advanced NMR Prediction**: Web scraping from NMRDB + LLM fallback for accurate similarity calculation
- **Enhanced Similarity Scoring**: Uses both H-NMR and C-NMR with configurable tolerance and preferences
- **Comprehensive Visualization**: Tracks progression and generates detailed reports
- **Batch Processing**: Handle multiple molecules efficiently
- **Configurable Parameters**: Adjustable iteration limits, similarity thresholds, and NMR prediction settings
- **Lightweight Mode**: Core functionality without heavy dependencies for compatibility

## 🏗️ Project Structure

```
spectrum-elucidator/
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── data_utils.py            # Data loading, parsing, advanced NMR similarity
│   ├── llm_interface.py         # LLM communication, SMILES extraction/validation
│   ├── elucidation_engine.py    # Iterative loop, scoring, persistence
│   ├── visualization.py         # Plots and reports (auto-creates parent dirs)
│   ├── nmr_predictor.py         # Optional NMR prediction (web + LLM fallback)
│   └── similarity.py            # 13C matching (F1) + utilities
├── data/
│   └── updated_table.csv        # Dataset with SMILES, H_NMR, C_NMR
├── examples/
│   └── (see example_* scripts in project root)
├── tests/
│   └── run_all.py               # Consolidated test runner
├── example_single_elucidation.py
├── example_batch_elucidation.py
├── example_enhanced_elucidation.py
├── example_lightweight.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Install

- Lightweight (no heavy deps):
```bash
pip install pandas numpy matplotlib seaborn openai pathlib2 beautifulsoup4 urllib3
```

- Full (optional features):
```bash
pip install -r requirements.txt
# Optional extras
# pip install selenium webdriver-manager rdkit-pypi
```

### Configure

```bash
export OPENAI_API_KEY="your_api_key_here"
```
Optionally edit `config.json` (see `config_template.json`).

### Run

- Lightweight example:
```bash
python example_lightweight.py
```

- Single/batch/enhanced:
```bash
python example_single_elucidation.py
python example_batch_elucidation.py
python example_enhanced_elucidation.py
```

- Tests:
```bash
python tests/run_all.py
```

## ⚙️ NMR Source Selection (DB vs Web vs LLM)

Order of operations when a new SMILES is generated:

1. **Validate/normalize SMILES** (regex and optional RDKit).
2. **Database lookup**: if the SMILES exists in `data/updated_table.csv`, use its stored `H_NMR` and/or `C_NMR`.
3. **Prediction (optional)** if not found in DB and enabled:
   - Web (NMRDB) if `nmr_predictor.use_web_scraping=True`.
   - LLM fallback if `nmr_predictor.use_llm_fallback=True`.
4. **Similarity computation**:
   - 13C: `src/similarity.py` (tolerant peak matching + F1).
   - 1H: advanced parser/matcher in `data_utils.py` (shift, integration, multiplicity).
   - Combine (average or weighted; `elucidation.prefer_c_nmr=True` to weight 13C higher).

User controls:
- `elucidation.use_nmr_predictor`: False = DB only; True = allow prediction.
- `nmr_predictor.use_web_scraping`: True/False (try NMRDB).
- `nmr_predictor.use_llm_fallback`: True/False (ask LLM if web fails).
- If you want to always prefer predicted NMR over DB, ask us to add `prefer_predicted_over_db` or `nmr_source_priority`.

## 🔁 Iterative Workflow (after SMILES is generated)

- Extract and validate SMILES from LLM response; fallback heuristics if needed.
- Choose NMR source (DB → Web → LLM) per config.
- Normalize/format spectra; optional solvent filtering for 13C.
- Compute similarity (13C + 1H) and combine scores.
- Record iteration (`ElucidationStep`) with metadata (source used).
- Stop if similarity ≥ threshold; otherwise append history and refine prompt for next iteration.

## 📊 Visualization & Outputs

- Auto-creates `results/` if needed; saves:
  - Similarity progression and SMILES evolution plots
  - Text summary report per molecule
  - JSON per-iteration and final results in `elucidation_results/`

## 🧪 Testing

- Consolidated tests:
```bash
python tests/run_all.py
```
Runs available tests (parsing, advanced similarity, LLM interaction where possible). No heavy deps required for most tests.

## 🔍 Troubleshooting

- Missing directory errors: handled; parent dirs are created automatically.
- Invalid SMILES from LLM: fallback extraction; iteration recorded with similarity=0 if unresolved.
- NumPy/PyTorch conflicts: use lightweight mode or optional deps only as needed.
- Web/LLM rate limits: increase `delay_between_iterations` or predictor timeouts.

## 📌 Notes on Similarity Implementations

- 13C: tolerant one-to-one matching (F1, mean |Δ|), with optional solvent peak removal.
- 1H: advanced parsing of shift, integration, multiplicity; normalized scoring across columns.
- Tolerances are configurable; defaults picked conservatively (e.g., 0.05–0.20 ppm).

## 🗺️ Roadmap / Extensibility

- Optional `nmr_source_priority` (e.g., [db, web, llm] or [llm, web, db]).
- Cache predicted NMR per SMILES to avoid repeated requests.
- Add RDKit-based structure sanity checks and depiction.

## 📚 References

- OpenAI API docs (`https://platform.openai.com/docs`)
- SMILES (`https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system`)
- NMR (`https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance_spectroscopy`)
- NMRDB (`https://www.nmrdb.org/`)
- RDKit (`https://www.rdkit.org/docs/`)
