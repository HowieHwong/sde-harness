# MatLLMSearch - LLM-based Crystal Structure Discovery

LLM-based Crystal Structure Generation and Optimization for Materials Discovery, integrated with the SDE-Harness framework. MatLLMSearch uses evolution-guided large language models to generate novel crystal structures and optimize them for stability and mechanical properties.

## Original Code Repository
[https://github.com/JingruG/MatLLMSearch](https://github.com/JingruG/MatLLMSearch)

Paper: [MatLLMSearch: Crystal Structure Discovery with Evolution-Guided Large Language Models](https://arxiv.org/abs/2502.20933)

## 📦 Install

1. Get to the project folder:
   ```bash
   cd projects/matllmsearch
   ```

2. Install the SDE-Harness framework (from the repo root):
   ```bash
   cd ../..
   pip install -e .
   cd projects/matllmsearch
   ```

3. Set up the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ScienceBench_MatLLMSearch
   ```

4. Configure models and credentials in the repo-root `config/` directory.
   These files are git-ignored so your keys never get committed:
   ```bash
   cd ../..
   cp config/models.template.yaml config/models.yaml
   cp config/credentials.template.yaml config/credentials.yaml
   cd projects/matllmsearch
   ```
   Then add your API key(s). Either paste the key directly into
   `config/credentials.yaml`:
   ```yaml
   openai:
     api_key: sk-...
   ```
   Or reference an environment variable (recommended on shared servers, keeps
   the key out of any file):
   ```yaml
   openai:
     api_key: ${OPENAI_API_KEY}
   ```
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Other providers (Anthropic, DeepSeek, xAI, Gemini, AWS Bedrock, ...) are
   supported the same way — see `config/models.template.yaml` and
   `config/credentials.template.yaml`. Any model passed to `--model` must be
   defined in `config/models.yaml`. Example for Bedrock:
   ```yaml
   # config/models.yaml
   bedrock/claude-sonnet-4-5:
     provider: bedrock
     model: us.anthropic.claude-sonnet-4-5-20250929-v1:0
     credentials: bedrock

   # config/credentials.yaml
   bedrock:
     aws_access_key_id: ${AWS_ACCESS_KEY_ID}
     aws_secret_access_key: ${AWS_SECRET_ACCESS_KEY}
     aws_region_name: ${AWS_REGION_NAME}
   ```

5. Download required data files:
   ```bash
   # Phase diagram data (required for E_hull / stability calculations)
   wget -O data/2023-02-07-ppd-mp.pkl.gz https://figshare.com/ndownloader/files/48241624

   # Seed structures (optional - enables few-shot generation)
   # data/band_gap_processed_5000.csv:
   #   https://drive.google.com/file/d/14e5p3EoKzOHqw7hKy8oDsaGPK6gwhnLV/view?usp=sharing
   ```

6. (Optional) Run the smoke test (no GPU / API key required):
   ```bash
   python test.py
   ```

## 🖥️ Hardware

- **LLM generation** with **API models** (e.g. `openai/gpt-5-mini`, `deepseek/deepseek-reasoner`) runs remotely and needs **no local GPU**.
- **LLM generation** with **local Hugging Face models** (e.g. `meta-llama/Meta-Llama-3.1-70B-Instruct`) requires a **local GPU**.
- **Stability evaluation** (CHGNet/M3GNet relaxation) runs on **GPU (recommended)** or **CPU**. Use `--device cuda` or `--device cpu`.

## 🎯 Usage

### Crystal Structure Generation (CSG)
Generate novel crystal structures using evolutionary optimization:

```bash
python cli.py csg \
    --model openai/gpt-5-mini \
    --population-size 100 \
    --max-iter 10 \
    --opt-goal e_hull_distance \
    --data-path data/band_gap_processed_5000.csv \
    --save-label csg_experiment
```

### Crystal Structure Prediction (CSP)
Predict ground state structures for a target compound:

```bash
python cli.py csp \
    --compound Ag6O2 \
    --model openai/gpt-5-mini \
    --population-size 10 \
    --max-iter 5 \
    --save-label ag6o2_prediction
```

### Analysis
Evaluate generated structures and compute validity, diversity, novelty, and stability metrics.

**From a CSV file:**
```bash
python cli.py analyze \
    --input data/llama_test.csv \
    --output evaluation_results.json \
    --data-path data/band_gap_processed_5000.csv \
    --device cuda
```

**From a previous CSG run directory:**
```bash
python cli.py analyze \
    --results-path logs/csg_experiment \
    --output reevaluated_results.json \
    --data-path data/band_gap_processed_5000.csv
```

**Generate via API and evaluate:**
```bash
python cli.py analyze --generate \
    --model openai/gpt-5-mini \
    --data-path data/band_gap_processed_5000.csv \
    --max-iter 10 \
    --population-size 10 \
    --reproduction-size 5 \
    --parent-size 2 \
    --output gpt5_results.json
```

### View Help
```bash
python cli.py --help
python cli.py csg --help
```

## Common Parameters

- `--model`: Model to use (e.g. `openai/gpt-5-mini`, `deepseek/deepseek-reasoner`). Must be defined in `config/models.yaml`.
- `--population-size`: Population size for the evolutionary loop (default: 100)
- `--reproduction-size`: Number of offspring per generation (default: 5)
- `--parent-size`: Number of parent structures per group (default: 2)
- `--max-iter`: Maximum number of evolutionary iterations
- `--opt-goal`: `e_hull_distance` (stability), `bulk_modulus_relaxed`, or `multi-obj`
- `--fmt`: Structure format, `poscar` or `cif` (default: poscar)
- `--data-path`: Path to seed structures CSV (also the reference pool for novelty)
- `--device`: `cuda` or `cpu` (default: cuda)
- `--seed`: Random seed (default: 42)

## 🏗️ Project Structure

```
projects/matllmsearch/
├── cli.py                          # Command line entry point
├── src/
│   ├── modes/
│   │   ├── csg.py                  # Crystal Structure Generation mode
│   │   ├── csp.py                  # Crystal Structure Prediction mode
│   │   └── analyze.py              # Analysis mode
│   ├── utils/
│   │   ├── structure_generator.py  # LLM-based structure generator
│   │   ├── stability_calculator.py # CHGNet/M3GNet stability evaluation
│   │   ├── materials_oracle.py     # Materials property oracle (SDE-Harness Oracle)
│   │   ├── evaluate_structures.py  # Validity / diversity / novelty / SUN metrics
│   │   ├── e_hull_calculator.py    # Energy-above-hull calculation
│   │   ├── data_loader.py          # Data loading utilities
│   │   └── config.py               # Prompt templates
├── data/                           # Data files (downloaded separately)
├── environment.yml                 # Conda environment
├── test.py                         # Smoke test (no GPU / API key)
├── requirements.txt
└── README.md                       # This document

# Model / credential configuration lives in the repo-root config/ directory:
#   ../../config/models.yaml
#   ../../config/credentials.yaml
```

## Architecture

MatLLMSearch is integrated with SDE-Harness core components:

- **StructureGenerator**: Uses the SDE-Harness `Generation` class for LLM-based structure creation.
- **MaterialsOracle**: Subclasses the SDE-Harness `Oracle` to evaluate structures with CHGNet/M3GNet for stability and mechanical properties.
- **StabilityCalculator**: ML interatomic potentials for energy and mechanical property prediction.

## Evaluation Measurements

Crystal structure discovery. Each experiment began with an initial population of 100 groups of parents (100 × 2 = 200 parent structures), seeded from the MatBench-bandgap dataset selected with lowest deformation energy by CHGNet. The mutation and crossover operations for LLMs were implemented by prompting the LLMs with two sampled parent structures based on their fitness values (minimizing $E_\text{d}$) and querying them to propose 5 new structures either through mutation of one structure or crossover of both structures. After generating new offspring in each generation, we evaluated the new offspring and merged their evaluations with the parent evaluations from the previous iteration. The merged pool of parents and children were then ranked by their fitness values (minimizing $E_\text{d}$), and the top-100 × 2 candidates were kept in the population as the pool for the next iteration. We evaluate generated structures through metrics that assess validity, diversity, novelty, and stability. Structural validity checks three-dimensional periodicity, positive lattice volume, and valid atomic positions. Composition validity verifies positive element counts and reasonable number of elements ($\leq 10$). Structural diversity is computed by deduplicating the generated set using pymatgen's StructureMatcher algorithm, then calculating the ratio of unique structures to total generated. Composition diversity measures the fraction of distinct chemical compositions. For novelty assessment, we compare generated structures against the initial reference pool. Composition novelty identifies structures whose reduced formulas are absent from the reference set. Structural novelty is determined by grouping reference structures by formula, then for each generated structure with a matching formula, using StructureMatcher to check if it matches any reference structure with the same composition; unmatched structures are considered structurally novel. Stability evaluation uses CHGNet to relax structures and compute formation energy, then calculates energy above the convex hull ($E_\text{d}$) via a pre-computed patched phase diagram database. We report metastability rates at three thresholds: $E_\text{d} < 0.0$ eV/atom (thermodynamically stable), $E_\text{d} < 0.03$ eV/atom (highly metastable), and $E_\text{d} < 0.10$ eV/atom (M3GNet metastability criterion). The integrated SUN (Structures Unique and Novel) score combines stability and novelty: (1) filter to structures with $E_\text{d} < 0.0$ eV/atom; (2) identify unique structures within this stable subset using pymatgen's Structure.matches with scaling enabled; (3) check novelty against the reference pool; (4) compute SUN score as the number of structures simultaneously stable, unique, and novel, divided by the total number of generated structures.

## Results

Comparison of different methods on crystal structure generation. All LLM models were tested with `temperature: 1.0` and `max_tokens: 8000`. Parents and children are merged to form the next generation:

| Method | Structural Validity(%) | Comp Validity(%) | Metastability (E_d < 0.1 eV/atom, %) | Metastability (E_d < 0.0 eV/atom, %) | Sun Rate(%) |
|--------|---------------------|---------------|-----------------------------------|-----------------------------------|----------|
| CDVAE | 100 | 86.70 | 28.8 | - | - |
| DiffCSP | 100 | 83.25 | - | 5.06 | 3.34 |
| GPT-5-mini | 100 | 100 | 74.60 | 50.05 | 46.24 |
| GPT-5-chat | 100 | 100 | 64.36 | 46.93 | 44.37 |
| GPT-5 | 100 | 100 | 88.33 | 63.22 | 55.31 |
| Claude Sonnet 4.5 | 100 | 100 | 78.71 | 50.21 | 38.99 |
| DeepSeek Reasoner | 100 | 100 | 88.90 | 61.22 | 48.25 |
| Grok-4 | 100 | 100 | 87.13 | 60.29 | 49.80 |

## Output

Results are saved in the specified log directory:

### CSG / CSP Output
- `generations.csv`: Generated structures with properties for each iteration
- `metrics.csv`: Optimization metrics over iterations
- `token_usage_summary.json`: Token usage summary

### Analysis Output
- The file specified by `--output`: comprehensive evaluation results, including validity, diversity, novelty (with SUN score), and stability metrics.

## 🐛 Troubleshooting

1. **`Models configuration file not found` / `Credentials file not found`**
   ```bash
   cp config/models.template.yaml config/models.yaml
   cp config/credentials.template.yaml config/credentials.yaml
   # then add your API key(s)
   ```

2. **`Environment variable ... is not set`** (when using `${VAR}` in credentials.yaml)
   ```bash
   export OPENAI_API_KEY="your-actual-key"
   ```

3. **Missing phase diagram file** (E_hull calculation fails)
   ```bash
   wget -O data/2023-02-07-ppd-mp.pkl.gz https://figshare.com/ndownloader/files/48241624
   ```

4. **No GPU available**
   ```bash
   # Use an API model for generation and run evaluation on CPU
   python cli.py analyze --input data/llama_test.csv --device cpu --output eval.json
   ```

## Citation

If you use MatLLMSearch in your research, please cite:

```bibtex
@misc{gan2025matllmsearch,
      title={MatLLMSearch: Crystal Structure Discovery with Evolution-Guided Large Language Models},
      author={Jingru Gan and Peichen Zhong and Yuanqi Du and Yanqiao Zhu and Chenru Duan and Haorui Wang and Daniel Schwalbe-Koda and Carla P. Gomes and Kristin A. Persson and Wei Wang},
      year={2025},
      eprint={2502.20933},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2502.20933},
}
```
