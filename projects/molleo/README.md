# MolLEO - Molecular Language-Enhanced Evolutionary Optimization

LLM-augmented evolutionary search over chemical space, refactored to run on the
`sde-harness` framework.

![image](images/README/molleo_overview.gif)

## Original Code Repository
[https://github.com/zoom-wang112358/MOLLEO](https://github.com/zoom-wang112358/MOLLEO)

Paper: [Efficient Evolutionary Search Over Chemical Space with Large Language Models](https://openreview.net/forum?id=awWiNvQwf3) (ICLR 2025) · [arXiv](https://arxiv.org/abs/2406.16976) · [Website](https://molleo.github.io/)

## 📦 Install

Run these commands from the harness root unless a step says otherwise.

### 1. Enter the MolLEO project folder

```bash
cd projects/molleo
```

### 2. Set up the conda environment

```bash
conda env create -f environment.yml
conda activate ScienceBench_MolLEO
```

The pins in `environment.yml` are load-bearing (about 7 GB on disk, mostly
PyTDC's transitive dependencies):

| Pin | Why |
| --- | --- |
| `scikit-learn==1.2.2` | TDC ships the JNK3/GSK3B/DRD2 oracles as scikit-learn pickles. Newer scikit-learn fails with `node array from the pickle has an incompatible dtype`. |
| `setuptools==80.10.2` | PyTDC imports `pkg_resources`, which was removed in setuptools 81. |
| `litellm==1.73.6.post1` + `pydantic==2.11.10` | litellm 1.73.x raises ``` `Message` is not fully defined ``` on pydantic ≥ 2.12, and litellm ≥ 1.98 cannot be imported on Python 3.10. |

If you prefer pip inside an existing Python 3.10 environment, use the identical
pin set in `requirements.txt`.

### 3. Download the required data and oracle models

There is no model checkpoint to fetch by hand, but the oracles and the starting
population are downloaded from the TDC servers on first use. TDC resolves its
cache relative to the working directory, so run this **from `projects/molleo`**.
It fetches the oracle models into `./oracle/` (~80 MB) and the ZINC starting
pool into `./data/zinc.tab` (~12 MB):

```bash
python -c "
from tdc import Oracle
from tdc.generation import MolGen

smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'  # aspirin
for name in ['QED', 'SA', 'LogP', 'DRD2', 'JNK3', 'GSK3β']:
    print(name, Oracle(name)(smiles))
MolGen(name='zinc')
print('TDC assets ready')
"
```

Expected output (the scores are deterministic, so they double as a check that
the oracle pickles loaded correctly):

```text
QED 0.5501217966938848
SA 1.580039750008826
LogP 1.1367875481574274
DRD2 0.00030779248192998733
JNK3 0.0
GSK3β 0.0
TDC assets ready
```

Scoring each oracle once matters: the pickles are only unpickled on first call,
so constructing them alone would not catch a scikit-learn mismatch. The
optimizer downloads the same files on demand, so this step is technically
optional — but doing it up front separates "download failed" from "run failed".

### 4. Configure model files in the harness root

Both files are required for **every** run, including `--model none`: the
optimizer builds its `Generation` object before it knows whether an LLM will be
called.

```bash
cd ../..

cat > models.yaml <<'EOF'
openai/gpt-4o-2024-08-06:
  provider: openai
  model: gpt-4o-2024-08-06
  credentials: openai
EOF

cat > credentials.yaml <<'EOF'
openai:
  api_key: ${OPENAI_API_KEY}
EOF

export OPENAI_API_KEY="your-api-key-here"
cd projects/molleo
```

Any LiteLLM-supported provider works — copy `config/models.template.yaml` and
`config/credentials.template.yaml` instead if you want the full example list.

## 🎯 Usage

### Command Line Interface

MolLEO exposes three optimization modes:

```bash
# Single objective optimization
python cli.py single --oracle jnk3

# Multi-objective optimization (weighted sum)
python cli.py multi --max-obj jnk3 qed --min-obj sa

# Multi-objective optimization (Pareto selection)
python cli.py multi-pareto --max-obj gsk3b qed --min-obj sa
```

#### Running with Parameters

```bash
# GPT-4o guided search on JNK3, 20 generations, three seeds
python cli.py single --oracle jnk3 --model openai/gpt-4o-2024-08-06 \
  --generations 20 --seed 1 2 3

# No LLM: plain genetic algorithm with random mutations
python cli.py single --oracle qed --model none --generations 20

# Weighted-sum multi-objective run with a larger population
python cli.py multi --max-obj jnk3 qed --min-obj sa \
  --population-size 50 --offspring-size 100 --generations 10
```

#### View Help

```bash
# View all modes
python cli.py --help

# View help for a specific mode
python cli.py single --help
```

### Common Parameters

All modes support the following parameters:

- `--model`: Model name from the harness root `models.yaml` (default: `openai/gpt-4o-2024-08-06`); `none` disables the LLM and runs random mutations only
- `--population-size`: Population kept between generations (default: 10)
- `--offspring-size`: Offspring generated per generation (default: 20)
- `--generations`: Number of generations (default: 3)
- `--mutation-rate`: Mutation probability (default: 0.01)
- `--initial-size`: Number of starting molecules sampled from ZINC (default: 20)
- `--seed`: One or more random seeds; the run is repeated once per seed (default: 0)
- `--output-dir`: Directory for the result JSON (default: `results`)

Mode-specific parameters:

- `single`: `--oracle {jnk3,gsk3b,drd2,qed,sa,logp}` (required)
- `multi`: `--max-obj` (required), `--min-obj`, `--weights`
- `multi-pareto`: `--max-obj` (required), `--min-obj`

### Verified Smoke Test

After completing the setup above, run this minimal check (no API key needed —
it never calls an LLM):

```bash
python cli.py single --oracle qed --model none \
  --generations 2 --population-size 6 --offspring-size 6 --initial-size 6 --seed 0
```

It takes about 10 seconds (longer on the very first run if you skipped step 3,
because the ZINC pool is fetched then) and ends with:

```text
📊 Optimization Results:
Best molecule: CC[C@](C)(C#N)NC(=O)C1=CC(F)=CC=C1Br
Best score: 0.9324
Total oracle calls: 12

Results saved to: results/results_qed_random_0.json
```

Seeds are applied to Python's `random` and to NumPy, so a repeated run with the
same seed and the pinned environment reproduces the same molecules and scores
byte for byte. (Numbers above were produced on Linux / Python 3.10 / RDKit
2023.9.6 from a clean `conda env create`; a different RDKit build can shift the
molecules it explores.) Runs that call an LLM are not deterministic — the
provider's sampling is outside our control.

To check the LLM path end to end (this one does spend tokens):

```bash
python cli.py single --oracle qed --model openai/gpt-4o-2024-08-06 \
  --generations 1 --population-size 4 --offspring-size 4 --initial-size 4 --seed 0
```

### Running Against a Local Model

Any OpenAI-compatible endpoint works, so you can drive the search with a locally
served model instead of a paid API. Serve the model, e.g. with vLLM:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8123 --served-model-name qwen3-4b
```

then add it to the harness root config:

```yaml
# models.yaml
local/qwen3-4b:
  provider: openai
  model: qwen3-4b
  credentials: local_vllm

# credentials.yaml
local_vllm:
  api_key: EMPTY
  api_base: http://127.0.0.1:8123/v1
```

```bash
python cli.py single --oracle logp --model local/qwen3-4b --generations 5
```

### Output

Each seed prints per-generation progress, a result summary, and writes a JSON
file to `--output-dir`:

```text
Generation 1: Best score = 0.9324, Oracle calls = 9
Generation 2: Best score = 0.9324, Oracle calls = 12

📊 Optimization Results:
Best molecule: ...
Best score: ...
Total oracle calls: ...

Results saved to: results/results_qed_random_0.json
```

One file is written per seed, e.g. `--seed 1 2` produces
`results_logp_random_1.json` and `results_logp_random_2.json`. The JSON contains
`best_molecule`, `best_score`, `best_scores_history`, `best_molecules_history`,
`final_population`, `oracle_calls`, and `all_results` (every molecule the oracle
ever scored). Pareto runs additionally contain `pareto_front`.

Runs are chatty: every reproduction prints `DEBUG:` lines and RDKit prints
sanitization warnings for intermediate molecules. Both are normal.

Everything a run writes stays inside `projects/molleo` — `oracle/` and `data/`
(TDC downloads) and `results/` (run output). All three are git-ignored.

## 🔄 Differences from the Original MolLEO

If you are coming from [the original repository](https://github.com/zoom-wang112358/MOLLEO),
the entry point changed:

| Original | This port |
| --- | --- |
| `cd single_objective && python run.py molleo --mol_lm GPT-4 --oracles jnk3 --seed 1 2 3` | `python cli.py single --oracle jnk3 --model openai/gpt-4o-2024-08-06 --seed 1 2 3` |
| `cd multi_objective && python run.py molleo_multi --min_obj sa --max_obj jnk3 qed` | `python cli.py multi --min-obj sa --max-obj jnk3 qed` |
| `python run.py molleo_multi_pareto ...` | `python cli.py multi-pareto ...` |

- `--mol_lm` became `--model` and accepts any model declared in the harness root
  `models.yaml` (any LiteLLM provider), or `none` for the LLM-free GA baseline.
- The BioT5 and MoleculeSTM molecule editors of the original are **not** wired
  into this port; LLM mutations go through the harness `Generation` class. The
  original implementation is kept under `_archive/` for reference only.
- The genetic operators are vendored under `src/ga/`, so no code outside this
  folder is needed.

## 🏗️ Project Structure

```
projects/molleo/
├── cli.py                  # Command line entry point
├── src/                    # Source code directory
│   ├── core/               # Core components
│   │   ├── molleo_optimizer.py  # Evolutionary loop + LLM mutations
│   │   └── prompts.py           # Prompt templates
│   ├── ga/                 # Genetic algorithm operations
│   │   ├── crossover.py    # Molecular crossover
│   │   └── mutations.py    # Molecular mutations
│   ├── modes/              # Running mode modules
│   │   ├── single_objective.py  # Single objective mode
│   │   └── multi_objective.py   # Weighted-sum and Pareto modes
│   ├── oracles/            # Property evaluation
│   │   ├── base.py         # Base oracle class
│   │   └── tdc_oracles.py  # TDC oracle wrappers
│   ├── utils/              # Utility modules
│   │   ├── evolutionary_ops.py  # Mating pool, reproduction
│   │   └── mol_utils.py         # SMILES/Mol helpers
│   ├── generation.py       # SDE-Harness Generation subclass
│   └── weave_utils.py      # Weave logging helpers
├── tests/                  # Runnable example scripts
├── example_usage.py        # Python API examples
├── environment.yml         # Environment configuration
├── requirements.txt        # Same pins, for pip-only setups
└── README.md               # This document
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   ```bash
   export OPENAI_API_KEY="your-actual-key"
   ```

2. **`Models configuration file not found: .../models.yaml`**

   The harness root needs `models.yaml` and `credentials.yaml` even for
   `--model none`. Re-run step 4 of the install section.

3. **`node array from the pickle has an incompatible dtype`**

   scikit-learn drifted away from the pinned version; the TDC oracle pickles
   only load on 1.2.x.
   ```bash
   pip install "scikit-learn==1.2.2"
   ```

4. **`ModuleNotFoundError: No module named 'pkg_resources'`**

   setuptools 81+ dropped `pkg_resources`, which PyTDC imports.
   ```bash
   pip install "setuptools==80.10.2"
   ```

5. **LLM calls always fall back to random mutations**

   MolLEO catches generation errors and falls back to a random mutation, so a
   broken LLM configuration still looks like a successful run. Grep the output
   for `LLM mutation failed:` — the message names the cause, most often:

   - `Environment variable OPENAI_API_KEY is not set` → export the key (step 4).
   - ``` `Message` is not fully defined ``` → litellm/pydantic mismatch:
     ```bash
     pip install "litellm==1.73.6.post1" "pydantic==2.11.10"
     ```

6. **Missing Data Files / TDC download failures**
   ```bash
   # Check what has been downloaded (paths are relative to projects/molleo)
   ls oracle/ data/
   # Re-run the download step from section 3
   ```

7. **Weave Login / Logging Issues**

   MolLEO disables Weave network logging by default so local checks need no
   Weights & Biases account or network access. To enable it for a real run:
   ```bash
   MOLLEO_ENABLE_WEAVE=true python cli.py single --oracle qed --model none --generations 1
   ```

8. **Environment Issues**
   ```bash
   # Recreate environment
   conda env remove -n ScienceBench_MolLEO
   conda env create -f environment.yml
   conda activate ScienceBench_MolLEO
   ```

## 📚 Examples

### Quick Start

```bash
# 1. Activate the environment and set your key
conda activate ScienceBench_MolLEO
export OPENAI_API_KEY="your-key"
cd projects/molleo

# 2. Run the LLM-free baseline
python cli.py single --oracle qed --model none

# 3. Run LLM-guided search on a protein target
python cli.py single --oracle jnk3 --model openai/gpt-4o-2024-08-06 --generations 10
```

### Advanced Usage

```bash
# Pareto multi-objective search over three seeds
python cli.py multi-pareto \
  --max-obj gsk3b qed \
  --min-obj sa \
  --model openai/gpt-4o-2024-08-06 \
  --population-size 50 \
  --offspring-size 100 \
  --generations 10 \
  --seed 1 2 3 \
  --output-dir results/pareto
```

### Python API

Run from `projects/molleo` so that `src` is importable:

```python
from src.core import MolLEOOptimizer
from src.oracles import TDCOracle

oracle = TDCOracle("qed")

optimizer = MolLEOOptimizer(
    oracle=oracle,
    population_size=10,
    offspring_size=10,
    mutation_rate=0.05,
    model_name="openai/gpt-4o-2024-08-06",
    use_llm_mutations=True,
)

results = optimizer.optimize(
    starting_smiles=["CCO", "CCN", "CCC"],
    num_generations=3,
)
print(results["best_molecule"], results["best_score"])
```

`example_usage.py` shows the same API driven through the harness `Workflow`
class. `tests/` holds runnable scripts: `example_no_llm.py` needs no API key
(a 10-generation QED run, ~15 seconds; it sets no seed of its own, so its
numbers move between runs), while `example_with_llm.py` and
`example_comprehensive.py` need a configured model.

## 📄 License

This project is based on the original MolLEO project and follows the
corresponding license.

## 📖 Citation

```
@inproceedings{
      wang2025efficient,
      title={Efficient Evolutionary Search Over Chemical Space with Large Language Models},
      author={Haorui Wang and Marta Skreta and Cher Tian Ser and Wenhao Gao and Lingkai Kong and Felix Strieth-Kalthoff and Chenru Duan and Yuchen Zhuang and Yue Yu and Yanqiao Zhu and Yuanqi Du and Alan Aspuru-Guzik and Kirill Neklyudov and Chao Zhang},
      booktitle={The Thirteenth International Conference on Learning Representations},
      year={2025},
      url={https://openreview.net/forum?id=awWiNvQwf3}
}
```

## 🔗 Related Links

- Original code repository: [https://github.com/zoom-wang112358/MOLLEO](https://github.com/zoom-wang112358/MOLLEO)
- Project website: [https://molleo.github.io/](https://molleo.github.io/)
- Oracle definitions: [Therapeutics Data Commons](https://tdcommons.ai/functions/oracles/)
