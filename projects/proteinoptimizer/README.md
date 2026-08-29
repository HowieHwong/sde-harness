# ProteinOptimizer

<p align="center" style="width:90%;margin:0 auto;">
  <img src="assets/protein_main.png" alt="Framework" style="width:100%;max-width:1200px;min-width:300px;display:block;margin:0 auto;"/>
</p>

ProteinOptimizer is a framework for optimizing protein sequences using large language models (LLMs) and evolutionary algorithms. A genetic algorithm (GA) evolves populations of fixed-length sequences toward a chosen fitness objective; mutations can be sampled uniformly at random (baseline) or proposed by an LLM. Both single-objective and multi-objective (weighted-sum or Pareto) optimization are supported.

## Original Code Repository
This project is a self-contained re-implementation of the relevant parts of the original **LLMProteinOptimizer** paper (LMRL Workshop @ ICLR 2025), refactored to live inside the `sde-harness` codebase.

## Supported Datasets / Oracles
* **Syn-3bfo**: A synthetic protein fitness landscape. Includes a Potts model for energy-based evaluation. Fitness score is unbounded.
* **GB1**: Protein G domain B1, with experimental fitness data. Fitness range: [0, 8.76].
* **TrpB**: Tryptophan synthase, with experimental fitness data. Fitness range: [0, 1].
* **AAV**: AAV2 Capsid protein, with fitness predicted by a pre-trained CNN oracle.
* **GFP**: Green Fluorescent Protein, with fitness predicted by a pre-trained CNN oracle.

## 📦 Install

1. Clone the repository and get to the project folder:
   ```bash
   git clone https://github.com/HowieHwong/sde-harness.git
   cd sde-harness
   git checkout project/proteinoptimizer
   cd projects/proteinoptimizer
   ```

2. Data and model checkpoints — **no external download required.**
   All datasets (`data/*/`) and pre-trained CNN oracle checkpoints (`src/utils/ckpt/*/`) for the five supported oracles are shipped with the repository. After `git clone` you should see:
   ```bash
   ls data/                           # AAV  GB1  GFP  Syn-3bfo  TrpB
   ls src/utils/ckpt/                 # AAV  GFP   (CNN oracles for AAV and GFP)
   ```
   If any subfolder is missing (e.g. `data/Syn-3bfo/3bfo_1_A_model_state_dict.npz` or `src/utils/ckpt/AAV/mutations_0/percentile_0.0_1.0/cnn_oracle.ckpt`), re-run `git checkout project/proteinoptimizer` inside the repo — those blobs come from the branch.

3. Set up the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ScienceBench_ProteinOptimizer
   ```
   This installs PyTorch, pandas/numpy/matplotlib/seaborn, `omegaconf` (needed to load the CNN oracle configs), the `sde_harness` framework in editable mode (from the repo root), and LLM-provider SDKs (`openai`, `anthropic`, `google-generativeai`) plus `litellm` and `weave`.

4. Configure LLM providers (required for anything other than `--model none`).
   From the repository root:
   ```bash
   cd ../..                                                    # back to sde-harness root
   cp config/models.template.yaml   models.yaml
   cp config/credentials.template.yaml credentials.yaml
   # Edit credentials.yaml and fill in api_key values for the providers you plan to use
   # (openai, anthropic, deepseek, ...). Edit models.yaml if you need to add a
   # model tag that isn't in the template (e.g. openai/gpt-5, anthropic/claude-sonnet-4-5).
   cd projects/proteinoptimizer
   ```
   Both `models.yaml` and `credentials.yaml` are resolved from the repo root by the LLM-guided GA (`src/core/protein_optimizer.py`). Model tags used in `run_all.sh` are `openai/gpt-5-mini`, `openai/gpt-5`, `openai/gpt-5-chat-latest`, `deepseek/deepseek-reasoner`, and `anthropic/claude-sonnet-4-5` — add any missing tags to `models.yaml`.

5. (Optional) Disable Weave logging.
   By default each mode calls `weave.init(...)`. To run offline without a W&B/Weave account:
   ```bash
   export WEAVE_DISABLED=true
   ```

## 🎯 Usage

### Command Line Interface

ProteinOptimizer exposes four sub-commands via `cli.py`:

| Sub-command | Purpose |
| ----------- | ------- |
| `single`         | Single-objective GA on any oracle |
| `multi`          | Multi-objective GA with a weighted-sum score (fitness + Hamming distance) |
| `multi-pareto`   | Multi-objective GA with Pareto-front selection |
| `workflow`       | Runs the GA inside an SDE-Harness `Workflow` (logs to Weave) |

#### Basic Usage

```bash
# Single-objective, LLM-free baseline
python cli.py single --oracle gb1 --model none

# Single-objective with GPT-5
python cli.py single --oracle aav --model openai/gpt-5

# Multi-objective (weighted sum: maximize fitness, minimize Hamming)
python cli.py multi --oracle gb1 --fitness-weight 1.0 --hamming-weight -0.2 --model openai/gpt-5

# Multi-objective with Pareto selection
python cli.py multi-pareto --oracle trpb --model openai/gpt-5

# SDE-Harness Workflow
python cli.py workflow --oracle gb1 --model openai/gpt-5
```

#### Running with Parameters

```bash
# Match the settings used in run_all.sh (results below)
python cli.py single \
  --oracle gb1 \
  --model openai/gpt-5 \
  --generations 8 \
  --population-size 200 \
  --offspring-size 100 \
  --seed 0
```

#### View Help

```bash
# View all modes
python cli.py --help

# View help for a specific mode
python cli.py single --help
```

### Common Parameters

All modes support:

- `--oracle`: One of `syn-3bfo | gb1 | trpb | aav | gfp` (default: `syn-3bfo`).
- `--model`: LLM tag from `models.yaml`, or `none` for random mutations (default: `openai/gpt-5-mini`).
- `--population-size`: Population size (default: 10).
- `--offspring-size`: Offspring produced per generation (default: 20).
- `--generations`: Number of generations (default: 3).
- `--mutation-rate`: Per-position mutation probability (default: 0.01).
- `--initial-size`: Number of initial sequences (default: 20).
- `--seed`: Random seed(s); accepts multiple values, e.g. `--seed 0 1 2` (default: `[0]`).
- `--output-dir`: Directory for JSON results (default: `results`).
- `--resume-results`: Path to a previous run's `results_single_*.json` to continue from.
- `--continue-generations`: When resuming, how many additional generations to run.

Multi-only:
- `--fitness-weight`: Weight for fitness/Potts energy (maximize).
- `--hamming-weight`: Weight for Hamming distance from wild-type (minimize).

### Batch Script

`run_all.sh` reproduces every model × dataset cell from the results table:
```bash
bash run_all.sh
```
It launches one background `python3 cli.py single …` per (model, oracle) with `--generations 8 --population-size 200 --offspring-size 100`, logging to `<oracle>_single.log`. Edit the `models` and `datasets` arrays as needed.

### Evaluation

Aggregate the JSON results into a summary table:
```bash
python src/analyze.py --glob "./results/*.json" --higher-is-better 1
```

### Visualization

Generate publication-quality plots from result JSON files:
```bash
python src/plot.py --input_dir ./results --out_dir ./figures
```

Optional arguments:
- `--title_prefix`: Add a prefix to plot titles (e.g. `--title_prefix "Experiment 1"`).

The script generates three figures:
1. **ProteinOptimizerResult.{png,pdf}** — bar plot of final Top-1 per model, averaged across all tasks.
2. **PO_top1_convergence.{png,pdf}** — Top-1 score vs. iteration for each model.
3. **PO_top1_by_task_grouped.{png,pdf}** — grouped bar plot of Top-1 per task per model.

The script expects JSON files named `results_single_<task>_<seed>_<model>.json` in the input directory (this is the default naming used by `cli.py single`).

## Results

Mean over all five oracles (`syn-3bfo`, `gb1`, `trpb`, `aav`, `gfp`) at seed 0, using the run_all.sh settings (`--generations 8 --population-size 200 --offspring-size 100`). Numbers are exactly what `src/analyze.py --higher-is-better 1` prints.

| Model             |  Top_1 |  Top_5 | Top_10 |
|:------------------|-------:|-------:|-------:|
| Baseline          | 0.7514 | 0.6899 | 0.6564 |
| GPT5-mini         | 0.7867 | 0.7262 | 0.6821 |
| DeepSeek          | 0.8713 | 0.8022 | 0.7649 |
| Claude-Sonnet-4-5 | 0.7759 | 0.6967 | 0.6427 |
| GPT-5             | 0.8561 | 0.8129 | 0.7842 |
| GPT-5-chat-latest | 0.8582 | 0.7896 | 0.7438 |

- `Top_1` — mean across oracles of each run's best score.
- `Top_5` / `Top_10` — mean across oracles of each run's average score over the top-5 / top-10 candidates in the final population.

Reproduce with:
```bash
bash run_all.sh
python src/analyze.py --glob "./results/*.json" --higher-is-better 1
```

## 🏗️ Project Structure

```
projects/proteinoptimizer/
├── cli.py                       # Command line entry point
├── run_all.sh                   # Batch driver: all models × all oracles
├── environment.yml              # Conda environment (includes -e ../..)
├── requirements.txt             # Extra pip deps (pandas, matplotlib, omegaconf, ...)
├── data/                        # Datasets (shipped with the repo)
│   ├── Syn-3bfo/
│   │   ├── fitness.csv
│   │   └── 3bfo_1_A_model_state_dict.npz   # Potts model weights
│   ├── GB1/fitness.csv
│   ├── TrpB/fitness.csv
│   ├── AAV/ground_truth.csv                # + AAV_wild_type.csv, gt_medium_range.csv, ...
│   └── GFP/ground_truth.csv                # + GFP_wild_type.csv, mutation_point.csv, ...
├── src/
│   ├── core/                    # ProteinOptimizer GA + Pareto / multi-objective wrappers
│   ├── modes/                   # single_objective / multi_objective_protein / multi_pareto_protein / workflow
│   ├── oracles/                 # FitnessOracle (CSV-based) + AAV/GFP CNN oracles + PottsObjective
│   ├── utils/
│   │   ├── potts_model.py       # Mogwai Potts loader
│   │   ├── predictors.py        # BaseCNN
│   │   ├── tokenize.py          # Amino-acid tokenizer
│   │   └── ckpt/                # Pre-trained CNN oracles (shipped with the repo)
│   │       ├── AAV/mutations_0/percentile_0.0_1.0/{cnn_oracle.ckpt, config.yaml}
│   │       └── GFP/mutations_0/percentile_0.0_1.0/{cnn_oracle.ckpt, config.yaml}
│   ├── workflow.py              # SDE-Harness Workflow wrapper
│   ├── generation.py            # Thin wrapper over sde_harness.core.Generation
│   ├── analyze.py               # Summary table from result JSONs
│   └── plot.py                  # Publication-quality plots
├── results/                     # JSON outputs from cli.py (written at run time)
├── figures/                     # PNG/PDF outputs from plot.py
├── assets/                      # README figures
└── README.md                    # This document
```

## 🔧 Extension Development

### Adding a New Oracle / Dataset

1. Drop the new dataset into `data/<DatasetName>/fitness.csv` (and, optionally, a Potts `*.npz`).
2. Add an oracle class in `src/oracles/`:
   * For a CSV-lookup fitness landscape, subclass `FitnessOracle` in `fitness_oracles.py` (mirror `GB1Oracle` / `TrpBOracle` / `Syn3bfoOracle`).
   * For an ML-model oracle, mirror `AAVOracle` / `GFPOracle` in `ml_oracles.py` and place the checkpoint under `src/utils/ckpt/<DatasetName>/...`.
3. Export it from `src/oracles/__init__.py`.
4. Add the string tag to `oracle_choices` in `cli.py` and the `if args.oracle == ...` dispatch in `src/modes/single_objective.py` (and the multi-objective modes if needed).

### Adding a New Mode

1. Create `src/modes/new_mode.py` implementing `run_new_mode(args)`.
2. Add a subparser in `cli.py` (mirror the `single` / `multi` subparsers).
3. Import and dispatch it inside the `if args.mode == ...` block in `cli.py`.

## 🐛 Troubleshooting

### Common Issues

1. **`ModuleNotFoundError: sde_harness`**
   The environment was not installed via `environment.yml`, or `-e ../..` failed. Re-run:
   ```bash
   pip install -e ../..
   ```

2. **API Key / credentials errors**
   Confirm `models.yaml` and `credentials.yaml` exist at the **repo root** (not inside `projects/proteinoptimizer/`), and that the `credentials:` tag on each model in `models.yaml` matches a section in `credentials.yaml`.
   ```bash
   ls ../../models.yaml ../../credentials.yaml
   ```

3. **Missing checkpoint (`FileNotFoundError: cnn_oracle.ckpt`)**
   You are likely on the wrong branch or the checkpoint blobs were skipped:
   ```bash
   git checkout project/proteinoptimizer
   ls src/utils/ckpt/AAV/mutations_0/percentile_0.0_1.0/
   ```

4. **Weave asks for a W&B login and blocks the run**
   Disable Weave for offline execution:
   ```bash
   export WEAVE_DISABLED=true
   ```

5. **CUDA out of memory / no GPU**
   The CNN oracles will run on CPU if CUDA is not available (see `torch.cuda.is_available()` fallback in `src/oracles/ml_oracles.py`), just more slowly.

## 📚 Examples

### Quick Start

```bash
# 1. Environment
conda activate ScienceBench_ProteinOptimizer
export OPENAI_API_KEY="your-key"          # optional if using credentials.yaml
cd projects/proteinoptimizer

# 2. Baseline (no LLM) on GB1
python cli.py single --oracle gb1 --model none --generations 4

# 3. LLM-guided single-objective on AAV
python cli.py single --oracle aav --model openai/gpt-5 --generations 4

# 4. Aggregate + plot
python src/analyze.py --glob "./results/*.json" --higher-is-better 1
python src/plot.py --input_dir ./results --out_dir ./figures
```

### Reproducing the Paper Table

```bash
bash run_all.sh
# Wait for all background jobs to finish, then:
python src/analyze.py --glob "./results/*.json" --higher-is-better 1
python src/plot.py --input_dir ./results --out_dir ./figures
```

## 📄 License

This refactor inherits the original Apache 2.0 license for the Potts model code and follows the MIT license of SDE-Harness. See the root `LICENSE` file.

## 🔗 Related Links

- Parent framework: [SDE-Harness](https://github.com/HowieHwong/sde-harness)
- LiteLLM (LLM provider abstraction): [https://docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers)
- Weave (experiment logging): [https://wandb.ai/site/weave/](https://wandb.ai/site/weave/)

## Citation

If you find this work useful, please cite our paper:

```
@inproceedings{wang2025large,
  title={Large Language Model is Secretly a Protein Sequence Optimizer},
  author={Wang, Yinkai and He, Jiaxing and Du, Yuanqi and Chen, Xiaohui and Li, Jianan Canal and Liu, Liping and Xu, Xiaolin and Hassoun, Soha},
  booktitle={Learning Meaningful Representations of Life (LMRL) Workshop at ICLR 2025},
  year={2025}
}
```
