# ProteinOptimizer — LLM-Guided Protein Sequence Optimization

<p align="center" style="width:90%;margin:0 auto;">
  <img src="assets/protein_main.png" alt="Framework" style="width:100%;max-width:1200px;min-width:300px;display:block;margin:0 auto;"/>
</p>

ProteinOptimizer evolves populations of fixed-length protein sequences with a genetic algorithm (GA)
whose mutation operator can be delegated to an LLM. It supports single-objective and multi-objective
(weighted-sum and Pareto) optimization over five protein fitness landscapes, and is a self-contained
re-implementation of **LLMProteinOptimizer** inside the `sde-harness` codebase.

## Original Code Repository
This project is a self-contained re-implementation of the relevant parts of the original **LLMProteinOptimizer** paper (LMRL Workshop @ ICLR 2025), refactored to live inside the `sde-harness` codebase.

Paper: [arXiv:2501.09274](https://arxiv.org/abs/2501.09274)

## Supported Datasets / Oracles

| Oracle key  | Dataset                       | Fitness source                              | Reported score                       |
|:------------|:------------------------------|:--------------------------------------------|:-------------------------------------|
| `syn-3bfo`  | Synthetic 3bfo landscape      | Potts model (`data/Syn-3bfo/*.npz`)          | Potts energy rescaled by [-3, 3]; unbounded, can be negative |
| `gb1`       | Protein G domain B1           | Experimental CSV (raw fitness ∈ [0, 8.76])   | min–max normalized to [0, 1]         |
| `trpb`      | Tryptophan synthase B subunit | Experimental CSV (raw fitness ∈ [0, 1])      | min–max normalized to [0, 1]         |
| `aav`       | AAV2 capsid protein           | Pre-trained CNN oracle (`src/utils/ckpt/AAV`)| min–max normalized to [0, 1]         |
| `gfp`       | Green fluorescent protein     | Pre-trained CNN oracle (`src/utils/ckpt/GFP`)| min–max normalized to [0, 1]         |

---

## 📦 Install

Run these commands from the harness root unless a step says otherwise.

### 1. Enter the ProteinOptimizer project folder

```bash
cd projects/proteinoptimizer
```

### 2. Data and model checkpoints (no download required)

**All data files and all model checkpoints are committed to this repository** — a plain
`git clone` of `sde-harness` already contains everything the five oracles need
(~32 MB of CSV/NPZ data plus ~92 MB of `.ckpt` files). There is nothing to fetch from
Zenodo/Hugging Face and no `wget` step.

Verify the files are present before running anything:

```bash
ls data/GB1/fitness.csv data/TrpB/fitness.csv data/Syn-3bfo/fitness.csv \
   data/Syn-3bfo/3bfo_1_A_model_state_dict.npz \
   data/AAV/ground_truth.csv data/GFP/ground_truth.csv
ls src/utils/ckpt/AAV/mutations_0/percentile_0.0_1.0/cnn_oracle.ckpt \
   src/utils/ckpt/GFP/mutations_0/percentile_0.0_1.0/cnn_oracle.ckpt
```

Expected layout (only the files actually loaded at runtime are shown):

```
projects/proteinoptimizer/
├─ data/
│  ├─ Syn-3bfo/
│  │  ├─ fitness.csv
│  │  └─ 3bfo_1_A_model_state_dict.npz      # Potts model weights
│  ├─ GB1/fitness.csv
│  ├─ TrpB/fitness.csv
│  ├─ AAV/ground_truth.csv
│  └─ GFP/ground_truth.csv
└─ src/utils/ckpt/                           # CNN oracles for AAV / GFP
   ├─ AAV/mutations_0/percentile_0.0_1.0/{cnn_oracle.ckpt,config.yaml}
   └─ GFP/mutations_0/percentile_0.0_1.0/{cnn_oracle.ckpt,config.yaml}
```

If `src/utils/ckpt/` is missing (e.g. a sparse or partial clone), the `aav` and `gfp`
oracles cannot run; `syn-3bfo`, `gb1` and `trpb` still work.

### 3. Set up the conda environment

```bash
conda env create -f environment.yml
conda activate ScienceBench_ProteinOptimizer
```

The environment installs PyTorch, `litellm`, `weave`, `omegaconf`, `pandas`, `matplotlib`,
`seaborn` and `tabulate`. A GPU is optional — the CNN oracles fall back to CPU automatically.

`numpy`, `pandas` and `scipy` are installed **by conda rather than pip on purpose**. Recent numpy
wheels are built for `manylinux_2_28`, so on an older system (CentOS 7 / glibc 2.17) pip finds no
usable wheel, falls back to the source build, and fails with `NumPy requires GCC >= 9.3`. The conda
packages are prebuilt against their own toolchain and sidestep this. Keep them in the conda section
if you edit this file.

<details>
<summary>Alternative: pip-only install</summary>

```bash
conda create -n ScienceBench_ProteinOptimizer python=3.10 -y
conda activate ScienceBench_ProteinOptimizer
pip install -r ../../requirements.txt
pip install -r requirements.txt
```
</details>

### 4. Configure model files in the harness root

```bash
cd ../..

cat > models.yaml <<'EOF'
openai/gpt-5-mini:
  provider: openai
  model: gpt-5-mini
  credentials: openai

openai/gpt-5:
  provider: openai
  model: gpt-5
  credentials: openai

openai/gpt-5-chat-latest:
  provider: openai
  model: gpt-5-chat-latest
  credentials: openai

anthropic/claude-sonnet-4-5:
  provider: anthropic
  model: claude-sonnet-4-5
  credentials: anthropic

deepseek/deepseek-reasoner:
  provider: deepseek
  model: deepseek-reasoner
  credentials: deepseek
EOF

cat > credentials.yaml <<'EOF'
openai:
  api_key: your-openai-key-here
anthropic:
  api_key: your-anthropic-key-here
deepseek:
  api_key: your-deepseek-key-here
EOF

cd projects/proteinoptimizer
```

`models.yaml` and `credentials.yaml` are read from the **harness root**, not from this project
folder. The value passed to `--model` must be a key in `models.yaml`. Any provider supported by
[LiteLLM](https://docs.litellm.ai/docs/providers) can be added the same way; to swap in a newer
model, add an entry here and pass its key to `--model`.

### 5. Weave logging (off by default)

Weave network logging is disabled by default so runs need no Weights & Biases account.
Enable it explicitly if you want the traces:

```bash
PROTEINOPT_ENABLE_WEAVE=true python cli.py single --oracle gb1 --generations 1 --model none
```

---

## 🎯 Usage

### Command Line Interface

```bash
python cli.py <mode> [options]
```

Modes:

| Mode           | Description                                                        | Writes JSON |
|:---------------|:-------------------------------------------------------------------|:------------|
| `single`       | Single-objective GA (fitness only) — used for the paper results     | yes         |
| `multi`        | Two-objective weighted sum: `w_f · fitness + w_h · Hamming distance`| yes         |
| `multi-pareto` | Two-objective Pareto front (fitness ↑, Hamming distance ↓)          | no (stdout) |
| `workflow`     | Same GA wrapped in the SDE-Harness `Workflow` loop                  | no (stdout) |

### Common Parameters

All modes accept:

- `--oracle`: one of `syn-3bfo`, `gb1`, `trpb`, `aav`, `gfp` (default: `syn-3bfo`)
- `--model`: model key from the harness-root `models.yaml` (default: `openai/gpt-5-mini`).
  Use `--model none` for the **baseline** (random mutations, no API calls).
- `--population-size`: population kept each generation (default: 10)
- `--offspring-size`: offspring produced per generation (default: 20)
- `--generations`: number of GA generations (default: 3)
- `--initial-size`: number of starting sequences (default: 20)
- `--mutation-rate`: per-position mutation probability (default: 0.01)
- `--seed`: random seed(s), space-separated for multiple runs (default: 0)
- `--output-dir`: directory for result JSON files (default: `results`)
- `--resume-results` / `--continue-generations`: continue from a previous run's JSON

`multi` additionally accepts `--fitness-weight` (default 0.5) and `--hamming-weight` (default -0.5).

### Verified Smoke Test

After completing the setup above, run this minimal, API-free check (takes well under a minute):

```bash
python cli.py single \
  --oracle gb1 \
  --generations 1 \
  --population-size 8 \
  --offspring-size 8 \
  --initial-size 8 \
  --model none \
  --output-dir results_smoke
```

It loads the GB1 landscape, runs one GA generation, prints the best sequence/score and writes
`results_smoke/results_single_gb1_0_baseline.json`. (`--output-dir results_smoke` keeps the
committed reference runs in `results/` intact.)

Then confirm the LLM path works end-to-end with one small API-backed run:

```bash
python cli.py single \
  --oracle gb1 \
  --generations 1 \
  --population-size 8 \
  --offspring-size 8 \
  --initial-size 8 \
  --model openai/gpt-5-mini \
  --output-dir results_smoke
```

This writes `results_smoke/results_single_gb1_0_gpt-5-mini.json`. The last lines of the run report how
many mutations actually came from the LLM:

```text
LLM-guided mutations: 8/8 succeeded, 0 fell back to random mutation.
```

**If `fell back to random mutation` is non-zero for every mutation, the run is the random baseline
wearing the model's name.** A bad API key, a model key missing from `models.yaml`, or a provider
outage all produce this; the first failure is printed as a `[WARN] LLM call to … failed (…)` line
with the underlying error. Fix the credentials before launching the full sweep.

### Output

Each `single` / `multi` run writes
`results/results_{single|multi}_<oracle>_<seed>_<model>.json` containing:

```jsonc
{
  "best_sequence": "…",               // best sequence found
  "best_score": 0.87,                 // its oracle score
  "best_scores_history": [...],       // best score after each generation
  "best_sequences_history": [...],    // the corresponding sequence per generation
  "final_population": [["SEQ", 0.87], ...],
  "oracle_calls": 1234,
  "llm_mutations": 800,               // mutations actually produced by the LLM
  "llm_fallbacks": 0,                 // mutations that fell back to random (both 0 for `--model none`)
  "all_results": {"SEQ": 0.87, ...}   // every sequence ever scored; dominates the file size
}
```

`single` and `multi` write the same set of keys. The `llm_mutations` / `llm_fallbacks` fields are
new — result files produced before this README will not have them, which analysis scripts tolerate.

`<model>` is the part of `--model` after the `/` (e.g. `gpt-5-mini`), or `baseline` for `--model none`.

---

## 🔁 Reproducing the Paper Results

### 1. Run all model × dataset combinations

```bash
bash run_all.sh
```

`run_all.sh` launches, for each model in `{none (baseline), openai/gpt-5-mini,
deepseek/deepseek-reasoner, anthropic/claude-sonnet-4-5, openai/gpt-5, openai/gpt-5-chat-latest}`
and each dataset in `{syn-3bfo, gb1, trpb, aav, gfp}`:

```bash
python cli.py single --oracle <dataset> --generations 8 \
  --population-size 200 --offspring-size 100 --seed 0 --model <model>
```

Each generation issues one LLM call per offspring attempt (up to `2 × offspring_size` attempts,
each retried up to 5 times if the model returns no valid sequence), so one LLM-backed
(model, dataset) run makes roughly 800–1600 API calls and the default sweep is the dominant cost.
Baseline (`--model none`) runs make no API calls and finish in minutes.

Results land in `results/`, logs in `logs/`. The script runs `python -u` so the per-generation
lines stream into the logs; without `-u`, Python block-buffers stdout when redirected to a file
and a long run looks frozen for many minutes at a time. Edit the `models`, `datasets`, `seeds` and GA-size
variables at the top of the script to change the sweep. Note that the script starts every job in
parallel in the background (30 jobs for the default sweep) — reduce the lists if you are rate-limited
by a provider. To run a single cell of the grid, just run the `python cli.py single …` command above
directly.

### 2. Summarize

```bash
python src/analyze.py --glob "./results/results_single_*.json" --higher-is-better 1
```

This prints one table of Top-1 / Top-5 / Top-10 scores averaged over the five datasets, per model.

> **`analyze.py` collapses model names into families.** Any model whose name starts with
> `deepseek` (both `deepseek-chat` and `deepseek-reasoner`) is averaged into a single `DeepSeek`
> row, and anything unrecognised is dropped as `Other`. Keep one model per results folder, or
> extend `_family()` in `src/analyze.py`, before comparing models.
>
> **Use the `results_single_*` glob, not `./results/*.json`.** The `results/` folder also contains
> multi-objective (`results_multi_*`) runs, whose scores are on a different scale; including them
> changes every number in the table (e.g. Baseline Top-1 becomes 0.4927 instead of 0.7514).

### 3. Plot

```bash
python src/plot.py --input_dir ./results --out_dir ./figures
```

Produces (PNG + PDF):
1. `ProteinOptimizerResult.*` — final Top-1 per model, averaged over tasks
2. `PO_top1_convergence.*` — Top-1 vs. generation per model
3. `PO_top1_by_task_grouped.*` — Top-1 per task per model

`plot.py` only reads files named `results_single_<task>_0_<model>.json` (seed 0).
Optional `--title_prefix "…"` adds a prefix to the plot titles.

### Reference numbers

Produced by the sweep above (`--generations 8 --population-size 200 --offspring-size 100 --seed 0`,
averaged over the five datasets):

| Model             |   Top_1 |   Top_5 |   Top_10 |
|:------------------|--------:|--------:|---------:|
| Baseline          |  0.7514 |  0.6899 |   0.6564 |
| GPT5-mini         |  0.7867 |  0.7262 |   0.6821 |
| DeepSeek          |  0.8713 |  0.8022 |   0.7649 |
| Claude-Sonnet-4-5 |  0.7759 |  0.6967 |   0.6427 |
| GPT-5             |  0.8561 |  0.8129 |   0.7842 |
| GPT-5-chat-latest |  0.8582 |  0.7896 |   0.7438 |

`Baseline` is `--model none` (random mutations). These are the JSON files committed under
`results/`; re-running `run_all.sh` overwrites them, so copy the folder first if you want to
keep the reference runs. LLM runs are not bit-reproducible (provider-side sampling), so expect
small deviations from these numbers even at a fixed `--seed`.

---

## 🏗️ Project Structure

```
projects/proteinoptimizer/
├── cli.py                      # Command line entry point
├── run_all.sh                  # Full paper sweep
├── environment.yml             # Conda environment (tested path)
├── requirements.txt            # pip fallback
├── src/
│   ├── modes/
│   │   ├── single_objective.py       # `single` mode
│   │   ├── multi_objective_protein.py# `multi` (weighted-sum) mode
│   │   └── multi_pareto_protein.py   # `multi-pareto` mode
│   ├── core/
│   │   ├── protein_optimizer.py      # GA + LLM-guided mutation
│   │   ├── pareto_optimizer.py       # Pareto GA
│   │   ├── pareto.py, multiobjective.py
│   ├── oracles/
│   │   ├── fitness_oracles.py        # GB1 / TrpB / Syn-3bfo (CSV + Potts)
│   │   ├── ml_oracles.py             # AAV / GFP (CNN checkpoints)
│   │   ├── multi_objective_oracles.py# Hamming-distance objective
│   │   └── base.py
│   ├── utils/
│   │   ├── potts_model.py, predictors.py, tokenize.py, evolutionary_ops.py
│   │   └── ckpt/                     # AAV & GFP CNN checkpoints (shipped)
│   ├── generation.py                 # LLM wrapper around sde_harness Generation
│   ├── workflow.py                   # SDE-Harness Workflow integration
│   ├── weave_utils.py                # Weave on/off switch
│   ├── analyze.py                    # Result summary table
│   └── plot.py                       # Figures
├── data/                       # Fitness datasets (shipped)
├── results/                    # Result JSONs (committed runs + new runs)
├── figures/                    # Generated figures
└── README.md                   # This document
```

---

## 🐛 Troubleshooting

### Common Issues

1. **`models.yaml` / `credentials.yaml` not found**

   Both files are read from the **harness root** (`sde-harness/`), not from this folder.
   ```bash
   ls ../../models.yaml ../../credentials.yaml
   ```

2. **API key errors / all models score like the baseline**

   Check that the `credentials:` tag in `models.yaml` matches a block in `credentials.yaml`
   and that the key is real. The GA catches generation errors and falls back to random
   mutation, so a bad key shows up as baseline-like scores rather than a crash.

3. **`FileNotFoundError` for a `.ckpt` or `fitness.csv`**

   Re-check step 2 of the install. `aav`/`gfp` need `src/utils/ckpt/{AAV,GFP}/...`;
   `syn-3bfo` needs `data/Syn-3bfo/3bfo_1_A_model_state_dict.npz`.

4. **Weave login prompt / offline errors**

   Weave is disabled by default (`WEAVE_DISABLED=true`). If you enabled it with
   `PROTEINOPT_ENABLE_WEAVE=true` and have no W&B account, unset it:
   ```bash
   unset PROTEINOPT_ENABLE_WEAVE
   ```

5. **`ModuleNotFoundError: No module named 'src'`**

   Run `cli.py` from inside `projects/proteinoptimizer/` (the CLI resolves `src.*` relative
   to the current directory).

6. **`analyze.py` prints an unaligned table**

   Install `tabulate` (`pip install tabulate`); without it the script falls back to plain text.

7. **`conda env create` is slow, or fails building `tiktoken`**

   The classic conda solver can take several minutes on this environment file; that is normal.
   To speed it up: `conda install -n base conda-libmamba-solver` and then
   `conda env create -f environment.yml --solver=libmamba`.
   `tiktoken` is pinned to `0.9.0` because newer releases publish no cp310 wheel and fall back to
   a source build that needs a Rust toolchain. If you change the pin and see
   `Failed building wheel for tiktoken`, put the pin back.

8. **`LLM-guided mutations: 0/N succeeded` even though the key works**

   Check the `[WARN]` line printed with the first failure. Two common causes beyond a bad key:
   - The provider returned an empty completion. Some models decline the protein-engineering
     prompt and return `finish_reason: refusal` with no content — verified on
     `anthropic/claude-sonnet-4-5` in August 2026. The run then silently reproduces the random
     baseline, so check `llm_fallbacks` in the result JSON before trusting a model's numbers.
   - The model spends its whole budget on reasoning tokens and returns no visible text
     (reasoning models). Add a larger token budget or disable thinking via `__call_args` in
     `models.yaml`.

   Sanity-check a single call before launching the sweep:
   ```bash
   cd ../..
   python -c "
   from sde_harness.core import Generation
   g = Generation(models_file='models.yaml', credentials_file='credentials.yaml')
   r = g.generate(prompt='Reply with the word OK', model_name='<your-model-key>', max_tokens=20)
   print(r['finish_reason'], repr(r['text']))"
   cd projects/proteinoptimizer
   ```

9. **`UserWarning: CUDA initialization: The NVIDIA driver on your system is too old`**

   The pip `torch` wheel is built against a newer CUDA than the local driver, so PyTorch falls
   back to CPU. Everything still runs (the CNN oracles are small); to use the GPU, install a
   torch build matching your driver, e.g.
   `pip install torch --index-url https://download.pytorch.org/whl/cu118`.

10. **CUDA out of memory / no GPU**

    The CNN oracles use `cuda` when available and CPU otherwise. To force CPU:
    ```bash
    CUDA_VISIBLE_DEVICES= python cli.py single --oracle aav --model none
    ```

---

## 📚 Examples

```bash
# Baseline GA, no API calls, Syn-3bfo
python cli.py single --oracle syn-3bfo --generations 8 --population-size 200 \
  --offspring-size 100 --model none

# LLM-guided single-objective run on AAV
python cli.py single --oracle aav --generations 8 --population-size 200 \
  --offspring-size 100 --model openai/gpt-5

# Weighted-sum multi-objective (maximize fitness, minimize Hamming distance)
python cli.py multi --oracle gb1 --generations 10 \
  --fitness-weight 1.0 --hamming-weight -0.2 --model openai/gpt-5

# Pareto front
python cli.py multi-pareto --oracle trpb --generations 20 --model openai/gpt-5

# SDE-Harness workflow wrapper
python cli.py workflow --oracle gb1 --generations 3 --model openai/gpt-5

# Multiple seeds in one command
python cli.py single --oracle gb1 --generations 8 --seed 0 1 2 --model openai/gpt-5-mini
```

---

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

---

## 📄 License

This refactor inherits the original Apache 2.0 license for the Potts model code and follows the
MIT license of SDE-Harness. See the root `LICENSE` file.

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wang2025large,
  title={Large Language Model is Secretly a Protein Sequence Optimizer},
  author={Wang, Yinkai and He, Jiaxing and Du, Yuanqi and Chen, Xiaohui and Li, Jianan Canal and Liu, Liping and Xu, Xiaolin and Hassoun, Soha},
  booktitle={Learning Meaningful Representations of Life (LMRL) Workshop at ICLR 2025},
  year={2025}
}
```

## 🔗 Related Links

- Paper: [arXiv:2501.09274](https://arxiv.org/abs/2501.09274)
- SDE-Harness framework: [https://github.com/HowieHwong/sde-harness](https://github.com/HowieHwong/sde-harness)
- Harness-level docs (Generation / Prompt / Oracle / Workflow): the root [`README.md`](../../README.md)
- Provider and model configuration: [LiteLLM providers](https://docs.litellm.ai/docs/providers)
