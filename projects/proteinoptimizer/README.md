# ProteinOptimizer 

ProteinOptimizer is a framework for optimizing protein sequences using large language models (LLMs) and evolutionary algorithms. This project combines the power of state-of-the-art language models with traditional optimization techniques to improve protein properties.  It is a direct, self-contained re-implementation of the relevant parts of the original **LLMProteinOptimizer**
project, refactored to live exclusively inside the `sde-harness` codebase.

The default benchmark bundled with the repository is **Syn-3bfo** – a fitness
landscape derived from the _Streptomyces peptidase_ (PDB **3BFO**).  Two types
of optimisation are supported:

* **Single-objective** – maximise the Syn-3bfo fitness score.
* **Multi-objective** –
  * Weighted-sum aggregation (user-provided weights)
  * Pareto front (non-dominated selection)

All optimisation runs stream metadata to [Weave](https://wandb.ai/site/weave/)
for convenient experiment tracking.

---
## Installation
```bash
# create env (example — choose your own manager)
conda create -n proteinopt python=3.10
conda activate proteinopt

# install requirements for the project itself
pip install -r projects/proteinoptimizer/requirements.txt

# add optional extras (Weave tracking)
pip install weave
```

The code uses nothing outside the standard Python scientific stack
(`numpy, pandas, scipy`) plus `weave` for logging.

---
## Dataset placement
```
projects/
└─ proteinoptimizer/
   ├─ data/
   │  └─ Syn-3bfo/
   │     ├─ fitness.csv            # <— required (159 KB)
   │     └─ 3bfo_1_A_model_state_dict.npz  # optional Potts landscape (11 MB)
   └─ ...
```
* `fitness.csv` – ground-truth experimental scores (columns `Combo, fitness`).
* `*.npz` – direct-coupling Potts model exported from Mogwai (optional, speeds up
  evaluation & generalises outside the CSV lookup).

---
## Quick start

### 1. Single-objective
```bash
python cli.py single \
       --generations 10 \
       --population-size 100 \
       --offspring-size 200 \
       --mutation-rate 0.02 \
       --seed 0
       --model "openai/gpt-4o-2024-08-06"  # optional LLM
```

### 2. Multi-objective (weighted sum)
```bash
# maximise +1 * fitness  –  minimise 0.5 * fitness (demo)
python cli.py multi \
       --potts-weight 1.0 --hamming-weight -0.1 \
       --generations 8
```

### 3. Multi-objective (Pareto)
```bash
python cli.py multi-pareto \
       --generations 8
```

### 4. SDE-Harness Workflow
```bash
python cli.py workflow --generations 3
```

Logs & artefacts can be inspected in the Weave UI under
`proteinoptimizer_*` projects.

---
## Code overview
```
proteinoptimizer/
├─ cli.py                     # command-line entry point
├─ data/                      # Syn-3bfo dataset (see above)
├─ src/
│  ├─ core/
│  │  ├─ protein_optimizer.py # GA for sequences
│  │  ├─ multiobjective.py    # weighted-sum oracle wrapper
│  │  ├─ pareto.py            # non-dominated sort helpers
│  │  └─ pareto_optimizer.py  # simplified NSGA-II GA
│  ├─ utils/
│  │  └─ potts_model.py       # local copy of Google DCA Potts landscape
│  ├─ oracles/
│  │  └─ protein_oracles.py   # Syn-3bfo oracle (CSV / Potts)
│  └─ modes/
│     ├─ single_objective.py
│     ├─ multi_objective_protein.py
│     └─ multi_pareto_protein.py
└─ tests/ (legacy molecule tests kept for reference, ignored)
```

---
## Extending to other protein datasets
1. Drop a new `data/<DatasetName>/fitness.csv` (and optional Potts model) into
the `data/` folder.
2. Implement a new oracle in `src/oracles/` mirroring
   `Syn3bfoOracle`.
3. Wire it into CLI as needed.

---
## License
This refactor inherits the original Apache 2.0 license for the Potts model code
and follows the MIT license of SDE-Harness.  See the root `LICENSE` file.

## Citation

If you find this work useful, please cite our paper:

```
@article{wang2025large,
  title={Large Language Model is Secretly a Protein Sequence Optimizer},
  author={Wang, Yinkai and He, Jiaxing and Du, Yuanqi and Chen, Xiaohui and Li, Jianan Canal and Liu, Li-Ping and Xu, Xiaolin and Hassoun, Soha},
  journal={arXiv preprint arXiv:2501.09274},
  year={2025}
}
