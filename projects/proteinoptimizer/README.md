# ProteinOptimizer 
<p align="center" style="width:90%;margin:0 auto;">
  <img src="assets/protein_main.png" alt="Framework" style="width:100%;max-width:1200px;min-width:300px;display:block;margin:0 auto;"/>
</p>
ProteinOptimizer is a framework for optimizing protein sequences using large language models (LLMs) and evolutionary algorithms. This project combines the power of state-of-the-art language models with traditional optimization techniques to improve protein properties.  It is a direct, self-contained re-implementation of the relevant parts of the original **LLMProteinOptimizer**
project, refactored to live exclusively inside the `sde-harness` codebase.

This project is an adaptation of the Mol-LLeo framework for protein sequence optimization. It uses a genetic algorithm (GA) to evolve populations of protein sequences towards desired objectives. It supports both single-objective and multi-objective optimization, with optional LLM-guided mutations.

## Supported Datasets / Oracles
* **Syn-3bfo**: A synthetic protein fitness landscape. Includes a Potts model for energy-based evaluation. The fitness score is unbounded.
* **GB1**: Protein G domain B1, with fitness data. Fitness range: [0, 8.76].
* **TrpB**: Tryptophan synthase, with fitness data. Fitness range: [0, 1].
* **AAV**: AAV2 Capsid protein, with fitness predicted by a pre-trained ML model.
* **GFP**: Green Fluorescent Protein, with fitness predicted by a pre-trained ML model.

---
## Dataset placement
The project expects the following file structure for data and ML models. The `GGS_utils` directory contains the necessary checkpoints for the `AAV` and `GFP` oracles.

```
projects/
└─ proteinoptimizer/
   ├─ data/
   │  ├─ Syn-3bfo/
   │  │  ├─ fitness.csv
   │  │  └─ 3bfo_1_A_model_state_dict.npz
   │  ├─ GB1/
   │  │  └─ fitness.csv
   │  ├─ TrpB/
   │  │  └─ fitness.csv
   │  ├─ AAV/
   │  │  └─ *.csv
   │  └─ GFP/
   │     └─ *.csv
   ├─ src/
   │  └─ utils/
   │     └─ GGS_utils/  # <— required for AAV and GFP oracles
   │        ├─ ckpt/
   │        └─ ...
   └─ ...
```
The fitness score of Syn-3bfo is not bounded.

---
## Quick start

### 1. Single-objective

```bash
# Optimize for AAV fitness using the ML model
python cli.py single --oracle aav --generations 4 --population-size 32 --offspring-size 32
```

### 2. Multi-objective (weighted sum)

This mode optimizes a combined score of two competing objectives:

*   **Fitness Score:** The score from the selected oracle (e.g., experimental fitness from the CSV or predicted Potts energy). A *higher* score is better.
*   **Hamming Distance:** The number of mutations between a sequence and a reference (wild-type). A *lower* distance means the sequence is more similar to the original.

The final score is a weighted sum: `(fitness_weight * Fitness) + (hamming_weight * Hamming Distance)`.

```bash
# Weighted-sum multi-objective with GB1
python cli.py multi --oracle gb1 --generations 10 --fitness-weight 1.0 --hamming-weight -0.2

# Weighted-sum multi-objective with Syn-3bfo (will use Potts model automatically)
python cli.py multi --oracle syn-3bfo --generations 10 --fitness-weight 1.0 --hamming-weight -0.2
```

### 3. Multi-objective (Pareto)

This mode doesn't use weights. Instead, it finds the set of "non-dominated" solutions (the Pareto front). It simultaneously tries to maximize the fitness/Potts score and minimize the Hamming distance.

```bash
# Pareto front with TrpB
python cli.py multi-pareto --oracle trpb --generations 20
```

### 4. SDE-Harness Workflow
```bash
python cli.py workflow --generations 3 --model "openai/gpt-4o-2024-08-06" --oracle gb1
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
@inproceedings{wang2025large,
  title={Large Language Model is Secretly a Protein Sequence Optimizer},
  author={Wang, Yinkai and He, Jiaxing and Du, Yuanqi and Chen, Xiaohui and Li, Jianan Canal and Liu, Liping and Xu, Xiaolin and Hassoun, Soha},
  booktitle={Learning Meaningful Representations of Life (LMRL) Workshop at ICLR 2025},
  year={2025}
}

```