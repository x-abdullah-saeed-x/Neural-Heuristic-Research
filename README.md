# Neural-Guided Heuristics vs Classical Pathfinding in Dynamic Grid-Mazes

**Course:** AI-2002 Artificial Intelligence  
**Track:** B (Research & Development)  
**Deliverable:** 3 (Comparative Analysis, Failure Analysis, and Final Report)
**Integration Branch:** `deliverable-3`
**Repository:** <https://github.com/x-abdullah-saeed-x/Neural-Heuristic-Research>

## Team

- **Abdullah Zia** (Evaluation, Analysis, and Report Lead) — A\* engine, evaluation harness, baselines, statistical analysis, failure analysis, fallback models, figures, LaTeX integration.
- **Muhammad Abdullah Saeed** (Data and Training Lead) — Dataset loaders, split manifests, training loop, hyperparameter tuning, checkpointing, artefact policy.
- **Muhammad Hamid Abad** (Model and Search Integration Lead) — Model architecture, feature extraction, heuristic wrappers, ablations, architecture diagram.

## Research Question and Result

> Can a neural network that predicts the exact cost-to-goal replace the Manhattan
> distance inside A\* and reduce node expansions on procedurally generated mazes?

**The hypothesis is not supported.** The learned field fits the target well but
does not improve search:

| Heuristic (282 test mazes) | Mean expanded | vs Manhattan | Optimality ratio | Runtime |
|---|---:|---:|---:|---:|
| Manhattan | 180.8 | — | 1.000 | 1.89 ms |
| Oracle $h^\ast$ | 52.8 | −70.8% | 1.000 | 0.63 ms |
| Neural raw | 557.0 | **+208.1%** | 1.033 | 6.62 ms |
| Neural clipped to Manhattan | 188.0 | +4.0% | 1.001 | 2.68 ms |

The raw neural heuristic wins on only 1.8% of mazes (paired Wilcoxon $p=4.56\times10^{-47}$,
bootstrap 95% CI [316.9, 440.3]); Manhattan clipping restores near-parity but still yields no
gain (wins 18.4%, $p=5.84\times10^{-11}$). Offline fidelity is high
($R^2=0.8597$, MAE $=4.35$ cells at $N=60$, macro AUC-ROC $=0.9449$), which isolates the
failure to the structure of the learned field rather than to regression fit.

Phase 5 explains the failure: the raw field violates adjacent-cell consistency
on 64.1% of free-cell pairs, overestimates near the goal and underestimates far
from it, and is most harmful on low-density and direct mazes where Manhattan is
already optimal. A feature ablation shows that the two goal-offset features
carry almost all of the signal; the $5\times5$ wall window adds regression
accuracy but increases expansions.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/                         # generated data (git-ignored)
├── data_scripts/                 # maze generation, labels, cleaning, features
├── src/
│   ├── search.py                 # deterministic A* with tie-breaking
│   ├── metrics.py                # regression, classification, search metrics
│   ├── final_analysis.py         # paired tests, consistency, bias, strata
│   ├── evaluate.py               # split reconstruction and evaluation harness
│   ├── heuristics.py             # Manhattan / oracle / neural wrappers
│   ├── sklearn_heuristic.py      # scikit-learn fallback wrapper
│   └── model.py, features.py, train.py, tune.py, ...
├── scripts/
│   ├── run_evaluation.py         # authoritative benchmark + alpha sweep
│   ├── failure_analysis.py       # Phase 5 paired tests, bias, strata
│   ├── run_feature_ablation.py   # seven feature-subset MLP runs
│   ├── sample_predictions.py     # qualitative search traces and fields
│   ├── run_torch_search.py       # checkpoint audit + optional smoke benchmark
│   ├── demo.py                   # one-maze interactive demo
│   ├── make_figures.py           # publication figures
│   ├── results_to_latex.py       # D2 tables
│   └── make_phase5_report_assets.py
├── models/                       # fallback joblib models + smoke checkpoint
├── results/
│   ├── evaluation/               # per-maze CSV, summaries, alpha sweep
│   ├── analysis/                 # failure, ablation, torch, sample outputs
│   └── reproducibility/          # checkpoint audit
├── figures/                      # 300 dpi figures
├── report/
│   ├── main.tex                  # final report
│   ├── sections/                 # phases 1–5, appendix, collaboration
│   └── generated/                # data-driven tables and macros
├── docs/VIVA_GUIDE.md            # viva questions and evidence map
└── tests/                        # 170 passed, 2 skipped
```

## Installation

```powershell
git clone https://github.com/x-abdullah-saeed-x/Neural-Heuristic-Research.git
cd Neural-Heuristic-Research
git checkout deliverable-3

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CPU-only PyTorch is optional; the headline results use the scikit-learn backend:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Reproducing the Results

### 1. Data (only if `data/cleaned` is absent)

```powershell
python data_scripts/generate_mazes.py --num-per-config 50 --seed 42
python data_scripts/compute_labels.py
python data_scripts/clean_data.py
python data_scripts/features.py
```

### 2. Tests

```powershell
python -m pytest -q
```

### 3. Fallback models (committed; retrain only if needed)

```powershell
python scripts/train_fallback_models.py --samples 100000 --max-iter 50 --seed 42
```

### 4. Authoritative evaluation

```powershell
python scripts/run_evaluation.py --backend sklearn --split test --output results/evaluation
```

This writes `test_per_maze.csv`, `test_summary.json`, `regression_metrics.json`,
`classification_metrics.json`, and `alpha_sweep.json`.

### 5. Phase 5 analysis

```powershell
python scripts/failure_analysis.py
python scripts/run_feature_ablation.py
python scripts/sample_predictions.py
python scripts/run_torch_search.py
python scripts/make_phase5_report_assets.py
```

`run_torch_search.py` audits the committed PyTorch checkpoint and refuses to
benchmark it unless `--allow-untrained` is given, because the checkpoint is a
one-epoch smoke test. See `results/reproducibility/torch_checkpoint_audit.json`.

### 6. Figures and report

```powershell
python scripts/make_figures.py
python scripts/results_to_latex.py
python scripts/make_ablation_summary.py
```

Compile `report/main.tex` in Overleaf, or locally with Tectonic from the
repository root (the report uses repo-root-relative paths):

```powershell
tectonic -Z "search-path=$((Get-Location).Path)" report/main.tex --outdir report
```

The compiled PDF is `report/main.pdf`.

## Interactive Demo

```powershell
python scripts/demo.py --list-samples
python scripts/demo.py --case good
python scripts/demo.py --case typical
python scripts/demo.py --case bad
python scripts/demo.py --all-samples
python scripts/demo.py --file mazes_perlin_20_10.npz --index 42 --mode clipped_manhattan
```

The demo prints an ASCII maze, the Manhattan and neural paths, expanded-cell
counts, and field diagnostics. It defaults to the scikit-learn model and raw
mode; `--backend torch` requires `--allow-untrained` for the smoke checkpoint.

## Model Artefacts

See [`models/README.md`](models/README.md) for checksums and reproduction notes.
The committed scikit-learn regressor and classifier produce every headline
number. The PyTorch checkpoint is a smoke test and is audited rather than
reported as a trained result.

## Known Limitations

1. **PyTorch checkpoint is a smoke run.** It was trained for one epoch on 20k
   cells; the report treats it as a reproducibility caveat, not a result.
2. **Synthetic static environments.** Mazes are 2D, four-connected, unit cost,
   and generated from two families (recursive backtracker and Perlin).
3. **Single training seed for the headline model.** The fold/seed checks cover
   the data split, but the fallback MLP is reported for seed 42 only.
4. **Ablation uses a 24-maze search subset.** Regression metrics use 60k test
   cells; the search-impact columns are therefore less precise than the main
   282-maze benchmark.
5. **Clipping is safe but not free.** Manhattan clipping restores admissibility
   and near-parity, yet it still adds 4.0% expansions and 1.7× runtime over
   Manhattan on this corpus.

## Contribution Matrix

| Member | GitHub | Core Modules | Primary Responsibilities |
|---|---|---|---|
| **Abdullah Zia** | `@HarveyWebbs` | `src/search.py`, `src/metrics.py`, `src/evaluate.py`, `src/analysis.py`, `src/final_analysis.py` | A\* search, evaluation harness, baselines, paired statistics, failure analysis, LaTeX integration |
| **Muhammad Abdullah Saeed** | `@x-abdullah-saeed-x` | `src/data.py`, `src/train.py`, `src/tune.py`, `src/config.py` | Data loaders, split manifests, training loop, tuning, checkpointing, artefact policy |
| **Muhammad Hamid Abad** | `@Hamid4325` | `src/model.py`, `src/features.py`, `src/heuristics.py` | Multi-task architecture, feature refactoring, heuristic wrappers, ablations |

## Viva Preparation

See [`docs/VIVA_GUIDE.md`](docs/VIVA_GUIDE.md) for the one-page evidence map,
likely examiner questions, and defensible answers grounded in the committed
artefacts.
