# Deliverable 3 report integration — preserving corrected Deliverable 2

This ZIP contains **only new paths**. Extract at the repository root (`Neural Heuristic Research/`), with folder structure preserved. Do not merge the old `deliverable-3` branch or replace `report/main.tex`.

## What the new files do

- `report/d3_final/phase5_results.tex` — complete final comparative-analysis chapter, using the user's actual D3 experiments.
- `report/d3_final/validate_evidence.py` — checks the source results without modifying the data.
- `report/d3_final/build_report.py` — creates a **full, consolidated D1+D2+D3 manuscript** by reading the existing corrected `report/main.tex` and inserting the Phase 5 chapter. The source D2 report remains untouched.
- `report/d3_final/README.md` — build instructions and provenance limitations.
- `docs/D3_FINAL_REPORT_GUIDE.md` — this guide.

The generated `report/d3_final/main.tex` also stays separate from the original D2 report and should be committed only after checking it and successfully building a PDF.

## Required evidence already created locally

- `figures/d3/prediction_fields.png`
- `results/analysis/d3/full_test_raw/torch_search.json`
- `results/analysis/d3/full_test_raw/failure_analysis/failure_summary.json`
- `results/analysis/d3/full_test_raw/field_diagnostics/field_summary.json`
- `results/analysis/d3/full_test_raw/field_diagnostics/inconsistency_by_size.png`
- `results/analysis/d3/full_test_raw/field_diagnostics/inconsistency_vs_expansion.png`
- `results/analysis/d3/full_test_raw/field_diagnostics/prediction_mae_by_size.png`
- `results/analysis/d3/feature_ablation_full/feature_ablation.csv`
- Corrected D2 assets referenced by `report/main.tex` and its included sections.

## Steps (in Windows CMD from the project root)

```bat
python report/d3_final/build_report.py
python report/d3_final/validate_evidence.py
python -m pytest -q
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report/d3_final report/d3_final/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report/d3_final report/d3_final/main.tex
```

If `pdflatex` is not installed, upload the full repository's required `report/`, `figures/`, and necessary `results/analysis/d3/` assets to Overleaf and set `report/d3_final/main.tex` as the main file. Its inputs/figure paths assume compilation from the repository root.

If compilation fails, send the **first** LaTeX error message. Don't alter the existing D2 report to fix a problem in the D3 copy.

## Scientific provenance

- Main D3 PyTorch search: trained checkpoint `neural_heuristic_seed42.pt` (best epoch 6, val MAE 0.05161), 282 test mazes, 498.58 vs Manhattan 180.79 nodes, 232 optimal paths, 50 suboptimal.
- Paired results: +317.78 nodes, bootstrap 95% CI +256.52 to +385.52, Wilcoxon p approximately 5.81e-43.
- PyTorch field diagnostics: 911,949 reachable cells; 55.32% overestimated, 48.44% adjacent inconsistency; these are descriptive/exploratory.
- Feature-ablation study: five **separately trained scikit-learn models** using 30,000 training cells per configuration. Do not present these as ablations of the trained PyTorch weights.
- Existing D2 Phase 4 fallback benchmarks use a different model and are maintained as **separate experimental evidence**.

## Safety check before committing

```bat
git branch --show-current
git status --short
git diff -- report/main.tex report/sections/phase3_training.tex report/sections/phase4_evaluation.tex
```

The last command should show no changes to the original D2 files.
