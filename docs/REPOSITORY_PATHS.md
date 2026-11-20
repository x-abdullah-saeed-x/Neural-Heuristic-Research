# Paths in this ZIP

The files below are placed at their actual repository-relative locations, not inside a second project directory:

```text
Neural-Heuristic-Research/
├── DELIVERABLE_3_START_HERE.md
├── src/
│   └── final_analysis.py               # New statistical/field-analysis helpers
├── scripts/
│   ├── failure_analysis.py             # Paired search evaluation based on D2 CSV
│   ├── run_feature_ablation.py         # Retrains feature groups on D2 split
│   ├── make_ablation_summary.py        # Summarises new D3 ablations
│   ├── run_torch_search.py             # Audits actual D2 weights; optional search
│   ├── demo.py                         # Test-maze demonstration
│   ├── sample_predictions.py           # Prediction field figures
│   └── make_phase5_report_assets.py    # New LaTeX tables; never overwrites D2
├── tests/
│   ├── test_final_analysis.py
│   ├── test_feature_ablation.py
│   └── test_demo_and_torch_audit.py
├── report/
│   └── sections/
│       └── phase5_d3_draft.tex         # Draft only, not wired to D2 main.tex
└── docs/
    ├── DELIVERABLE_3_CHECKLIST.md
    ├── VIVA_GUIDE_D3.md
    └── REPOSITORY_PATHS.md
```

No D2 paths are replaced; existing datasets, trained weights and output metrics are intentionally not included. Outputs will be generated later in `results/analysis/d3/`, `results/demo/d3/`, `figures/d3/`, and `report/generated/d3/`.
