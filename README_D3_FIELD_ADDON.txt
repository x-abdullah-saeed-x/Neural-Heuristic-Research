ADDITIVE Deliverable 3 prediction-field diagnostics

Copy these two new files into your repository, preserving folder paths:
  scripts/analyze_prediction_fields.py
  tests/test_field_diagnostics_d3.py

Your corrected Deliverable 2 files should NOT be overwritten.

Run from project root in your existing .venv:
  python -m pytest -q tests/test_field_diagnostics_d3.py
  python scripts/analyze_prediction_fields.py --checkpoint models/neural_heuristic_seed42.pt --mazes 282 --search-results results/analysis/d3/full_test_raw/torch_search.json --output results/analysis/d3/full_test_raw/field_diagnostics
  type results\analysis\d3\full_test_raw\field_diagnostics\field_summary.json

Outputs saved only under results/analysis/d3/full_test_raw/field_diagnostics:
  field_per_maze.csv, field_by_group.csv, field_summary.json,
  prediction_mae_by_size.png, inconsistency_by_size.png,
  inconsistency_vs_expansion.png.

What is measured:
  Overestimation and prediction error on reachable, free cells only.
  Consistency violation on adjacent pairs of reachable free cells only;
  for an undirected unit-cost grid, violation means |h(u)-h(v)| > 1.
  Association (Spearman correlation) with the per-maze ratio of neural
  to Manhattan expanded nodes (not proof of causation).

Test mazes must be the same 282-maze held-out split as saved search JSON;
 mode must be raw with alpha=1.0. Checkpoint is your trained D2 model.
