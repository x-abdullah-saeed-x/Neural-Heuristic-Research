# Deliverable 3 — viva preparation template

These are **questions and verification prompts**, not prewritten conclusions.

1. **What changed from D2 to D3?** D2 established models and benchmarks. D3 evaluates *why* search behaves as observed, which model components matter, and whether results reproduce.
2. **How are comparisons fair?** Use identical held-out mazes, the same search algorithm and documented tie-breaking, and record the model backend, weight file, seed and settings.
3. **Why can good regression be bad for search?** Small average error does not ensure each local heuristic value yields useful search order. Compare node expansions, overestimation, consistency and optimality with actual output.
4. **How are feature groups tested?** Train on the same D2 train split with the same budget, changing only the selected feature columns, and compare against Manhattan on the same mazes.
5. **Was the PyTorch model fully trained?** Examine the exact real checkpoint and its D2 run summary. Do **not** copy a smoke-test description from an older branch.
6. **Did the hypothesis hold?** Read the fresh paired D3 output; disclose a negative result if the learned heuristic is weaker.
7. **What can be reproduced?** Show committed scripts/configs, fixed split, exact model checksums, metrics and generated tables.

Artifacts to demonstrate:

- `results/analysis/d3/paired_comparisons.csv`
- `results/analysis/d3/feature_ablation.csv`
- `results/analysis/d3/torch_checkpoint_audit.json`
- `figures/d3/prediction_fields.png`
- `results/demo/d3/demo.json`
- `report/generated/d3/paired_table.tex`
