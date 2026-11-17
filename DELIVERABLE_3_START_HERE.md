# Deliverable 3 — Rebuild from the corrected Deliverable 2

This ZIP is an **additive starter implementation**, created after reviewing the `main` and `deliverable-3` branches of `x-abdullah-saeed-x/Neural-Heuristic-Research` on 17 November 2026.

It follows the newer branch's **Phase 5 research workflow**—paired search comparisons, feature ablations, failure diagnostics, reproducibility checks, qualitative prediction examples, a demo, and final-report preparation—but is **not a copy of its generated results**. All numbers must come from your corrected D2 data, model checkpoints, and new D3 runs.

## Safety

- Contains **only new paths** that did not exist in `main` at the time this starter was prepared.
- Does not overwrite `README.md`, `.gitignore`, `src/search.py`, `report/main.tex`, any D2 training/evaluation result, existing `report/sections/phase5_results.tex`, or existing models.
- Does not contain the other D3 branch's older evaluation metrics, model binaries, images, or finished PDF.
- The report section is a **draft** stored as `report/sections/phase5_d3_draft.tex`; it is not automatically included in the D2 report.

## IMPORTANT: abandon the unfinished old merge before extracting

Your terminal is currently in a partially resolved merge on `integration/d3-on-final-d2`. From inside your repository, run:

```cmd
git merge --abort
git switch main
git status
```

You should see a clean `main`. Create a **different** new branch:

```cmd
git switch -c d3/rebuild-from-corrected-d2
```

Now extract the *contents* of this ZIP directly into your repository root, retaining the `src/`, `scripts/`, `docs/`, `tests/`, and `report/` paths. No existing D2 file should be overwritten. Inspect with `git status` before adding/committing.

If Git reports `There is no merge to abort`, skip that command only when `git status` shows no merge in progress. Do not use `git reset --hard` or force-push.

## Incremental run order

1. **D3 search failure analysis from D2's already-recorded benchmark**:
   ```cmd
   python scripts/failure_analysis.py
   ```
   Output is saved under `results/analysis/d3/`. This reads the existing D2 results but never rewrites them.
2. **Feature ablation using the original training/test splits** (requires your local generated data):
   ```cmd
   python scripts/run_feature_ablation.py --samples 30000 --mazes 24
   python scripts/make_ablation_summary.py
   ```
   These are small initial runs for checking the pipeline, **not** full-report headline numbers. Choose and record a final experimental budget later and rerun.
3. **Verify real trained PyTorch checkpoint metadata** (retrieve D2 checkpoint from the approved Hugging Face location if needed):
   ```cmd
   python scripts/run_torch_search.py --checkpoint models/neural_heuristic_seed42.pt --audit-only
   ```
   Then test a subset:
   ```cmd
   python scripts/run_torch_search.py --checkpoint models/neural_heuristic_seed42.pt --mazes 24
   ```
4. **Generate fresh qualitative figure and demonstration**:
   ```cmd
   python scripts/sample_predictions.py --backend torch --model models/neural_heuristic_seed42.pt
   python scripts/demo.py --index 0 --backend torch --model models/neural_heuristic_seed42.pt
   ```
   To begin without checkpoint weights, `python scripts/demo.py --index 0 --backend none` runs only classical baselines.
5. **Create new LaTeX D3 tables**:
   ```cmd
   python scripts/make_phase5_report_assets.py
   ```
   The generated tables go to `report/generated/d3/`; D2 generated tables are not changed.
6. **Run new tests**:
   ```cmd
   python -m pytest -q tests/test_final_analysis.py tests/test_feature_ablation.py tests/test_demo_and_torch_audit.py
   ```

## Before final submission

Read `docs/DELIVERABLE_3_CHECKLIST.md`, rerun with the final agreed sample counts on the same held-out maze split, preserve seeds and weights checksums, then integrate **only the verified new D3 evidence** into the final report. Do not claim PyTorch was only a one-epoch smoke test unless the actual checkpoint under evaluation proves that. Do not claim a neural heuristic wins if measurements do not show it.

## Model/back-end distinction

The original main D2 report uses the scikit-learn fallback for its recorded A* benchmark, while also having later separate multi-seed PyTorch training. Compare each backend explicitly and never silently mix numbers from different weights, sample sizes, or runs. To reproduce scikit-learn results, retrieve the correct fallback weight file or retrain it with recorded settings.
