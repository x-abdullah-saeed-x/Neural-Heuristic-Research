# Final D3 LaTeX manuscript

**Source is deliberately separated from corrected D2.** `report/main.tex` and all original D2 sections are unchanged.

Generate the complete report from repo root:

```bat
python report/d3_final/build_report.py
```

This copies the corrected D2 report into `report/d3_final/main.tex`, updates the summary to reflect the actual D3 PyTorch experiment, and inserts `report/d3_final/phase5_results.tex` before the collaboration section. No other D2 file is modified. The build adapts LaTeX search paths so the original D2 generated tables and figures remain discoverable.

Check that the local evidence agrees with the published values:

```bat
python report/d3_final/validate_evidence.py
```

Compile twice from the repository root with an installed LaTeX distribution:

```bat
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report/d3_final report/d3_final/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report/d3_final report/d3_final/main.tex
```

Result: `report/d3_final/main.pdf`. This output has not been compiled inside ChatGPT, because your local D2 figures/tables and D3 experiment figures are not available together here; check the local compilation before submitting.

The existing generated tables (`report/generated/d3/paired_table.tex` and `feature_table.tex`) remain available independently; the new manuscript uses explicitly authored tables so its scientific numbers have been checked against the terminal results shared during D3 experiments.
