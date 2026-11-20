"""D3 paired failure analysis from original D2 results; does NOT overwrite D2 files.

Run from repository root:
  python scripts/failure_analysis.py --csv results/evaluation/test_per_maze.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.final_analysis import paired_comparison, subgroup_comparisons


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='results/evaluation/test_per_maze.csv')
    p.add_argument('--baseline', default='manhattan')
    p.add_argument('--output', default='results/analysis/d3')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(argv)
    df = pd.read_csv(a.csv)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    treatments = [v for v in df['heuristic'].dropna().unique() if v.startswith('neural_')]
    if not treatments:
        raise SystemExit('No neural heuristics found. First run the original D2 evaluation.')
    rows, groups = [], []
    for treatment in sorted(treatments):
        result = paired_comparison(df, a.baseline, treatment, seed=a.seed)
        rows.append(result)
        g = subgroup_comparisons(df, treatment, a.baseline)
        if not g.empty:
            g.insert(0, 'heuristic', treatment)
            groups.append(g)
    (output / 'failure_summary.json').write_text(json.dumps({'comparisons': rows}, indent=2), encoding='utf-8')
    if groups:
        pd.concat(groups, ignore_index=True).to_csv(output / 'failure_by_group.csv', index=False)
    report = pd.DataFrame([{k: v for k,v in row.items() if not isinstance(v, list)} for row in rows])
    report.to_csv(output / 'paired_comparisons.csv', index=False)
    print(report[['treatment','n_matched_mazes','mean_node_difference_treatment_minus_baseline',
                  'mean_expansion_ratio']].to_string(index=False))
    print(f'Results saved to {output}; original D2 evaluation files unchanged.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
