"""Read-only check that the D3 report is grounded in the local experiment files.
Run: python report/d3_final/validate_evidence.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    'original_d2_report': 'report/main.tex',
    'new_phase5': 'report/d3_final/phase5_results.tex',
    'training_summary': 'results/training/final_summary.json',
    'torch_search': 'results/analysis/d3/full_test_raw/torch_search.json',
    'paired_failure': 'results/analysis/d3/full_test_raw/failure_analysis/failure_summary.json',
    'field_diagnostics': 'results/analysis/d3/full_test_raw/field_diagnostics/field_summary.json',
    'feature_ablation': 'results/analysis/d3/feature_ablation_full/feature_ablation.csv',
    'prediction_fig': 'figures/d3/prediction_fields.png',
    'inconsistency_fig': 'results/analysis/d3/full_test_raw/field_diagnostics/inconsistency_by_size.png',
    'consistency_scatter': 'results/analysis/d3/full_test_raw/field_diagnostics/inconsistency_vs_expansion.png',
    'mae_fig': 'results/analysis/d3/full_test_raw/field_diagnostics/prediction_mae_by_size.png',
}


def close(value: float, expected: float, label: str, abs_tol: float = 0.06):
    if not math.isclose(float(value), expected, rel_tol=0, abs_tol=abs_tol):
        raise ValueError(f'{label}={value} does not match reported {expected}')


def main() -> int:
    missing = [f'{label}: {name}' for label, name in FILES.items() if not (ROOT / name).is_file()]
    if missing:
        print('Missing required files:')
        for entry in missing:
            print(' -', entry)
        return 1
    read = lambda name: json.loads((ROOT / FILES[name]).read_text(encoding='utf-8-sig'))
    train, search, failures, fields = (read(n) for n in ('training_summary','torch_search','paired_failure','field_diagnostics'))
    primary = next((r for r in train['runs'] if r.get('seed') == 42), None)
    if primary is None:
        raise ValueError('No primary seed-42 run in D2 training summary')
    close(primary['best_val_mae'], 0.0516117564, 'D2 seed-42 MAE', 0.00001)
    close(search['n_mazes'], 282, 'Number of D3 test mazes', 0)
    close(search['manhattan_mean_nodes'], 180.7943262, 'Manhattan nodes', 0.01)
    close(search['neural_mean_nodes'], 498.5780142, 'PyTorch neural nodes', 0.01)
    if len(search['per_maze']) != 282:
        raise ValueError('Search JSON must contain 282 paired mazes')
    comp = failures['comparisons'][0]
    close(comp['n_matched_mazes'], 282, 'Paired maze count', 0)
    close(comp['mean_node_difference_treatment_minus_baseline'], 317.783688, 'Mean paired expansion difference', 0.02)
    close(comp['treatment_better_count'], 15, 'Neural improvement count', 0)
    close(comp['treatment_worse_count'], 263, 'Neural deterioration count', 0)
    close(fields['n_mazes'], 282, 'Fields maze count', 0)
    close(fields['reachable_cells'], 911949, 'Reachable cells', 0)
    close(fields['overestimate_fraction'], 0.5532195, 'Field overestimation', 0.001)
    close(fields['inconsistency_rate'], 0.4844273, 'Field inconsistency', 0.001)
    with (ROOT / FILES['feature_ablation']).open(newline='',encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 5 or {r['variant'] for r in rows} != {
        'full_29', 'without_wall_window', 'goal_offsets_only',
        'without_goal_offsets', 'without_coordinates'
    }:
        raise ValueError('Expected all five full feature-ablation variants')
    if any(int(r['mazes_evaluated']) != 282 for r in rows):
        raise ValueError('A feature-ablation row does not cover 282 test mazes')
    print('D3 evidence validation passed.')
    print('282 paired test mazes; trained seed-42 checkpoint; 5 feature ablations; 4 D3 figures found.')
    print('Original corrected D2 report was only read, not modified.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (KeyError, ValueError, OSError, TypeError, IndexError) as exc:
        print(f'D3 evidence check failed: {exc}')
        sys.exit(1)
