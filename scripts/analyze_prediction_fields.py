"""Deliverable 3: inspect true vs PyTorch-predicted h* on held-out D2 mazes.

Does not retrain models, change D2 artifacts, or overwrite previously created D3 files.

python scripts/analyze_prediction_fields.py --checkpoint models/neural_heuristic_seed42.pt \
    --mazes 282 --search-results results/analysis/d3/full_test_raw/torch_search.json \
    --output results/analysis/d3/full_test_raw/field_diagnostics
"""
from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def _json_number(v):
    v = float(v)
    return v if np.isfinite(v) else None


def valid_cell_mask(grid, truth, pred):
    return ((np.asarray(grid) == 0) & np.isfinite(truth) & (truth >= 0) &
            np.isfinite(pred))


def field_metrics(grid, truth, prediction):
    """Count reachable-cell error and unit-cost consistency violations."""
    grid = np.asarray(grid)
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if grid.ndim != 2 or grid.shape != truth.shape or grid.shape != prediction.shape:
        raise ValueError('grid, truth, prediction must have matching 2D shapes')
    valid = valid_cell_mask(grid, truth, prediction)
    if not valid.any():
        raise ValueError('No reachable/free/finite cells in this maze')
    err = (prediction - truth)[valid]
    violating = 0
    tested = 0
    for left, right, mask_l, mask_r in (
        (prediction[:-1, :], prediction[1:, :], valid[:-1, :], valid[1:, :]),
        (prediction[:, :-1], prediction[:, 1:], valid[:, :-1], valid[:, 1:]),
    ):
        pairs = mask_l & mask_r
        tested += int(pairs.sum())
        violating += int(np.sum(pairs & (np.abs(left - right) > 1.0 + 1e-6)))
    return {
        'reachable_cells': int(valid.sum()),
        'overestimated_cells': int(np.sum(err > 1e-8)),
        'absolute_error_sum_cells': float(np.abs(err).sum()),
        'signed_error_sum_cells': float(err.sum()),
        'squared_error_sum_cells': float(np.square(err).sum()),
        'mae_cells': float(np.abs(err).mean()),
        'rmse_cells': float(np.sqrt(np.square(err).mean())),
        'mean_signed_error_cells': float(err.mean()),
        'overestimate_fraction': float(np.mean(err > 1e-8)),
        'adjacent_reachable_pairs': tested,
        'inconsistent_pairs': violating,
        'inconsistency_rate': violating / tested if tested else None,
    }


def maze_metadata(filename):
    m = re.match(r'^mazes_([^_]+)_(\d+)_(\d+)\.npz$', Path(filename).name)
    return {'generator': m.group(1), 'size': int(m.group(2)),
            'density': int(m.group(3))} if m else {
                'generator': 'unknown', 'size': None, 'density': None}


def _weighted_summary(frame):
    cells = int(frame['reachable_cells'].sum())
    pairs = int(frame['adjacent_reachable_pairs'].sum())
    result = {
        'n_mazes': int(len(frame)),
        'reachable_cells': cells,
        'adjacent_reachable_pairs': pairs,
        'inconsistent_pairs': int(frame['inconsistent_pairs'].sum()),
        'inconsistency_rate': float(frame['inconsistent_pairs'].sum() / pairs) if pairs else None,
        'overestimated_cells': int(frame['overestimated_cells'].sum()),
        'overestimate_fraction': float(frame['overestimated_cells'].sum() / cells),
        'mae_cells': float(frame['absolute_error_sum_cells'].sum() / cells),
        'rmse_cells': float(np.sqrt(frame['squared_error_sum_cells'].sum() / cells)),
        'mean_signed_error_cells': float(frame['signed_error_sum_cells'].sum() / cells),
        'mean_per_maze_mae_normalized': float(frame['mae_normalized'].mean()),
    }
    if 'neural_nodes' in frame.columns:
        result.update({
            'manhattan_mean_nodes': float(frame['manhattan_nodes'].mean()),
            'neural_mean_nodes': float(frame['neural_nodes'].mean()),
            'neural_better_maze_count': int((frame['neural_nodes'] < frame['manhattan_nodes']).sum()),
            'mean_optimality_ratio': _json_number(frame['optimality_ratio'].mean()),
        })
    return result


def _save_plots(df, dest):
    by_size = df.groupby('size', dropna=True).agg(
        mean_mae_cells=('mae_cells', 'mean'),
        mean_inconsistency=('inconsistency_rate', 'mean'),
        mean_overestimate=('overestimate_fraction', 'mean'),
    ).reset_index().sort_values('size')
    if not by_size.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(by_size['size'], by_size['mean_mae_cells'], marker='o')
        ax.set(xlabel='Maze side length (cells)', ylabel='Mean absolute error (grid cells)',
               title='Trained PyTorch prediction error by maze size')
        fig.tight_layout()
        fig.savefig(dest / 'prediction_mae_by_size.png', dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(by_size['size'], by_size['mean_inconsistency'] * 100, marker='o')
        ax.set(xlabel='Maze side length (cells)', ylabel='Mean inconsistent neighbor pairs (%)',
               title='Prediction-field inconsistency by maze size')
        fig.tight_layout()
        fig.savefig(dest / 'inconsistency_by_size.png', dpi=160)
        plt.close(fig)
    if 'neural_nodes' in df.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ratios = df['neural_nodes'] / df['manhattan_nodes'].clip(lower=1)
        ax.scatter(df['inconsistency_rate'] * 100, ratios, s=18, alpha=0.55)
        ax.set(xlabel='Inconsistent neighboring pairs (%)',
               ylabel='Neural / Manhattan node expansions per maze',
               title='Is prediction-field inconsistency associated with extra search?')
        fig.tight_layout()
        fig.savefig(dest / 'inconsistency_vs_expansion.png', dpi=160)
        plt.close(fig)


def main(argv=None):
    from src.evaluate import iter_split_mazes
    from src.heuristics import NeuralHeuristic

    parser = argparse.ArgumentParser(description='Evaluate PyTorch heuristic field bias and consistency on held-out D2 mazes.')
    parser.add_argument('--checkpoint', default='models/neural_heuristic_seed42.pt')
    parser.add_argument('--mazes', type=int, default=282)
    parser.add_argument('--search-results', default='results/analysis/d3/full_test_raw/torch_search.json')
    parser.add_argument('--output', default='results/analysis/d3/full_test_raw/field_diagnostics')
    parser.add_argument('--manifest', default='data/processed/split_manifest.json')
    parser.add_argument('--cleaned-dir', default='data/cleaned')
    args = parser.parse_args(argv)
    if args.mazes < 1:
        parser.error('--mazes must be at least 1')
    if not Path(args.checkpoint).is_file():
        parser.error('Checkpoint does not exist: ' + args.checkpoint)
    search_data = None
    search_path = Path(args.search_results)
    if search_path.is_file():
        search_data = json.loads(search_path.read_text(encoding='utf-8'))
        if (search_data.get('backend') != 'torch' or search_data.get('mode') != 'raw' or
            float(search_data.get('alpha', -1)) != 1.0 or
            search_data.get('checkpoint_name') != Path(args.checkpoint).name or
            len(search_data.get('per_maze', [])) != args.mazes):
            raise ValueError('Search JSON must be raw alpha=1.0 for same checkpoint and maze count')
        search_by_id = {str(r['maze_id']): r for r in search_data['per_maze']}
        if len(search_by_id) != args.mazes:
            raise ValueError('Duplicate maze IDs in the search JSON')
    else:
        print('Search JSON unavailable; creating prediction diagnostics without search comparisons.')
        search_by_id = None

    items = list(islice(iter_split_mazes('test', cleaned_dir=args.cleaned_dir,
                                        manifest_path=args.manifest), args.mazes))
    if len(items) != args.mazes:
        raise ValueError(f'Expected {args.mazes} test mazes, found {len(items)}')
    rows = []
    for i, m in enumerate(items, start=1):
        neural = NeuralHeuristic(args.checkpoint, m['grid'], m['goal'], mode='raw')
        values = neural.predictions()  # predicted h* in raw grid-cell units
        grid = np.asarray(m['grid'])
        truth = np.asarray(m['h_star'], dtype=float)
        diag = field_metrics(grid, truth, values)
        data = {'maze_id': str(m['maze_id']), 'file_name': m['file_name'],
                'local_index': int(m['local_index']), **maze_metadata(m['file_name']),
                **diag,
                'mae_normalized': diag['mae_cells'] / grid.shape[0]}
        if search_by_id is not None:
            prior = search_by_id.get(data['maze_id'])
            if prior is None:
                raise ValueError(f'Missing search measurement for {data["maze_id"]}')
            data.update({k: prior[k] for k in ('neural_nodes', 'manhattan_nodes',
                                               'neural_cost', 'optimal_cost', 'optimality_ratio')})
        rows.append(data)
        if i % 25 == 0 or i == len(items):
            print(f'Analyzed {i}/{len(items)} mazes', flush=True)
    df = pd.DataFrame(rows)
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / 'field_per_maze.csv', index=False)
    groups = []
    for dimension in ['generator', 'size', 'density']:
        for label, subset in df.groupby(dimension, dropna=False):
            groups.append({'dimension': dimension, 'value': str(label), **_weighted_summary(subset)})
    pd.DataFrame(groups).to_csv(dest / 'field_by_group.csv', index=False)
    summary = _weighted_summary(df)
    summary['checkpoint_name'] = Path(args.checkpoint).name
    summary['source_search_results'] = str(search_path) if search_by_id is not None else None
    if 'neural_nodes' in df.columns:
        ratios = df['neural_nodes'] / df['manhattan_nodes'].clip(lower=1)
        for col in ('inconsistency_rate', 'overestimate_fraction', 'mae_cells'):
            result = spearmanr(df[col], ratios, nan_policy='omit')
            summary[f'spearman_{col}_vs_node_ratio'] = _json_number(result.statistic)
            summary[f'spearman_{col}_pvalue'] = _json_number(result.pvalue)
    (dest / 'field_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    _save_plots(df, dest)
    print(f"Completed {len(df)} mazes, {summary['reachable_cells']} reachable cells")
    print(f"Overestimation: {summary['overestimate_fraction']:.1%}; "
          f"adjacent-cell inconsistency: {summary['inconsistency_rate']:.1%}; "
          f"MAE: {summary['mae_cells']:.3f} cells")
    print(f'New D3 diagnostics saved under {dest}. Existing D2 files untouched.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
