"""Deliverable 3 analysis helpers. Additive: existing D2 source files are unchanged."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_mean_ci(values: Sequence[float], confidence: float = 0.95,
                      n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Percentile CI for the mean paired difference, using a deterministic RNG."""
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return float('nan'), float('nan')
    if len(x) == 1:
        return float(x[0]), float(x[0])
    if not (0 < confidence < 1):
        raise ValueError('confidence must be in (0, 1)')
    if n_boot < 1:
        raise ValueError('n_boot must be positive')
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = float(np.mean(x[rng.integers(0, len(x), size=len(x))]))
    a = (1 - confidence) / 2
    return tuple(float(v) for v in np.quantile(means, [a, 1 - a]))


def aligned_heuristics(df: pd.DataFrame, baseline: str, treatment: str) -> pd.DataFrame:
    """Pair exactly the same maze runs; reject duplicates instead of inflating sample size."""
    required = {'heuristic', 'nodes_expanded'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(df.columns))}')
    keys = ['file_name', 'local_index'] if {'file_name', 'local_index'}.issubset(df.columns) else ['maze_id']
    if not set(keys).issubset(df.columns):
        raise ValueError('Need file_name/local_index or maze_id columns to pair mazes')
    measures = [c for c in ('nodes_expanded', 'runtime_ms', 'optimality_ratio') if c in df.columns]
    a = df.loc[df['heuristic'] == baseline, keys + measures]
    b = df.loc[df['heuristic'] == treatment, keys + measures]
    if a.empty or b.empty:
        raise ValueError(f'Cannot pair {baseline!r} with {treatment!r}; one is absent')
    paired = a.merge(b, on=keys, suffixes=('_baseline', '_treatment'), validate='one_to_one')
    if paired.empty:
        raise ValueError('No identical mazes found for both heuristics')
    return paired.sort_values(keys).reset_index(drop=True)


def paired_comparison(df: pd.DataFrame, baseline: str = 'manhattan',
                      treatment: str = 'neural_alpha_1.0', seed: int = 42) -> dict:
    paired = aligned_heuristics(df, baseline, treatment)
    b = paired['nodes_expanded_baseline'].to_numpy(dtype=float)
    t = paired['nodes_expanded_treatment'].to_numpy(dtype=float)
    diff = t - b
    ratio = t / np.where(b > 0, b, np.nan)
    lo, hi = bootstrap_mean_ci(diff, seed=seed)
    nonzero = diff[np.abs(diff) > 0]
    p_value = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
    answer = {
        'baseline': baseline, 'treatment': treatment,
        'n_matched_mazes': int(len(paired)),
        'baseline_mean_nodes': float(np.mean(b)),
        'treatment_mean_nodes': float(np.mean(t)),
        'mean_node_difference_treatment_minus_baseline': float(np.mean(diff)),
        'mean_node_difference_95pct_bootstrap_ci': [lo, hi],
        'mean_expansion_ratio': float(np.nanmean(ratio)),
        'treatment_better_count': int(np.sum(t < b)),
        'equal_count': int(np.sum(t == b)),
        'treatment_worse_count': int(np.sum(t > b)),
        'wilcoxon_p_value': p_value,
    }
    for col in ('runtime_ms', 'optimality_ratio'):
        if f'{col}_baseline' in paired:
            answer[f'baseline_mean_{col}'] = float(paired[f'{col}_baseline'].mean())
            answer[f'treatment_mean_{col}'] = float(paired[f'{col}_treatment'].mean())
    return answer


def subgroup_comparisons(df: pd.DataFrame, treatment: str,
                         baseline: str = 'manhattan') -> pd.DataFrame:
    """Compare search expansions across generator, size and density when available."""
    keys = ['file_name', 'local_index'] if {'file_name', 'local_index'}.issubset(df.columns) else ['maze_id']
    cols = [c for c in ('generator', 'size', 'density') if c in df.columns]
    merged = aligned_heuristics(df, baseline, treatment)
    if cols:
        metadata = df.loc[df['heuristic'] == baseline, keys + cols].drop_duplicates(keys)
        merged = merged.merge(metadata, on=keys, validate='one_to_one')
    merged['ratio'] = merged['nodes_expanded_treatment'] / merged['nodes_expanded_baseline'].replace(0, np.nan)
    rows = []
    for col in cols:
        for value, group in merged.groupby(col, dropna=False):
            rows.append({'dimension': col, 'value': str(value), 'mazes': len(group),
                         'baseline_mean_nodes': group['nodes_expanded_baseline'].mean(),
                         'treatment_mean_nodes': group['nodes_expanded_treatment'].mean(),
                         'mean_expansion_ratio': group['ratio'].mean()})
    return pd.DataFrame(rows, columns=['dimension','value','mazes','baseline_mean_nodes',
                                       'treatment_mean_nodes','mean_expansion_ratio'])


def consistency_violation_rate(grid: np.ndarray, predicted_costs: np.ndarray,
                               tolerance: float = 1e-6) -> dict:
    """Measure adjacent free-cell pairs where |h(u)-h(v)| > 1 for unit-cost grids."""
    grid = np.asarray(grid)
    pred = np.asarray(predicted_costs, dtype=float)
    if grid.ndim != 2 or pred.shape != grid.shape:
        raise ValueError('grid and predicted_costs must be equal-size 2D arrays')
    n_violate, n_pairs = 0, 0
    for a, b, fa, fb in [
        (pred[:-1, :], pred[1:, :], grid[:-1, :] == 0, grid[1:, :] == 0),
        (pred[:, :-1], pred[:, 1:], grid[:, :-1] == 0, grid[:, 1:] == 0),
    ]:
        valid = fa & fb & np.isfinite(a) & np.isfinite(b)
        n_pairs += int(np.sum(valid))
        n_violate += int(np.sum(valid & (np.abs(a - b) > 1.0 + tolerance)))
    return {'adjacent_pairs': n_pairs, 'violating_pairs': n_violate,
            'violation_rate': float(n_violate / n_pairs) if n_pairs else float('nan')}


def overestimation_summary(grid: np.ndarray, true_costs: np.ndarray,
                           predicted_costs: np.ndarray) -> dict:
    """Summarise predicted minus true cell cost on reachable, free cells."""
    grid = np.asarray(grid)
    true = np.asarray(true_costs, dtype=float)
    pred = np.asarray(predicted_costs, dtype=float)
    if grid.shape != true.shape or grid.shape != pred.shape:
        raise ValueError('grid, true_costs and predictions must have identical shapes')
    valid = (grid == 0) & np.isfinite(true) & (true >= 0) & np.isfinite(pred)
    residual = (pred - true)[valid]
    if not residual.size:
        return {'cells': 0, 'overestimate_fraction': float('nan'), 'mean_signed_error_cells': float('nan')}
    return {'cells': int(residual.size),
            'overestimate_fraction': float(np.mean(residual > 1e-8)),
            'mean_signed_error_cells': float(np.mean(residual)),
            'mae_cells': float(np.mean(np.abs(residual)))}
