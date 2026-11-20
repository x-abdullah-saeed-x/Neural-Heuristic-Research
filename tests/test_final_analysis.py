import numpy as np
import pandas as pd
import pytest
from src.final_analysis import (bootstrap_mean_ci, paired_comparison,
                                consistency_violation_rate, overestimation_summary)


def test_bootstrap_constant():
    assert bootstrap_mean_ci([3, 3, 3], n_boot=50) == (3.0, 3.0)


def test_paired_search_counts_and_order():
    data = pd.DataFrame([
        {'maze_id': 1, 'heuristic': 'manhattan', 'nodes_expanded': 10},
        {'maze_id': 2, 'heuristic': 'manhattan', 'nodes_expanded': 20},
        {'maze_id': 2, 'heuristic': 'neural_alpha_1.0', 'nodes_expanded': 25},
        {'maze_id': 1, 'heuristic': 'neural_alpha_1.0', 'nodes_expanded': 8},
    ])
    result = paired_comparison(data, seed=42)
    assert result['n_matched_mazes'] == 2
    assert result['treatment_better_count'] == 1
    assert result['treatment_worse_count'] == 1
    assert result['mean_node_difference_treatment_minus_baseline'] == 1.5


def test_duplicate_records_are_rejected():
    df = pd.DataFrame([{'maze_id': 1, 'heuristic': h, 'nodes_expanded': 10}
                       for h in ['manhattan','manhattan','neural_alpha_1.0']])
    with pytest.raises(ValueError):
        paired_comparison(df)


def test_consistency_violation():
    grid = np.zeros((2, 2), dtype=int)
    pred = np.array([[0.0, 3.0], [0.0, 0.0]])
    result = consistency_violation_rate(grid, pred)
    assert result['violating_pairs'] == 2
    assert result['adjacent_pairs'] == 4


def test_consistency_ignore_walls():
    grid = np.array([[0, 1], [0, 0]])
    pred = np.array([[0., 100.], [0., 0.]])
    assert consistency_violation_rate(grid, pred)['violating_pairs'] == 0


def test_overestimate():
    grid = np.zeros((2, 2), dtype=int)
    true = np.array([[0., 1.], [2., 3.]])
    pred = np.array([[0., 2.], [1., 3.]])
    s = overestimation_summary(grid, true, pred)
    assert s['cells'] == 4
    assert s['overestimate_fraction'] == 0.25
    assert s['mae_cells'] == 0.5
