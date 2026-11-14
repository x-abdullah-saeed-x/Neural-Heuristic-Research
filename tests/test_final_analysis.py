"""Unit tests for the Deliverable 3 analysis helpers."""

import numpy as np
import pandas as pd
import pytest

from src.final_analysis import (
    bootstrap_mean_ci,
    consistency_violation_rate,
    dead_end_mask,
    manhattan_lower_bound,
    paired_comparison,
    wall_neighbour_count,
)


def test_bootstrap_ci_of_constant_sample_is_degenerate():
    low, high = bootstrap_mean_ci(np.full(20, 7.0), n_boot=200)
    assert low == pytest.approx(7.0)
    assert high == pytest.approx(7.0)


def test_bootstrap_ci_of_empty_sample_is_nan():
    low, high = bootstrap_mean_ci(np.array([]))
    assert np.isnan(low) and np.isnan(high)


def test_consistency_violation_rate_detects_large_jump():
    grid = np.zeros((2, 2), dtype=np.uint8)
    smooth = np.zeros((2, 2), dtype=float)
    jumpy = np.array([[0.0, 5.0], [0.0, 0.0]])
    assert consistency_violation_rate(smooth, grid) == 0.0
    assert consistency_violation_rate(jumpy, grid) == pytest.approx(0.5)


def test_consistency_violation_rate_ignores_wall_pairs():
    grid = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    field = np.array([[0.0, 9.0], [0.0, 0.0]])
    rate = consistency_violation_rate(field, grid)
    assert 0.0 <= rate <= 1.0


def test_dead_end_mask_marks_corridor_ends_only():
    grid = np.zeros((1, 3), dtype=np.uint8)
    mask = dead_end_mask(grid)
    assert mask.tolist() == [[True, False, True]]


def test_wall_neighbour_count_counts_four_neighbours():
    grid = np.zeros((3, 3), dtype=np.uint8)
    grid[1, 1] = 1
    counts = wall_neighbour_count(grid)
    assert counts[0, 1] == 1
    assert counts[1, 0] == 1
    assert counts[0, 0] == 0
    assert counts[1, 1] == 0
    assert counts[2, 1] == 1


def test_manhattan_lower_bound():
    assert manhattan_lower_bound((0, 0), (2, 3)) == 5
    assert manhattan_lower_bound((4, 4), (4, 4)) == 0


def _comparison_frame(treatment_values):
    rows = []
    for index, treatment in enumerate(treatment_values):
        rows.append(
            {
                "file_name": "maze.npz",
                "local_index": index,
                "heuristic": "manhattan",
                "nodes_expanded": 10,
            }
        )
        rows.append(
            {
                "file_name": "maze.npz",
                "local_index": index,
                "heuristic": "neural_alpha_1.0",
                "nodes_expanded": treatment,
            }
        )
    return pd.DataFrame(rows)


def test_paired_comparison_detects_worse_treatment():
    frame = _comparison_frame([20, 30, 40, 50])
    result = paired_comparison(frame, n_boot=200)
    assert result["n_mazes"] == 4
    assert result["treatment_losses"] == 4
    assert result["treatment_wins"] == 0
    assert result["mean_difference"] == pytest.approx(25.0)
    assert result["relative_change"] == pytest.approx(2.5)
    assert result["mean_difference_ci_low"] <= result["mean_difference"]
    assert result["mean_difference_ci_high"] >= result["mean_difference"]


def test_paired_comparison_reports_ties():
    frame = _comparison_frame([10, 10, 10])
    result = paired_comparison(frame, n_boot=100)
    assert result["ties"] == 3
    assert result["wilcoxon_p_value"] == 1.0
    assert result["win_rate"] == 0.0
