"""Focused tests of added D3 cell-field analysis; no real checkpoint required."""
import numpy as np
from scripts.analyze_prediction_fields import field_metrics, maze_metadata


def test_perfect_field_has_no_errors_or_inconsistency():
    g = np.zeros((2, 2), dtype=int)
    truth = np.array([[2, 1], [1, 0]], dtype=float)
    out = field_metrics(g, truth, truth)
    assert out['reachable_cells'] == 4
    assert out['mae_cells'] == 0
    assert out['adjacent_reachable_pairs'] == 4
    assert out['inconsistent_pairs'] == 0


def test_overestimated_and_inconsistent_field():
    g = np.zeros((2, 2), dtype=int)
    truth = np.array([[2, 1], [1, 0]], dtype=float)
    pred = np.array([[6, 1], [1, 0]], dtype=float)
    out = field_metrics(g, truth, pred)
    assert out['overestimated_cells'] == 1
    assert out['inconsistent_pairs'] == 2
    assert out['adjacent_reachable_pairs'] == 4


def test_unreachable_and_wall_cells_are_excluded():
    g = np.array([[0, 1], [0, 0]], dtype=int)
    truth = np.array([[2, -1], [-1, 0]], dtype=float)
    pred = np.array([[1, 123], [100, 0]], dtype=float)
    out = field_metrics(g, truth, pred)
    assert out['reachable_cells'] == 2
    assert out['adjacent_reachable_pairs'] == 0
    assert out['inconsistency_rate'] is None


def test_parse_maze_metadata():
    assert maze_metadata('mazes_perlin_80_30.npz') == {'generator': 'perlin', 'size': 80, 'density': 30}
