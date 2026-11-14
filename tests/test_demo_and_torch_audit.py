"""Tests for the demo renderer and the PyTorch checkpoint audit."""

import os

import numpy as np
import pytest

from scripts.demo import ascii_maze, field_diagnostics, load_maze, torch_checkpoint_status
from scripts.run_torch_search import audit_checkpoint, load_mazes

CLEANED = os.path.join("data", "cleaned")
CHECKPOINT = os.path.join("models", "neural_heuristic_seed42.pt")
MANIFEST = os.path.join("results", "manifests", "split_manifest.json")


def _tiny_maze():
    grid = np.zeros((4, 4), dtype=np.uint8)
    grid[1, 1] = 1
    grid[1, 2] = 1
    return {
        "file_name": "tiny.npz",
        "local_index": 0,
        "grid": grid,
        "start": (0, 0),
        "goal": (3, 3),
        "h_star": np.zeros((4, 4)),
        "opt_cost": 6.0,
    }


def test_ascii_maze_marks_start_goal_walls_and_path():
    maze = _tiny_maze()
    text = ascii_maze(
        maze,
        path=[(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)],
        expanded=[(2, 0)],
        max_cells=10,
    )
    assert "S" in text and "G" in text and "#" in text
    assert "*" in text and "+" in text


def test_ascii_maze_truncates_large_grids():
    maze = _tiny_maze()
    maze["grid"] = np.zeros((200, 200), dtype=np.uint8)
    text = ascii_maze(maze, max_cells=20)
    assert "rows/columns omitted" in text


def test_field_diagnostics_matches_known_error():
    class _Field:
        def predictions(self):
            return np.array([[0.0, 1.0], [2.0, 3.0]])

    maze = {
        "grid": np.zeros((2, 2), dtype=np.uint8),
        "h_star": np.zeros((2, 2)),
    }
    diagnostics = field_diagnostics(_Field(), maze)
    assert diagnostics["field_mae_cells"] == pytest.approx(1.5)
    assert diagnostics["field_bias_cells"] == pytest.approx(1.5)
    assert diagnostics["field_overestimation_rate"] == pytest.approx(0.75)


def test_torch_audit_flags_committed_smoke_checkpoint():
    if not os.path.exists(CHECKPOINT):
        pytest.skip("torch checkpoint not present")
    pytest.importorskip("torch")
    audit = audit_checkpoint(CHECKPOINT)
    assert audit["under_trained"] is True
    assert audit["best_epoch"] <= 1
    assert audit["reasons"]
    assert "retrain" in audit["recommendation"].lower()


def test_demo_status_agrees_with_audit():
    if not os.path.exists(CHECKPOINT):
        pytest.skip("torch checkpoint not present")
    pytest.importorskip("torch")
    status = torch_checkpoint_status(CHECKPOINT)
    assert status["under_trained"] is True
    assert status["reasons"]


def test_load_mazes_respects_count():
    if not (os.path.exists(MANIFEST) and os.path.isdir(CLEANED)):
        pytest.skip("manifest or cleaned data not present")
    mazes = load_mazes(MANIFEST, CLEANED, count=2)
    assert len(mazes) == 2
    for maze in mazes:
        assert maze["grid"].ndim == 2
        assert len(maze["start"]) == 2
        assert len(maze["goal"]) == 2


def test_load_maze_matches_manifest_entry():
    if not os.path.isdir(CLEANED):
        pytest.skip("cleaned data not present")
    maze = load_maze("mazes_perlin_20_10.npz", 42, CLEANED)
    assert maze["grid"].shape == (20, 20)
    assert maze["start"] == (5, 0)
    assert maze["goal"] == (9, 15)
