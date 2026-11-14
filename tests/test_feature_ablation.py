"""Tests for the feature-group ablation harness."""

import numpy as np
import pytest

from scripts.run_feature_ablation import (
    SubsetNeuralHeuristic,
    feature_variants,
    subset_features,
)
from src.ablations import FEATURE_GROUPS


class _ZeroRegressor:
    def predict(self, features):
        return np.zeros(len(features), dtype=float)


def test_variant_names_cover_all_components():
    variants = feature_variants()
    assert set(variants) == {
        "all",
        "no_coordinates",
        "no_goal_offsets",
        "no_wall_window",
        "goal_offsets_only",
        "coordinates_only",
        "wall_window_only",
    }


def test_all_variant_uses_every_feature():
    assert feature_variants()["all"] == tuple(range(29))


def test_drop_variants_have_expected_dimensions():
    variants = feature_variants()
    assert len(variants["no_coordinates"]) == 27
    assert len(variants["no_goal_offsets"]) == 27
    assert len(variants["no_wall_window"]) == 4


def test_single_group_variants_are_exact():
    variants = feature_variants()
    assert variants["goal_offsets_only"] == FEATURE_GROUPS["goal_offsets"]
    assert variants["coordinates_only"] == FEATURE_GROUPS["coordinates"]
    assert variants["wall_window_only"] == FEATURE_GROUPS["wall_window"]


def test_subset_features_selects_columns():
    features = np.arange(2 * 29, dtype=float).reshape(2, 29)
    selected = subset_features(features, (2, 3))
    assert selected.shape == (2, 2)
    assert selected[0].tolist() == [2.0, 3.0]


def test_subset_heuristic_falls_back_on_walls():
    grid = np.zeros((3, 3), dtype=np.uint8)
    grid[0, 0] = 1
    heuristic = SubsetNeuralHeuristic(_ZeroRegressor(), grid, (2, 2), (2, 3))
    assert heuristic(0, 0) == pytest.approx(4.0)
    assert heuristic(1, 1) == pytest.approx(0.0)
