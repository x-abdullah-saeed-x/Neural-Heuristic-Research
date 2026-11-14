"""Feature-group ablation for the neural heuristic (Deliverable 3).

Trains the same scikit-learn MLP used by the committed fallback models on
different subsets of the 29-cell feature vector, then reports both regression
fidelity and the resulting A* node expansions. This answers the Phase 5
question of which architectural component carries the signal.

Variants:
    all                 all 29 features
    no_coordinates      drop (x/N, y/N)
    no_goal_offsets     drop the goal-relative offsets
    no_wall_window      drop the 5x5 wall context
    goal_offsets_only   keep only the two goal offsets
    coordinates_only    keep only the two coordinates
    wall_window_only    keep only the 25 wall-window cells

Usage:
    python scripts/run_feature_ablation.py
    python scripts/run_feature_ablation.py --train-samples 60000 --search-mazes 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ablations import FEATURE_GROUPS  # noqa: E402
from src.features import N_FEATURES, features_for_grid  # noqa: E402
from src.metrics import regression_metrics  # noqa: E402
from src.search import astar, ManhattanHeuristic  # noqa: E402

ALL_INDICES = tuple(range(N_FEATURES))


def feature_variants() -> dict[str, tuple[int, ...]]:
    """Named feature-index subsets used in the ablation."""
    def others(dropped: str) -> tuple[int, ...]:
        removed = set(FEATURE_GROUPS[dropped])
        return tuple(index for index in ALL_INDICES if index not in removed)

    keep = lambda *groups: tuple(index for group in groups for index in FEATURE_GROUPS[group])

    return {
        "all": ALL_INDICES,
        "no_coordinates": others("coordinates"),
        "no_goal_offsets": others("goal_offsets"),
        "no_wall_window": others("wall_window"),
        "goal_offsets_only": keep("goal_offsets"),
        "coordinates_only": keep("coordinates"),
        "wall_window_only": keep("wall_window"),
    }


def subset_features(features: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Select feature columns by index."""
    return np.asarray(features)[:, list(indices)]


class SubsetNeuralHeuristic:
    """Heuristic backed by a regressor trained on a feature subset."""

    name = "neural_subset"

    def __init__(
        self,
        model: MLPRegressor,
        grid: np.ndarray,
        goal: tuple[int, int],
        indices: Sequence[int],
        mode: str = "raw",
        alpha: float = 1.0,
    ):
        self.grid = np.asarray(grid)
        self.goal = (int(goal[0]), int(goal[1]))
        self.mode = mode
        self.alpha = float(alpha)
        self.n = int(self.grid.shape[0])
        features = features_for_grid(self.grid, self.goal).reshape(-1, N_FEATURES)
        predictions = model.predict(subset_features(features, indices)).reshape(self.n, self.n)
        self.field = np.maximum(predictions.astype(float), 0.0) * self.n
        rows, cols = np.mgrid[0 : self.n, 0 : self.n]
        self.manhattan = np.abs(rows - self.goal[0]) + np.abs(cols - self.goal[1])

    def __call__(self, r: int, c: int) -> float:
        if not (0 <= r < self.n and 0 <= c < self.n) or self.grid[r, c] == 1:
            return float(self.manhattan[r, c])
        value = float(self.field[r, c])
        if self.mode == "clipped_manhattan":
            return min(value, float(self.manhattan[r, c]))
        return value


def load_arrays(
    train_path: str,
    test_path: str,
    train_samples: int,
    test_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and subsample the processed feature matrices."""
    train = np.load(train_path)
    test = np.load(test_path)
    rng = np.random.default_rng(seed)

    def subsample(X, y, count):
        if count <= 0 or count >= len(X):
            return X, y
        idx = rng.choice(len(X), size=count, replace=False)
        return X[idx], y[idx]

    X_train, y_train = subsample(train["X"], train["y"], train_samples)
    X_test, y_test = subsample(test["X"], test["y"], test_samples)
    return X_train, y_train, X_test, y_test


def _manifest_entries(manifest_path: str, split: str = "test") -> list[dict[str, Any]]:
    """Read either manifest layout used in the repository."""
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "splits" in manifest:
        entries = manifest["splits"][split]["entries"]
        return [
            {"file": entry.get("file", entry.get("file_name")), "local_index": entry["local_index"]}
            for entry in entries
        ]
    entries = manifest.get(split, [])
    return [
        {
            "file": entry.get("file", entry.get("file_name")),
            "local_index": entry["local_index"],
        }
        for entry in entries
    ]


def load_test_mazes(manifest_path: str, cleaned_dir: str) -> list[dict[str, Any]]:
    """Yield test mazes from a split manifest without depending on its layout."""
    mazes: list[dict[str, Any]] = []
    for entry in _manifest_entries(manifest_path, "test"):
        with np.load(os.path.join(cleaned_dir, entry["file"])) as data:
            index = int(entry["local_index"])
            mazes.append(
                {
                    "file_name": entry["file"],
                    "local_index": index,
                    "grid": data["grids"][index],
                    "start": tuple(int(v) for v in data["starts"][index]),
                    "goal": tuple(int(v) for v in data["goals"][index]),
                    "h_star": data["h_star"][index],
                    "opt_cost": float(data["opt_cost"][index]),
                }
            )
    return mazes


def _search_mazes(
    manifest_path: str,
    cleaned_dir: str,
    count: int,
    max_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    mazes = [
        maze
        for maze in load_test_mazes(manifest_path, cleaned_dir)
        if maze["grid"].shape[0] <= max_size
    ]
    if count <= 0 or count >= len(mazes):
        return mazes
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(mazes), size=count, replace=False)
    return [mazes[i] for i in sorted(indices)]


def _evaluate_search(
    model: MLPRegressor,
    indices: Sequence[int],
    mazes: Sequence[dict[str, Any]],
) -> dict[str, float]:
    expanded = []
    ratios = []
    for maze in mazes:
        heuristic = SubsetNeuralHeuristic(model, maze["grid"], maze["goal"], indices)
        result = astar(maze["grid"], maze["start"], maze["goal"], heuristic)
        expanded.append(result.nodes_expanded)
        if result.found and maze["opt_cost"] > 0:
            ratios.append(result.cost / maze["opt_cost"])
    return {
        "search_mazes": len(mazes),
        "mean_nodes_expanded": float(np.mean(expanded)) if expanded else float("nan"),
        "median_nodes_expanded": float(np.median(expanded)) if expanded else float("nan"),
        "mean_optimality_ratio": float(np.mean(ratios)) if ratios else float("nan"),
    }


def run(
    train_path: str = "data/processed/train.npz",
    test_path: str = "data/processed/test.npz",
    output_dir: str = "results/analysis",
    train_samples: int = 100000,
    test_samples: int = 60000,
    max_iter: int = 50,
    search_mazes: int = 24,
    max_size: int = 60,
    manifest_path: str = "results/manifests/split_manifest.json",
    cleaned_dir: str = "data/cleaned",
    seed: int = 42,
) -> pd.DataFrame:
    """Train every feature variant and persist metrics and search impact."""
    os.makedirs(output_dir, exist_ok=True)
    X_train, y_train, X_test, y_test = load_arrays(
        train_path, test_path, train_samples, test_samples, seed
    )
    mazes = _search_mazes(manifest_path, cleaned_dir, search_mazes, max_size, seed)

    baseline_nodes = None
    if mazes:
        manhattan_nodes = []
        for maze in mazes:
            result = astar(
                maze["grid"], maze["start"], maze["goal"], ManhattanHeuristic(maze["goal"])
            )
            manhattan_nodes.append(result.nodes_expanded)
        baseline_nodes = float(np.mean(manhattan_nodes))

    rows: list[dict[str, Any]] = []
    for name, indices in feature_variants().items():
        started = time.perf_counter()
        model = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            max_iter=max_iter,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=5,
            verbose=False,
        )
        model.fit(subset_features(X_train, indices), y_train)
        elapsed = time.perf_counter() - started

        predictions = model.predict(subset_features(X_test, indices))
        metrics = regression_metrics(y_test, predictions, grid_size=60.0)
        search = _evaluate_search(model, indices, mazes)

        row = {
            "variant": name,
            "n_features": len(indices),
            "r2": metrics.get("r2", float("nan")),
            "mae": metrics.get("mae", float("nan")),
            "mae_cells_n60": metrics.get("mae_cells", float("nan")),
            "rmse": metrics.get("rmse", float("nan")),
            "pearson": metrics.get("pearson", float("nan")),
            "mean_signed_error_cells": metrics.get("mean_signed_error_cells", float("nan")),
            "train_seconds": elapsed,
            "iterations": int(model.n_iter_),
        }
        row.update(search)
        if baseline_nodes and search["mean_nodes_expanded"] == search["mean_nodes_expanded"]:
            row["manhattan_nodes"] = baseline_nodes
            row["nodes_ratio_vs_manhattan"] = search["mean_nodes_expanded"] / baseline_nodes
        rows.append(row)
        print(
            f"{name:20s} features={len(indices):2d} "
            f"R2={row['r2']:.4f} MAE={row['mae']:.4f} "
            f"nodes={row['mean_nodes_expanded']:.1f} ({elapsed:.1f}s)"
        )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_dir, "feature_ablation.csv"), index=False)
    payload = {
        "seed": seed,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "search_mazes": int(len(mazes)),
        "max_iter": max_iter,
        "manhattan_mean_nodes": baseline_nodes,
        "variants": table.to_dict(orient="records"),
    }
    with open(os.path.join(output_dir, "feature_ablation.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)
    return table


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="data/processed/train.npz")
    parser.add_argument("--test-path", default="data/processed/test.npz")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--train-samples", type=int, default=100000)
    parser.add_argument("--test-samples", type=int, default=60000)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--search-mazes", type=int, default=24)
    parser.add_argument("--max-size", type=int, default=60)
    parser.add_argument("--manifest", default="results/manifests/split_manifest.json")
    parser.add_argument("--cleaned-dir", default="data/cleaned")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    table = run(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        max_iter=args.max_iter,
        search_mazes=args.search_mazes,
        max_size=args.max_size,
        manifest_path=args.manifest,
        cleaned_dir=args.cleaned_dir,
        seed=args.seed,
    )
    best = table.sort_values("r2", ascending=False).iloc[0]
    print(f"best variant: {best['variant']} (R2={best['r2']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
