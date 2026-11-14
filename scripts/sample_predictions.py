"""Qualitative sample predictions for Deliverable 3.

Selects a good, a typical and a bad test maze by the neural/Manhattan node
ratio, then draws the actual A* trace (expanded cells and final path) for
Manhattan, raw neural and clipped neural heuristics side by side. A second
figure shows the true distance field, the prediction and its signed error on
the bad maze.

Usage:
    python scripts/sample_predictions.py
    python scripts/sample_predictions.py --backend torch --checkpoint models/neural_heuristic_seed42.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.final_analysis import (  # noqa: E402
    BASELINE,
    NEURAL_CLIPPED,
    NEURAL_RAW,
    expand_ratio_table,
    load_search_results,
)
from src.search import astar  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def _manhattan(goal: tuple[int, int]) -> Any:
    from src.search import ManhattanHeuristic

    return ManhattanHeuristic(goal)


def _neural(grid, goal, backend, checkpoint_path, regressor_path, mode):
    if backend == "torch":
        from src.heuristics import NeuralHeuristic

        return NeuralHeuristic(checkpoint_path, grid, goal, alpha=1.0, mode=mode)
    from src.sklearn_heuristic import SklearnNeuralHeuristic

    return SklearnNeuralHeuristic(regressor_path, grid, goal, alpha=1.0, mode=mode)


def select_mazes(
    search_csv: str,
    max_size: int = 60,
) -> dict[str, dict[str, Any]]:
    """Pick good / typical / bad mazes with a readable grid size."""
    df = load_search_results(search_csv)
    ratios = expand_ratio_table(df).reset_index()
    meta = (
        df[df["heuristic"] == BASELINE]
        .set_index(["file_name", "local_index"])[["generator", "size", "density", "opt_cost"]]
        .reset_index()
    )
    table = ratios.merge(meta, on=["file_name", "local_index"], how="left")
    table = table[table["size"] <= max_size].copy()
    if table.empty:
        table = ratios.merge(meta, on=["file_name", "local_index"], how="left")
    table["ratio"] = table[f"{NEURAL_RAW}_ratio"]

    ordered = table.sort_values("ratio")
    median_ratio = float(table["ratio"].median())
    typical_idx = (table["ratio"] - median_ratio).abs().idxmin()

    selections = {
        "good": ordered.iloc[0],
        "typical": table.loc[typical_idx],
        "bad": ordered.iloc[-1],
    }
    result: dict[str, dict[str, Any]] = {}
    for label, row in selections.items():
        result[label] = {
            "file_name": str(row["file_name"]),
            "local_index": int(row["local_index"]),
            "generator": str(row["generator"]),
            "size": int(row["size"]),
            "density": int(row["density"]),
            "neural_ratio": float(row["ratio"]),
            "manhattan_nodes": float(row[BASELINE]),
            "neural_nodes": float(row[NEURAL_RAW]),
            "clipped_nodes": float(row[NEURAL_CLIPPED]),
        }
    return result


def _draw_panel(
    ax,
    grid: np.ndarray,
    expanded: Sequence[tuple[int, int]],
    path: Sequence[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    title: str,
) -> None:
    canvas = np.zeros((*grid.shape, 3), dtype=float)
    canvas[grid == 1] = (0.22, 0.24, 0.28)
    canvas[grid == 0] = (0.97, 0.97, 0.95)

    if expanded:
        rows = np.fromiter((r for r, _ in expanded), dtype=int)
        cols = np.fromiter((c for _, c in expanded), dtype=int)
        canvas[rows, cols] = (0.62, 0.79, 0.93)

    ax.imshow(canvas, interpolation="nearest")

    if path:
        rows = np.fromiter((r for r, _ in path), dtype=int)
        cols = np.fromiter((c for _, c in path), dtype=int)
        ax.plot(cols, rows, color="#c1272d", linewidth=2.0, zorder=4)

    ax.scatter([start[1]], [start[0]], s=60, marker="o", facecolor="#1a9850", edgecolor="black", zorder=5)
    ax.scatter([goal[1]], [goal[0]], s=110, marker="*", facecolor="#f4d03f", edgecolor="black", zorder=5)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def make_trace_figure(
    selections: dict[str, dict[str, Any]],
    cleaned_dir: str,
    output_path: str,
    backend: str,
    checkpoint_path: str,
    regressor_path: str,
) -> dict[str, Any]:
    """Render the 3x3 search-trace figure and return the measured node counts."""
    records: dict[str, Any] = {}
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.2), dpi=300)

    for row, (label, choice) in enumerate(selections.items()):
        path = os.path.join(cleaned_dir, choice["file_name"])
        with np.load(path) as data:
            grid = data["grids"][choice["local_index"]]
            start = tuple(int(v) for v in data["starts"][choice["local_index"]])
            goal = tuple(int(v) for v in data["goals"][choice["local_index"]])
            opt_cost = float(data["opt_cost"][choice["local_index"]])

        variants = [
            (
                "Manhattan",
                _manhattan(goal),
                BASELINE,
            ),
            (
                "Neural (raw)",
                _neural(grid, goal, backend, checkpoint_path, regressor_path, "raw"),
                NEURAL_RAW,
            ),
            (
                "Neural (clipped)",
                _neural(grid, goal, backend, checkpoint_path, regressor_path, "clipped_manhattan"),
                NEURAL_CLIPPED,
            ),
        ]

        row_record: dict[str, Any] = {"case": label, **choice}
        for column, (name, heuristic, key) in enumerate(variants):
            result = astar(grid, start, goal, heuristic, record_expanded=True)
            ratio = result.cost / opt_cost if opt_cost else float("nan")
            title = (
                f"{label.title()} maze - {name}\n"
                f"expanded {result.nodes_expanded}, cost/opt {ratio:.3f}"
            )
            _draw_panel(axes[row, column], grid, result.expanded or [], result.path, start, goal, title)
            row_record[key] = {
                "nodes_expanded": int(result.nodes_expanded),
                "cost": float(result.cost),
                "optimality_ratio": float(ratio),
            }
        records[label] = row_record

    fig.suptitle(
        "Search traces: expanded cells (blue) and final path (red)",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path)
    plt.close(fig)
    return records


def make_field_figure(
    choice: dict[str, Any],
    cleaned_dir: str,
    output_path: str,
    backend: str,
    checkpoint_path: str,
    regressor_path: str,
) -> None:
    """Show true h*, raw prediction, clipped prediction and signed error."""
    path = os.path.join(cleaned_dir, choice["file_name"])
    with np.load(path) as data:
        grid = data["grids"][choice["local_index"]]
        goal = tuple(int(v) for v in data["goals"][choice["local_index"]])
        h_star = data["h_star"][choice["local_index"]].astype(float)

    if backend == "torch":
        from src.heuristics import NeuralHeuristic

        heuristic = NeuralHeuristic(checkpoint_path, grid, goal, alpha=1.0, mode="raw")
        raw = heuristic.predictions().astype(float)
    else:
        from src.sklearn_heuristic import SklearnNeuralHeuristic

        heuristic = SklearnNeuralHeuristic(regressor_path, grid, goal, alpha=1.0, mode="raw")
        raw = np.asarray(heuristic._pred_grid, dtype=float)

    rows, cols = np.mgrid[0 : grid.shape[0], 0 : grid.shape[1]]
    manhattan = np.abs(rows - goal[0]) + np.abs(cols - goal[1])
    clipped = np.minimum(raw, manhattan)
    mask = (grid == 0) & (h_star >= 0)

    fields = [
        (np.where(mask, h_star, np.nan), "Ground truth $h^\\ast$", "viridis"),
        (np.where(mask, raw, np.nan), "Raw prediction $\\hat{h}_\\theta$", "viridis"),
        (np.where(mask, clipped, np.nan), "Clipped prediction", "viridis"),
        (np.where(mask, raw - h_star, np.nan), "Signed error (raw)", "coolwarm"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), dpi=300)
    for ax, (field, title, cmap) in zip(axes, fields):
        if title.startswith("Signed"):
            limit = np.nanpercentile(np.abs(field), 95)
            image = ax.imshow(field, cmap=cmap, vmin=-limit, vmax=limit)
        else:
            image = ax.imshow(field, cmap=cmap)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        ax.scatter([goal[1]], [goal[0]], marker="*", s=90, color="red", edgecolor="black")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Bad case: {choice['file_name']} (maze {choice['local_index']}), "
        f"{int(choice['size'])}x{int(choice['size'])}, density {int(choice['density'])}%",
        fontsize=11,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-csv", default="results/evaluation/test_per_maze.csv")
    parser.add_argument("--cleaned-dir", default="data/cleaned")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--backend", choices=("sklearn", "torch"), default="sklearn")
    parser.add_argument("--checkpoint", default="models/neural_heuristic_seed42.pt")
    parser.add_argument("--regressor", default="models/fallback_mlp_regressor.joblib")
    parser.add_argument("--max-size", type=int, default=60)
    args = parser.parse_args(argv)

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    selections = select_mazes(args.search_csv, max_size=args.max_size)
    trace_path = os.path.join(args.figures_dir, "sample_search_traces.png")
    records = make_trace_figure(
        selections,
        args.cleaned_dir,
        trace_path,
        args.backend,
        args.checkpoint,
        args.regressor,
    )

    field_path = os.path.join(args.figures_dir, "sample_prediction_fields.png")
    make_field_figure(
        selections["bad"],
        args.cleaned_dir,
        field_path,
        args.backend,
        args.checkpoint,
        args.regressor,
    )

    import json

    payload = {
        "selections": selections,
        "measured": records,
        "backend": args.backend,
        "figures": [trace_path, field_path],
    }
    with open(os.path.join(args.output_dir, "sample_selections.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)

    for label, record in records.items():
        print(
            f"{label:8s} {record['file_name']}:{record['local_index']} "
            f"({record['generator']}, {record['size']}, {record['density']}%) "
            f"man={record[BASELINE]['nodes_expanded']} "
            f"raw={record[NEURAL_RAW]['nodes_expanded']} "
            f"clip={record[NEURAL_CLIPPED]['nodes_expanded']}"
        )
    print(f"wrote {trace_path}")
    print(f"wrote {field_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
