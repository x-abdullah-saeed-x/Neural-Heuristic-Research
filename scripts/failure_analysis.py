"""Deliverable 3 failure analysis for the neural heuristic.

Answers the Phase 5 questions:
  * where does the learned heuristic lose to Manhattan A*;
  * how does the loss depend on generator, size, density and path tortuosity;
  * why it happens, measured as distance bias, dead-end error and heuristic
    inconsistency;
  * what changes when the prediction is clipped to the Manhattan bound.

Writes tables and JSON to ``results/analysis`` and figures to ``figures``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.final_analysis import (  # noqa: E402
    BASELINE,
    NEURAL_CLIPPED,
    NEURAL_RAW,
    add_tortuosity_strata,
    consistency_violation_rate,
    dead_end_mask,
    expand_ratio_table,
    load_search_results,
    maze_metadata,
    paired_comparison,
    wall_neighbour_count,
    write_json,
)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)

COLORS = {
    BASELINE: "#3b6fb6",
    NEURAL_RAW: "#d1603d",
    NEURAL_CLIPPED: "#2a9d8f",
}


def _manhattan_field(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    n = grid.shape[0]
    rows, cols = np.mgrid[0:n, 0:n]
    return (np.abs(rows - goal[0]) + np.abs(cols - goal[1])).astype(float)


def build_prediction_fields(
    grid: np.ndarray,
    goal: tuple[int, int],
    backend: str = "torch",
    checkpoint_path: str = "models/neural_heuristic_seed42.pt",
    regressor_path: str = "models/fallback_mlp_regressor.joblib",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(raw, clipped, manhattan)`` heuristic fields for one maze."""
    if backend == "torch":
        from src.heuristics import NeuralHeuristic

        heuristic = NeuralHeuristic(checkpoint_path, grid, goal, alpha=1.0, mode="raw")
        raw = heuristic.predictions().astype(float)
    elif backend == "sklearn":
        from src.sklearn_heuristic import SklearnNeuralHeuristic

        heuristic = SklearnNeuralHeuristic(regressor_path, grid, goal, alpha=1.0, mode="raw")
        raw = np.asarray(heuristic._pred_grid, dtype=float)
    else:
        raise ValueError(f"Unknown backend {backend!r}; choose 'torch' or 'sklearn'")

    manhattan = _manhattan_field(grid, goal)
    clipped = np.minimum(raw, manhattan)
    return raw, clipped, manhattan


def per_maze_table(search_csv: str, cleaned_dir: str) -> pd.DataFrame:
    """Assemble the per-maze failure table used by the report."""
    df = load_search_results(search_csv)
    meta = maze_metadata(df, cleaned_dir=cleaned_dir)
    ratios = expand_ratio_table(df)

    baseline_rows = (
        df[df["heuristic"] == BASELINE]
        .set_index(["file_name", "local_index"])[["generator", "size", "density", "opt_cost"]]
    )
    table = ratios.join(meta[["tortuosity", "manhattan_lb", "start_r", "start_c", "goal_r", "goal_c"]])
    table = table.join(baseline_rows[["generator", "size", "density"]])
    table = table.rename(
        columns={
            BASELINE: "manhattan_nodes",
            NEURAL_RAW: "neural_nodes",
            f"{NEURAL_RAW}_ratio": "neural_ratio",
        }
    )
    if NEURAL_CLIPPED in table:
        table = table.rename(
            columns={
                NEURAL_CLIPPED: "clipped_nodes",
                f"{NEURAL_CLIPPED}_ratio": "clipped_ratio",
            }
        )
    table = add_tortuosity_strata(table.reset_index()).set_index(["file_name", "local_index"])
    return table


def _group_rows(per_maze: pd.DataFrame) -> pd.DataFrame:
    """Long-format expansion table across every reported grouping."""
    rows = []
    for column in ("generator", "size", "density", "tortuosity_stratum"):
        grouped = per_maze.groupby(column, observed=True)
        for value, group in grouped:
            row = {
                "group": column,
                "value": str(value),
                "n_mazes": int(len(group)),
                "manhattan_nodes": float(group["manhattan_nodes"].mean()),
                "neural_nodes": float(group["neural_nodes"].mean()),
                "neural_ratio": float(group["neural_nodes"].mean() / group["manhattan_nodes"].mean()),
                "neural_win_rate": float((group["neural_nodes"] < group["manhattan_nodes"]).mean()),
            }
            if "clipped_nodes" in group:
                row["clipped_nodes"] = float(group["clipped_nodes"].mean())
                row["clipped_ratio"] = float(
                    group["clipped_nodes"].mean() / group["manhattan_nodes"].mean()
                )
                row["clipped_win_rate"] = float(
                    (group["clipped_nodes"] < group["manhattan_nodes"]).mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def consistency_analysis(
    per_maze: pd.DataFrame,
    cleaned_dir: str,
    backend: str,
    checkpoint_path: str,
    regressor_path: str,
    max_mazes: int = 48,
    seed: int = 42,
    cells_per_maze: int = 5000,
) -> dict[str, Any]:
    """Measure heuristic inconsistency and residual bias on a sample of test mazes.

    Cell-level residuals are pooled across mazes before binning, so a distance
    bin contains comparable numbers of cells from every sampled maze instead of
    one bin per maze.
    """
    if max_mazes <= 0 or max_mazes >= len(per_maze):
        sample = per_maze
    else:
        sample = per_maze.sample(n=max_mazes, random_state=seed)

    violation = {"manhattan": [], NEURAL_RAW: [], NEURAL_CLIPPED: []}
    truth_chunks: list[np.ndarray] = []
    raw_error_chunks: list[np.ndarray] = []
    clipped_error_chunks: list[np.ndarray] = []
    wall_chunks: list[np.ndarray] = []
    dead_chunks: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for (file_name, local_index), _ in sample.iterrows():
        path = os.path.join(cleaned_dir, file_name)
        with np.load(path) as data:
            grid = data["grids"][int(local_index)]
            goal = tuple(int(v) for v in data["goals"][int(local_index)])
            h_star = data["h_star"][int(local_index)].astype(float)

        raw, clipped, manhattan = build_prediction_fields(
            grid,
            goal,
            backend=backend,
            checkpoint_path=checkpoint_path,
            regressor_path=regressor_path,
        )
        for name, field in (("manhattan", manhattan), (NEURAL_RAW, raw), (NEURAL_CLIPPED, clipped)):
            violation[name].append(consistency_violation_rate(field, grid))

        free = (grid == 0) & (h_star >= 0)
        flat = np.flatnonzero(free.ravel())
        if flat.size > cells_per_maze:
            flat = rng.choice(flat, size=cells_per_maze, replace=False)

        truth = h_star.ravel()[flat]
        truth_chunks.append(truth)
        raw_error_chunks.append(raw.ravel()[flat] - truth)
        clipped_error_chunks.append(clipped.ravel()[flat] - truth)
        wall_chunks.append(wall_neighbour_count(grid).ravel()[flat])
        dead_chunks.append(dead_end_mask(grid).ravel()[flat])

    truth = np.concatenate(truth_chunks) if truth_chunks else np.array([])
    raw_error = np.concatenate(raw_error_chunks) if raw_error_chunks else np.array([])
    clipped_error = np.concatenate(clipped_error_chunks) if clipped_error_chunks else np.array([])
    wall_count = np.concatenate(wall_chunks) if wall_chunks else np.array([])
    dead_end = np.concatenate(dead_chunks) if dead_chunks else np.array([])

    def field_summary(error: np.ndarray) -> dict[str, float]:
        if error.size == 0:
            return {}
        absolute = np.abs(error)
        return {
            "n_cells": int(error.size),
            "mae_cells": float(np.mean(absolute)),
            "rmse_cells": float(np.sqrt(np.mean(error**2))),
            "mean_signed_error_cells": float(np.mean(error)),
            "p95_absolute_error_cells": float(np.percentile(absolute, 95)),
            "overestimation_rate": float(np.mean(error > 0.5)),
            "underestimation_rate": float(np.mean(error < -0.5)),
            "within_one_cell_rate": float(np.mean(absolute <= 1.0)),
        }

    edges = np.unique(np.quantile(truth, np.linspace(0.0, 1.0, 9))) if truth.size else np.array([])
    distance_rows: list[dict[str, float]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (truth >= low) & (truth <= high)
        if not np.any(selected):
            continue
        for variant, error in (("raw", raw_error), ("clipped", clipped_error)):
            selected_error = error[selected]
            distance_rows.append(
                {
                    "variant": variant,
                    "true_distance_low": float(low),
                    "true_distance_high": float(high),
                    "count": int(selected.sum()),
                    "mean_signed_error_cells": float(np.mean(selected_error)),
                    "mae_cells": float(np.mean(np.abs(selected_error))),
                    "overestimation_rate": float(np.mean(selected_error > 0.5)),
                }
            )

    wall_rows: list[dict[str, float]] = []
    for count in sorted(np.unique(wall_count)):
        selected = wall_count == count
        if selected.sum() < 10:
            continue
        error = raw_error[selected]
        wall_rows.append(
            {
                "neighbour_walls": int(count),
                "count": int(selected.sum()),
                "mean_signed_error_cells": float(np.mean(error)),
                "mae_cells": float(np.mean(np.abs(error))),
            }
        )

    dead_rows: list[dict[str, float]] = []
    for label, selected in (("dead_end", dead_end), ("other_free", ~dead_end)):
        selected = selected & np.ones_like(dead_end, dtype=bool)
        if not np.any(selected):
            continue
        error = raw_error[selected]
        dead_rows.append(
            {
                "stratum": label,
                "count": int(selected.sum()),
                "mae_cells": float(np.mean(np.abs(error))),
                "mean_signed_error_cells": float(np.mean(error)),
                "overestimation_rate": float(np.mean(error > 0.5)),
            }
        )

    return {
        "mazes_evaluated": int(len(sample)),
        "backend": backend,
        "cells_per_maze_cap": int(cells_per_maze),
        "consistency_violation_rate": {
            name: float(np.nanmean(values)) if values else float("nan")
            for name, values in violation.items()
        },
        "consistency_violation_rate_std": {
            name: float(np.nanstd(values)) if values else float("nan")
            for name, values in violation.items()
        },
        "raw_field_summary": [field_summary(raw_error)],
        "clipped_field_summary": [field_summary(clipped_error)],
        "error_by_true_distance": distance_rows,
        "error_by_local_walls": wall_rows,
        "dead_end_vs_other": dead_rows,
    }


def make_figures(per_maze: pd.DataFrame, consistency: dict[str, Any], figures_dir: str) -> list[str]:
    """Render the failure-analysis figures and return their paths."""
    os.makedirs(figures_dir, exist_ok=True)
    written: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    for ax, column, title in zip(
        axes,
        ("density", "generator"),
        ("By obstacle density (%)", "By maze generator"),
    ):
        table = _group_rows(per_maze)
        rows = table[table["group"] == column]
        x = np.arange(len(rows))
        width = 0.38
        ax.bar(x - width / 2, rows["neural_ratio"], width, color=COLORS[NEURAL_RAW], label="neural (raw)")
        ax.bar(
            x + width / 2,
            rows["clipped_ratio"],
            width,
            color=COLORS[NEURAL_CLIPPED],
            label="neural (clipped)",
        )
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2, label="Manhattan parity")
        ax.set_xticks(x)
        ax.set_xticklabels(rows["value"], rotation=0)
        ax.set_xlabel(column.capitalize())
        ax.set_ylabel("Nodes expanded / Manhattan")
        ax.set_title(title)
        ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    path = os.path.join(figures_dir, "failure_expansion_by_group.png")
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=300)
    ax.scatter(
        per_maze["tortuosity"],
        per_maze["neural_ratio"],
        s=22,
        alpha=0.55,
        color=COLORS[NEURAL_RAW],
        label="Test maze",
    )
    ax.set_yscale("log")
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2, label="Manhattan parity")
    ordered = per_maze.sort_values("tortuosity")
    bins = np.unique(np.quantile(ordered["tortuosity"], np.linspace(0, 1, 7)))
    if bins.size > 1:
        centres, medians = [], []
        for low, high in zip(bins[:-1], bins[1:]):
            group = ordered[(ordered["tortuosity"] >= low) & (ordered["tortuosity"] <= high)]
            if not group.empty:
                centres.append(float(np.median(group["tortuosity"])))
                medians.append(float(np.median(group["neural_ratio"])))
        ax.plot(centres, medians, color="#222222", marker="o", linewidth=1.6, label="Binned median")
    ax.set_xlabel("Path tortuosity  $c^\\ast / d_M$")
    ax.set_ylabel("Neural / Manhattan nodes (log)")
    ax.set_title("Failure is concentrated on direct, low-detour mazes")
    ax.legend(frameon=True)
    fig.tight_layout()
    path = os.path.join(figures_dir, "failure_vs_tortuosity.png")
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    distance = consistency.get("error_by_true_distance", [])
    if distance:
        raw_rows = [row for row in distance if row.get("variant") == "raw"]
        clipped_by_centre = {
            round(0.5 * (row["true_distance_low"] + row["true_distance_high"]), 3): row
            for row in distance
            if row.get("variant") == "clipped"
        }
        centres = [0.5 * (row["true_distance_low"] + row["true_distance_high"]) for row in raw_rows]
        fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=300)
        width = 0.38
        x = np.arange(len(raw_rows))
        ax.bar(
            x - width / 2,
            [row["mean_signed_error_cells"] for row in raw_rows],
            width,
            color=COLORS[NEURAL_RAW],
            label="raw prediction",
        )
        ax.bar(
            x + width / 2,
            [
                clipped_by_centre.get(round(centre, 3), {}).get("mean_signed_error_cells", 0.0)
                for centre in centres
            ],
            width,
            color=COLORS[NEURAL_CLIPPED],
            label="clipped prediction",
        )
        ax.axhline(0.0, color="#444444", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{centre:.0f}" for centre in centres])
        ax.set_xlabel("True cost-to-goal $h^\\ast$ (cells)")
        ax.set_ylabel("Mean signed error (cells)")
        ax.set_title("Underestimation grows with distance from the goal")
        ax.legend(frameon=True)
        fig.tight_layout()
        path = os.path.join(figures_dir, "failure_error_by_distance.png")
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    rates = consistency.get("consistency_violation_rate", {})
    if rates:
        labels = [BASELINE, NEURAL_RAW, NEURAL_CLIPPED]
        values = [rates.get(name, 0.0) * 100.0 for name in labels]
        fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=300)
        bars = ax.bar(
            ["Manhattan", "Neural (raw)", "Neural (clipped)"],
            values,
            color=[COLORS[BASELINE], COLORS[NEURAL_RAW], COLORS[NEURAL_CLIPPED]],
        )
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", fontsize=9)
        ax.set_ylabel("Adjacent pairs violating consistency (%)")
        ax.set_title("Clipping restores near-consistency")
        ax.set_ylim(0, max(values) * 1.2 + 1.0)
        fig.tight_layout()
        path = os.path.join(figures_dir, "failure_consistency.png")
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    return written


def _correlations(per_maze: pd.DataFrame) -> dict[str, dict[str, float]]:
    log_ratio = np.log(per_maze["neural_ratio"].clip(lower=1e-3))
    result = {}
    for column in ("tortuosity", "size", "density", "manhattan_lb"):
        values = per_maze[column].astype(float)
        pearson = float(np.corrcoef(values, log_ratio)[0, 1])
        spearman = float(values.corr(log_ratio, method="spearman"))
        result[column] = {"pearson_vs_log_ratio": pearson, "spearman_vs_log_ratio": spearman}
    return result


def run(
    search_csv: str = "results/evaluation/test_per_maze.csv",
    cleaned_dir: str = "data/cleaned",
    output_dir: str = "results/analysis",
    figures_dir: str = "figures",
    backend: str = "sklearn",
    checkpoint_path: str = "models/neural_heuristic_seed42.pt",
    regressor_path: str = "models/fallback_mlp_regressor.joblib",
    max_consistency_mazes: int = 48,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full failure analysis and persist all artefacts."""
    os.makedirs(output_dir, exist_ok=True)
    search_df = load_search_results(search_csv)
    per_maze = per_maze_table(search_csv, cleaned_dir)

    raw_vs_man = paired_comparison(search_df, BASELINE, NEURAL_RAW)
    clipped_vs_man = paired_comparison(search_df, BASELINE, NEURAL_CLIPPED)
    raw_vs_clipped = paired_comparison(search_df, NEURAL_CLIPPED, NEURAL_RAW)

    consistency = consistency_analysis(
        per_maze,
        cleaned_dir=cleaned_dir,
        backend=backend,
        checkpoint_path=checkpoint_path,
        regressor_path=regressor_path,
        max_mazes=max_consistency_mazes,
        seed=seed,
    )

    groups = _group_rows(per_maze)
    correlations = _correlations(per_maze)

    worst = (
        per_maze.sort_values("neural_ratio", ascending=False)
        .head(10)
        .reset_index()[
            [
                "file_name",
                "local_index",
                "generator",
                "size",
                "density",
                "tortuosity",
                "manhattan_nodes",
                "neural_nodes",
                "neural_ratio",
            ]
        ]
        .to_dict(orient="records")
    )

    summary = {
        "n_mazes": int(len(per_maze)),
        "raw_vs_manhattan": raw_vs_man,
        "clipped_vs_manhattan": clipped_vs_man,
        "raw_vs_clipped": raw_vs_clipped,
        "correlations": correlations,
        "tortuosity_direct_fraction": float((per_maze["tortuosity"] <= 1.0001).mean()),
        "worst_mazes": worst,
        "consistency_and_bias": consistency,
    }

    per_maze.reset_index().to_csv(os.path.join(output_dir, "failure_per_maze.csv"), index=False)
    groups.to_csv(os.path.join(output_dir, "failure_by_group.csv"), index=False)
    write_json(summary, os.path.join(output_dir, "failure_summary.json"))
    figures = make_figures(per_maze, consistency, figures_dir)
    summary["figures"] = figures
    write_json(summary, os.path.join(output_dir, "failure_summary.json"))
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-csv", default="results/evaluation/test_per_maze.csv")
    parser.add_argument("--cleaned-dir", default="data/cleaned")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument(
        "--backend",
        choices=("torch", "sklearn"),
        default="sklearn",
        help="Model backend. The committed PyTorch checkpoint is a smoke-test "
        "artifact; the reported metrics use the scikit-learn fallback.",
    )
    parser.add_argument("--checkpoint", default="models/neural_heuristic_seed42.pt")
    parser.add_argument("--regressor", default="models/fallback_mlp_regressor.joblib")
    parser.add_argument("--consistency-mazes", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    summary = run(
        search_csv=args.search_csv,
        cleaned_dir=args.cleaned_dir,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        backend=args.backend,
        checkpoint_path=args.checkpoint,
        regressor_path=args.regressor,
        max_consistency_mazes=args.consistency_mazes,
        seed=args.seed,
    )
    raw = summary["raw_vs_manhattan"]
    clipped = summary["clipped_vs_manhattan"]
    print(
        f"raw neural: {raw['treatment_mean']:.1f} vs {raw['baseline_mean']:.1f} nodes "
        f"({raw['relative_change'] * 100:+.1f}%), wins {raw['win_rate'] * 100:.1f}%, "
        f"p={raw['wilcoxon_p_value']:.2e}"
    )
    print(
        f"clipped:    {clipped['treatment_mean']:.1f} vs {clipped['baseline_mean']:.1f} nodes "
        f"({clipped['relative_change'] * 100:+.1f}%), wins {clipped['win_rate'] * 100:.1f}%, "
        f"p={clipped['wilcoxon_p_value']:.2e}"
    )
    print(f"wrote results to {args.output_dir} and figures to {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
