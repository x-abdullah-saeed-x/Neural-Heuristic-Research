"""Deliverable 3 quantitative analysis utilities.

This module contains the reusable pieces behind the final Track B analysis:
paired heuristic comparisons, failure stratification, heuristic-consistency
diagnostics and the helpers used by the feature-group ablation. It is imported
by the command-line scripts in ``scripts/`` and covered by
``tests/test_final_analysis.py``.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

BASELINE = "manhattan"
NEURAL_RAW = "neural_alpha_1.0"
NEURAL_CLIPPED = "neural_clipped_manhattan"

MAZE_KEYS = ["file_name", "local_index"]

# Tortuosity is the optimal path cost divided by the straight-line Manhattan
# lower bound. A value of 1.0 means the maze did not force a detour at all.
TORTUOSITY_EDGES = [1.0, 1.05, 1.25, 1.5, np.inf]
TORTUOSITY_LABELS = ["direct (1.00)", "mild (<1.25)", "moderate (<1.50)", "severe (>=1.50)"]


def load_search_results(path: str) -> pd.DataFrame:
    """Load a per-maze search CSV and validate the columns this module needs."""
    df = pd.read_csv(path)
    required = {
        "file_name",
        "local_index",
        "heuristic",
        "nodes_expanded",
        "opt_cost",
        "size",
        "density",
        "generator",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def heuristic_series(
    df: pd.DataFrame,
    heuristic: str,
    metric: str = "nodes_expanded",
) -> pd.Series:
    """Return one metric for one heuristic indexed by (file_name, local_index)."""
    subset = df[df["heuristic"] == heuristic]
    if subset.empty:
        raise ValueError(f"Heuristic {heuristic!r} is not present in the results")
    series = subset.set_index(MAZE_KEYS)[metric]
    if series.index.has_duplicates:
        raise ValueError(f"Duplicate maze rows for heuristic {heuristic!r}")
    return series.sort_index()


def paired_differences(
    df: pd.DataFrame,
    baseline: str = BASELINE,
    treatment: str = NEURAL_RAW,
    metric: str = "nodes_expanded",
) -> pd.DataFrame:
    """Align two heuristics on the same mazes and return their paired rows."""
    base = heuristic_series(df, baseline, metric)
    treat = heuristic_series(df, treatment, metric)
    common = base.index.intersection(treat.index)
    paired = pd.DataFrame(
        {
            "baseline": base.loc[common].to_numpy(dtype=float),
            "treatment": treat.loc[common].to_numpy(dtype=float),
        },
        index=common,
    )
    paired["difference"] = paired["treatment"] - paired["baseline"]
    return paired


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    sample = np.asarray(values, dtype=float).ravel()
    if sample.size == 0:
        return (float("nan"), float("nan"))
    if sample.size == 1:
        return (float(sample[0]), float(sample[0]))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, sample.size, size=(n_boot, sample.size))
    means = sample[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(means, [100.0 * tail, 100.0 * (1.0 - tail)])
    return (float(low), float(high))


def paired_comparison(
    df: pd.DataFrame,
    baseline: str = BASELINE,
    treatment: str = NEURAL_RAW,
    metric: str = "nodes_expanded",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare two heuristics with paired non-parametric statistics.

    Reports the mean/median node counts, the paired Wilcoxon signed-rank test,
    a bootstrap confidence interval for the mean difference and the win/tie/loss
    counts. The difference is ``treatment - baseline``, so a negative value
    favours the treatment.
    """
    paired = paired_differences(df, baseline, treatment, metric)
    base = paired["baseline"].to_numpy()
    treat = paired["treatment"].to_numpy()
    diff = paired["difference"].to_numpy()

    if np.any(diff != 0):
        try:
            statistic, p_value = stats.wilcoxon(treat, base)
            statistic, p_value = float(statistic), float(p_value)
        except ValueError:
            statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = 0.0, 1.0

    low, high = bootstrap_mean_ci(diff, n_boot=n_boot, seed=seed)
    differs = diff != 0
    n = int(len(diff))

    return {
        "baseline": baseline,
        "treatment": treatment,
        "metric": metric,
        "n_mazes": n,
        "baseline_mean": float(np.mean(base)),
        "baseline_median": float(np.median(base)),
        "treatment_mean": float(np.mean(treat)),
        "treatment_median": float(np.median(treat)),
        "mean_difference": float(np.mean(diff)),
        "median_difference": float(np.median(diff)),
        "relative_change": float(np.mean(diff) / np.mean(base)) if np.mean(base) else float("nan"),
        "mean_difference_ci_low": low,
        "mean_difference_ci_high": high,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
        "treatment_wins": int(np.sum(diff < 0)),
        "ties": int(np.sum(~differs)),
        "treatment_losses": int(np.sum(diff > 0)),
        "win_rate": float(np.mean(diff < 0)),
    }


def manhattan_lower_bound(start: Sequence[int], goal: Sequence[int]) -> int:
    """Manhattan distance between start and goal rounded to whole cells."""
    return int(abs(int(start[0]) - int(goal[0])) + abs(int(start[1]) - int(goal[1])))


def maze_metadata(
    df: pd.DataFrame,
    cleaned_dir: str = "data/cleaned",
    baseline: str = BASELINE,
) -> pd.DataFrame:
    """Return per-maze start/goal-derived metadata indexed by maze key.

    The optimal cost and the Manhattan lower bound give the tortuosity
    ``opt_cost / d_M(start, goal)``, a direct measure of how much the maze
    forces a detour. ``d_M == 0`` cannot occur for the generated mazes but is
    clipped to one so that the ratio stays finite.
    """
    base_rows = df[df["heuristic"] == baseline]
    if base_rows.empty:
        raise ValueError(f"Baseline heuristic {baseline!r} is not present")

    records: list[dict[str, Any]] = []
    for file_name, group in base_rows.groupby("file_name"):
        path = os.path.join(cleaned_dir, file_name)
        with np.load(path) as data:
            starts = data["starts"]
            goals = data["goals"]
            for row in group.itertuples(index=False):
                index = int(row.local_index)
                start = tuple(int(v) for v in starts[index])
                goal = tuple(int(v) for v in goals[index])
                lower = manhattan_lower_bound(start, goal)
                records.append(
                    {
                        "file_name": file_name,
                        "local_index": index,
                        "start_r": start[0],
                        "start_c": start[1],
                        "goal_r": goal[0],
                        "goal_c": goal[1],
                        "manhattan_lb": lower,
                        "opt_cost": float(row.opt_cost),
                    }
                )

    meta = pd.DataFrame(records).set_index(MAZE_KEYS)
    meta["tortuosity"] = meta["opt_cost"] / meta["manhattan_lb"].clip(lower=1)
    return meta


def expand_ratio_table(
    df: pd.DataFrame,
    baseline: str = BASELINE,
    treatments: Sequence[str] = (NEURAL_RAW, NEURAL_CLIPPED),
    metric: str = "nodes_expanded",
) -> pd.DataFrame:
    """One row per maze with the baseline metric and each treatment's ratio."""
    table = pd.DataFrame({baseline: heuristic_series(df, baseline, metric)})
    for treatment in treatments:
        series = heuristic_series(df, treatment, metric)
        table[treatment] = series
        table[f"{treatment}_ratio"] = series / table[baseline].replace(0, np.nan)
    return table.dropna()


def group_expansion_table(
    df: pd.DataFrame,
    by: str,
    baseline: str = BASELINE,
    treatments: Sequence[str] = (NEURAL_RAW, NEURAL_CLIPPED),
    metric: str = "nodes_expanded",
) -> pd.DataFrame:
    """Mean expansions and treatment/baseline ratios grouped by a maze attribute."""
    columns = {baseline: heuristic_series(df, baseline, metric)}
    for treatment in treatments:
        columns[treatment] = heuristic_series(df, treatment, metric)
    table = pd.DataFrame(columns)

    meta = df[df["heuristic"] == baseline].set_index(MAZE_KEYS)
    table[by] = meta[by]

    grouped = table.groupby(by).mean(numeric_only=True)
    for treatment in treatments:
        grouped[f"{treatment}_ratio"] = grouped[treatment] / grouped[baseline]
    return grouped.reset_index()


def add_tortuosity_strata(table: pd.DataFrame, column: str = "tortuosity") -> pd.DataFrame:
    """Add a categorical tortuosity stratum column to a per-maze table."""
    result = table.copy()
    result["tortuosity_stratum"] = pd.cut(
        result[column],
        bins=TORTUOSITY_EDGES,
        labels=TORTUOSITY_LABELS,
        include_lowest=True,
        right=False,
    ).astype(str)
    return result


def consistency_violation_rate(
    h_field: np.ndarray,
    grid: np.ndarray,
    tolerance: float = 0.0,
) -> float:
    """Fraction of adjacent free-cell pairs that violate heuristic consistency.

    For a unit-cost grid, consistency requires ``|h(u) - h(v)| <= 1`` for every
    pair of adjacent free cells. Larger jumps mean the heuristic changes faster
    than the true cost can, which is what forces A* to reopen nodes and expand
    wide regions around overestimated obstacles.
    """
    h = np.asarray(h_field, dtype=float)
    walls = np.asarray(grid) == 1
    if h.shape != walls.shape:
        raise ValueError("h_field and grid must have the same shape")

    free = ~walls
    violations = 0
    pairs = 0
    for axis in (0, 1):
        first = np.take(h, range(h.shape[axis] - 1), axis=axis)
        second = np.take(h, range(1, h.shape[axis]), axis=axis)
        mask = np.take(free, range(free.shape[axis] - 1), axis=axis) & np.take(
            free, range(1, free.shape[axis]), axis=axis
        )
        jumps = np.abs(first - second)[mask] if mask.any() else np.array([])
        violations += int(np.sum(jumps > 1.0 + tolerance))
        pairs += int(jumps.size)

    return float(violations / pairs) if pairs else float("nan")


def overestimation_summary(
    h_pred: np.ndarray,
    h_star: np.ndarray,
    grid: np.ndarray,
    tolerance: float = 0.5,
) -> dict[str, float]:
    """Residual statistics of a predicted field on reachable free cells."""
    grid = np.asarray(grid)
    mask = (grid == 0) & (np.asarray(h_star) >= 0) & np.isfinite(h_pred)
    if not np.any(mask):
        return {}

    truth = np.asarray(h_star, dtype=float)[mask]
    pred = np.asarray(h_pred, dtype=float)[mask]
    error = pred - truth
    abs_error = np.abs(error)
    return {
        "n_cells": int(mask.sum()),
        "mae_cells": float(np.mean(abs_error)),
        "rmse_cells": float(np.sqrt(np.mean(error**2))),
        "mean_signed_error_cells": float(np.mean(error)),
        "p95_absolute_error_cells": float(np.percentile(abs_error, 95)),
        "overestimation_rate": float(np.mean(error > tolerance)),
        "underestimation_rate": float(np.mean(error < -tolerance)),
        "within_one_cell_rate": float(np.mean(abs_error <= 1.0)),
    }


def error_by_true_distance(
    h_pred: np.ndarray,
    h_star: np.ndarray,
    grid: np.ndarray,
    n_bins: int = 8,
) -> list[dict[str, float]]:
    """Residual statistics stratified by the true cost-to-goal.

    Underestimation far from the goal is the failure mode that leaves A* with a
    weak gradient near the start. This function makes that visible.
    """
    grid = np.asarray(grid)
    mask = (grid == 0) & (np.asarray(h_star) >= 0) & np.isfinite(h_pred)
    truth = np.asarray(h_star, dtype=float)[mask]
    pred = np.asarray(h_pred, dtype=float)[mask]
    if truth.size == 0:
        return []

    edges = np.quantile(truth, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return []

    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (truth >= low) & (truth <= high)
        error = pred[selected] - truth[selected]
        rows.append(
            {
                "true_distance_low": float(low),
                "true_distance_high": float(high),
                "count": int(selected.sum()),
                "mean_signed_error_cells": float(np.mean(error)),
                "mae_cells": float(np.mean(np.abs(error))),
                "overestimation_rate": float(np.mean(error > 0.5)),
            }
        )
    return rows


def wall_neighbour_count(grid: np.ndarray) -> np.ndarray:
    """Number of in-grid wall neighbours of every cell (0-4)."""
    free = np.asarray(grid) == 0
    padded_free = np.pad(free, 1, mode="constant", constant_values=False).astype(np.int16)
    free_used = (
        padded_free[:-2, 1:-1]
        + padded_free[2:, 1:-1]
        + padded_free[1:-1, :-2]
        + padded_free[1:-1, 2:]
    )
    padded_inside = np.pad(np.ones_like(free, dtype=bool), 1, mode="constant", constant_values=False)
    available = (
        padded_inside[:-2, 1:-1].astype(np.int16)
        + padded_inside[2:, 1:-1].astype(np.int16)
        + padded_inside[1:-1, :-2].astype(np.int16)
        + padded_inside[1:-1, 2:].astype(np.int16)
    )
    return (available - free_used).astype(np.int16)


def error_by_local_walls(
    h_pred: np.ndarray,
    h_star: np.ndarray,
    grid: np.ndarray,
) -> list[dict[str, float]]:
    """Residual statistics stratified by the number of wall neighbours (0-4)."""
    mask = (np.asarray(grid) == 0) & (np.asarray(h_star) >= 0) & np.isfinite(h_pred)
    wall_count = wall_neighbour_count(grid)

    rows = []
    for count in sorted(np.unique(wall_count[mask])):
        selected = mask & (wall_count == count)
        if selected.sum() < 10:
            continue
        error = np.asarray(h_pred, dtype=float)[selected] - np.asarray(h_star, dtype=float)[selected]
        rows.append(
            {
                "neighbour_walls": int(count),
                "count": int(selected.sum()),
                "mean_signed_error_cells": float(np.mean(error)),
                "mae_cells": float(np.mean(np.abs(error))),
            }
        )
    return rows


def dead_end_mask(grid: np.ndarray) -> np.ndarray:
    """Boolean mask of free cells with exactly one free 4-neighbour."""
    free = np.asarray(grid) == 0
    padded = np.pad(free, 1, mode="constant", constant_values=False).astype(np.int16)
    neighbours = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return free & (neighbours == 1)


def dead_end_summary(
    h_pred: np.ndarray,
    h_star: np.ndarray,
    grid: np.ndarray,
) -> list[dict[str, float]]:
    """Residual statistics for dead-end cells versus all other free cells."""
    mask = (np.asarray(grid) == 0) & (np.asarray(h_star) >= 0) & np.isfinite(h_pred)
    ends = dead_end_mask(grid) & mask
    rows = []
    for label, selected in (("dead_end", ends), ("other_free", mask & ~ends)):
        if not np.any(selected):
            continue
        error = np.asarray(h_pred, dtype=float)[selected] - np.asarray(h_star, dtype=float)[selected]
        rows.append(
            {
                "stratum": label,
                "count": int(selected.sum()),
                "mae_cells": float(np.mean(np.abs(error))),
                "mean_signed_error_cells": float(np.mean(error)),
                "overestimation_rate": float(np.mean(error > 0.5)),
            }
        )
    return rows


def stratified_expansion_summary(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    by: str = "tortuosity_stratum",
    baseline: str = BASELINE,
    treatments: Sequence[str] = (NEURAL_RAW, NEURAL_CLIPPED),
    metric: str = "nodes_expanded",
) -> pd.DataFrame:
    """Expansion ratios stratified by a per-maze metadata column."""
    ratios = expand_ratio_table(df, baseline, treatments, metric)
    merged = ratios.join(meta[[by]], how="inner").dropna()
    rows = []
    for value, group in merged.groupby(by, observed=True):
        row = {by: value, "n_mazes": int(len(group)), baseline: float(group[baseline].mean())}
        for treatment in treatments:
            mean_treat = float(group[treatment].mean())
            row[treatment] = mean_treat
            row[f"{treatment}_ratio"] = mean_treat / row[baseline] if row[baseline] else float("nan")
            row[f"{treatment}_win_rate"] = float(np.mean(group[treatment] < group[baseline]))
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_summary(values: Iterable[float]) -> dict[str, float]:
    """Compact summary of a numeric sequence, used in JSON reports."""
    sample = np.asarray(list(values), dtype=float)
    if sample.size == 0:
        return {}
    return {
        "n": int(sample.size),
        "mean": float(np.mean(sample)),
        "std": float(np.std(sample, ddof=1)) if sample.size > 1 else 0.0,
        "median": float(np.median(sample)),
        "min": float(np.min(sample)),
        "max": float(np.max(sample)),
    }


def write_json(payload: Mapping[str, Any], path: str) -> None:
    """Write an indented JSON document, creating parent directories."""
    import json

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
