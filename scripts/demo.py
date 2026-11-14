"""Interactive demo: solve one maze with Manhattan and the neural heuristic.

Examples:
    python scripts/demo.py --list-samples
    python scripts/demo.py --case bad
    python scripts/demo.py --all-samples
    python scripts/demo.py --file mazes_perlin_20_10.npz --index 42 --backend torch \
        --mode clipped_manhattan --allow-untrained
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.heuristics import ManhattanHeuristic  # noqa: E402
from src.search import astar  # noqa: E402

SAMPLE_PATH = "results/analysis/sample_selections.json"
SKLEARN_PATH = "models/fallback_mlp_regressor.joblib"
TORCH_PATH = "models/neural_heuristic_seed42.pt"


def load_maze(file_name: str, index: int, cleaned_dir: str = "data/cleaned") -> dict[str, Any]:
    """Load one maze from a cleaned ``.npz`` file."""
    with np.load(os.path.join(cleaned_dir, file_name)) as data:
        return {
            "file_name": file_name,
            "local_index": int(index),
            "grid": data["grids"][index],
            "start": tuple(int(v) for v in data["starts"][index]),
            "goal": tuple(int(v) for v in data["goals"][index]),
            "h_star": data["h_star"][index],
            "opt_cost": float(data["opt_cost"][index]),
        }


def load_samples(path: str = SAMPLE_PATH) -> dict[str, Any]:
    """Return the curated good/typical/bad sample mazes."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle).get("selections", {})


def torch_checkpoint_status(path: str = TORCH_PATH) -> dict[str, Any]:
    """Minimal adequacy check mirroring ``scripts/run_torch_search.py``."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    train_config = checkpoint.get("config", {}).get("train", {})
    epochs = int(train_config.get("epochs", 0) or 0)
    cells = int(train_config.get("max_train_cells_per_epoch", 0) or 0)
    best_epoch = int(checkpoint.get("best_epoch", 0) or 0)
    val_mae = float(checkpoint.get("val_mae", float("nan")))
    reasons = []
    if epochs <= 1:
        reasons.append(f"configured epochs={epochs}")
    if best_epoch <= 1:
        reasons.append(f"best_epoch={best_epoch}")
    if 0 < cells <= 50000:
        reasons.append(f"max_train_cells_per_epoch={cells}")
    if np.isfinite(val_mae) and val_mae >= 0.15:
        reasons.append(f"val_mae={val_mae:.4f}")
    return {"under_trained": bool(reasons), "reasons": reasons, "val_mae": val_mae}


def build_heuristic(
    backend: str,
    maze: dict[str, Any],
    alpha: float,
    mode: str,
    allow_untrained: bool,
):
    """Instantiate the requested neural heuristic."""
    if backend == "sklearn":
        from src.sklearn_heuristic import SklearnNeuralHeuristic

        return SklearnNeuralHeuristic(SKLEARN_PATH, maze["grid"], maze["goal"], alpha=alpha, mode=mode)

    from src.heuristics import NeuralHeuristic

    status = torch_checkpoint_status(TORCH_PATH)
    if status["under_trained"] and not allow_untrained:
        raise SystemExit(
            "Torch checkpoint is an under-trained smoke test "
            f"({'; '.join(status['reasons'])}). Pass --allow-untrained to use it anyway."
        )
    return NeuralHeuristic(TORCH_PATH, maze["grid"], maze["goal"], alpha=alpha, mode=mode)


def field_diagnostics(heuristic: Any, maze: dict[str, Any]) -> dict[str, float]:
    """Compare the predicted field with the exact cost-to-go where available."""
    if hasattr(heuristic, "predictions"):
        predicted = np.asarray(heuristic.predictions(), dtype=float)
    elif hasattr(heuristic, "_pred_grid"):
        predicted = np.asarray(heuristic._pred_grid, dtype=float)
    else:
        return {}

    truth = np.asarray(maze["h_star"], dtype=float)
    free = (maze["grid"] == 0) & (truth >= 0) & np.isfinite(truth)
    if not free.any():
        return {}
    error = predicted[free] - truth[free]
    denominator = float(np.sum((truth[free] - truth[free].mean()) ** 2))
    return {
        "field_mae_cells": float(np.mean(np.abs(error))),
        "field_bias_cells": float(np.mean(error)),
        "field_overestimation_rate": float(np.mean(error > 0)),
        "field_r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
    }


def ascii_maze(
    maze: dict[str, Any],
    path: Optional[list[tuple[int, int]]] = None,
    expanded: Optional[list[tuple[int, int]]] = None,
    max_cells: int = 80,
) -> str:
    """Render a compact ASCII view of walls, path, and expanded cells."""
    grid = np.asarray(maze["grid"])
    n = grid.shape[0]
    view = grid[:max_cells, :max_cells].copy()
    chars = np.where(view == 1, "#", ".").astype("<U1")
    if expanded:
        for r, c in expanded:
            if 0 <= r < max_cells and 0 <= c < max_cells and chars[r, c] == ".":
                chars[r, c] = "+"
    if path:
        for r, c in path:
            if 0 <= r < max_cells and 0 <= c < max_cells:
                chars[r, c] = "*"
    sr, sc = maze["start"]
    gr, gc = maze["goal"]
    if sr < max_cells and sc < max_cells:
        chars[sr, sc] = "S"
    if gr < max_cells and gc < max_cells:
        chars[gr, gc] = "G"
    lines = ["    " + "".join(str(c % 10) for c in range(min(n, max_cells)))]
    for r in range(min(n, max_cells)):
        lines.append(f"{r:3d} " + "".join(chars[r]))
    if n > max_cells:
        lines.append(f"... {n - max_cells} rows/columns omitted")
    return "\n".join(lines)


def parse_maze(maze: dict[str, Any]) -> tuple[str, int, int]:
    return maze["file_name"], int(maze["local_index"]), int(maze["grid"].shape[0])


def describe(maze: dict[str, Any]) -> str:
    return f"{maze['file_name']}[{maze['local_index']}] {maze['grid'].shape[0]}x{maze['grid'].shape[1]}"


def run_case(
    maze: dict[str, Any],
    backend: str,
    alpha: float,
    mode: str,
    allow_untrained: bool,
    show_ascii: bool,
    show_path: bool,
) -> dict[str, Any]:
    """Solve one maze with Manhattan and the neural heuristic and print a report."""
    manhattan = astar(maze["grid"], maze["start"], maze["goal"], ManhattanHeuristic(maze["goal"]))
    neural_heuristic = build_heuristic(backend, maze, alpha, mode, allow_untrained)
    neural = astar(maze["grid"], maze["start"], maze["goal"], neural_heuristic)
    diagnostics = field_diagnostics(neural_heuristic, maze)

    rows = [
        ("manhattan", manhattan),
        (f"neural ({backend}, {mode}, alpha={alpha:g})", neural),
    ]
    print(f"\nMaze {describe(maze)}  start={maze['start']} goal={maze['goal']}")
    print(f"{'Heuristic':46s} {'Expanded':>9s} {'Cost':>6s} {'Optimal':>8s} {'Time ms':>9s}")
    for label, result in rows:
        ratio = result.cost / maze["opt_cost"] if result.found and maze["opt_cost"] > 0 else float("nan")
        print(
            f"{label:46s} {result.nodes_expanded:9d} {result.cost:6.1f} "
            f"{ratio:8.3f} {result.runtime_ms:9.2f}"
        )
    change = neural.nodes_expanded / manhattan.nodes_expanded if manhattan.nodes_expanded else float("nan")
    print(f"neural expansion vs Manhattan: {change:.2f}x")

    if diagnostics:
        print(
            "field vs h*: "
            f"MAE={diagnostics['field_mae_cells']:.2f} cells, "
            f"bias={diagnostics['field_bias_cells']:+.2f}, "
            f"overestimated={diagnostics['field_overestimation_rate'] * 100:.1f}%, "
            f"R2={diagnostics['field_r2']:.3f}"
        )

    if show_path:
        print("path:", " -> ".join(f"({r},{c})" for r, c in neural.path[:24]), end="")
        print(" -> ..." if len(neural.path) > 24 else "")
    if show_ascii:
        print()
        print(ascii_maze(maze, neural.path, neural.expanded))
        print("legend: # wall  . free  + expanded  * path  S start  G goal")

    return {
        "file_name": maze["file_name"],
        "local_index": maze["local_index"],
        "grid_size": int(maze["grid"].shape[0]),
        "backend": backend,
        "alpha": alpha,
        "mode": mode,
        "manhattan_nodes": manhattan.nodes_expanded,
        "neural_nodes": neural.nodes_expanded,
        "nodes_ratio": float(change),
        "manhattan_cost": manhattan.cost,
        "neural_cost": neural.cost,
        "manhattan_optimality_ratio": float(manhattan.cost / maze["opt_cost"]),
        "neural_optimality_ratio": float(neural.cost / maze["opt_cost"]),
        "manhattan_runtime_ms": manhattan.runtime_ms,
        "neural_runtime_ms": neural.runtime_ms,
        **diagnostics,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=None, help="cleaned .npz file name")
    parser.add_argument("--index", type=int, default=None, help="maze index inside the file")
    parser.add_argument("--case", choices=["good", "typical", "bad"], default=None)
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--list-samples", action="store_true")
    parser.add_argument("--backend", choices=["sklearn", "torch"], default="sklearn")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--mode", choices=["raw", "scaled", "clipped_manhattan"], default="raw")
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--cleaned-dir", default="data/cleaned")
    parser.add_argument("--samples", default=SAMPLE_PATH)
    parser.add_argument("--output", default="results/demo/demo.json")
    parser.add_argument("--no-ascii", action="store_true")
    parser.add_argument("--show-path", action="store_true")
    args = parser.parse_args(argv)

    samples = load_samples(args.samples)
    if args.list_samples:
        for name, entry in samples.items():
            print(
                f"{name:8s} {entry['file_name']}[{entry['local_index']}] "
                f"{entry['size']}x{entry['size']} {entry['generator']} density={entry['density']} "
                f"manhattan={entry['manhattan_nodes']:.0f} neural={entry['neural_nodes']:.0f}"
            )
        return 0

    cases: list[tuple[str, dict[str, Any]]] = []
    if args.all_samples:
        for name, entry in samples.items():
            maze = load_maze(entry["file_name"], int(entry["local_index"]), args.cleaned_dir)
            cases.append((name, maze))
    elif args.case:
        if args.case not in samples:
            parser.error(f"sample {args.case!r} not found in {args.samples}")
        entry = samples[args.case]
        cases.append(
            (
                args.case,
                load_maze(entry["file_name"], int(entry["local_index"]), args.cleaned_dir),
            )
        )
    elif args.file is not None and args.index is not None:
        cases.append(("custom", load_maze(args.file, args.index, args.cleaned_dir)))
    else:
        parser.error("choose --case, --all-samples, --list-samples, or --file/--index")

    records = []
    for name, maze in cases:
        if len(cases) > 1:
            print(f"\n=== {name} ===")
        records.append(
            run_case(
                maze,
                backend=args.backend,
                alpha=args.alpha,
                mode=args.mode,
                allow_untrained=args.allow_untrained,
                show_ascii=not args.no_ascii and len(cases) == 1,
                show_path=args.show_path,
            )
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"backend": args.backend, "cases": records}, handle, indent=2, default=float)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
