"""Run the PyTorch neural heuristic through A* and audit its checkpoint.

The repository ships two model artefacts. The scikit-learn fallback produced
the committed headline numbers; the PyTorch checkpoint was produced by a
single-epoch smoke run. This script therefore refuses to report search numbers
from an under-trained checkpoint unless ``--allow-untrained`` is passed, and
always writes an audit of the checkpoint metadata so the limitation is visible.

Outputs:
    results/reproducibility/torch_checkpoint_audit.json
    results/analysis/torch_search.json
    results/analysis/torch_search_per_maze.csv

Usage:
    python scripts/run_torch_search.py                      # audit only, refuses
    python scripts/run_torch_search.py --allow-untrained    # smoke-run search
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.heuristics import ManhattanHeuristic, NeuralHeuristic, OracleHeuristic  # noqa: E402
from src.search import astar  # noqa: E402

WORKING_THRESHOLD_MAE = 0.15
WORKING_THRESHOLD_CELLS = 50000


def audit_checkpoint(path: str) -> dict[str, Any]:
    """Inspect checkpoint metadata and classify its training adequacy."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    train_config = checkpoint.get("config", {}).get("train", {})
    model_config = checkpoint.get("model_config", {})
    n_parameters = int(sum(value.numel() for value in checkpoint["model_state"].values()))

    epochs_configured = int(train_config.get("epochs", 0) or 0)
    cells_per_epoch = int(train_config.get("max_train_cells_per_epoch", 0) or 0)
    best_epoch = int(checkpoint.get("best_epoch", 0) or 0)
    val_mae = float(checkpoint.get("val_mae", float("nan")))

    reasons = []
    if epochs_configured <= 1:
        reasons.append(f"configured epochs={epochs_configured}")
    if best_epoch <= 1:
        reasons.append(f"best_epoch={best_epoch}")
    if 0 < cells_per_epoch <= WORKING_THRESHOLD_CELLS:
        reasons.append(f"max_train_cells_per_epoch={cells_per_epoch}")
    if np.isfinite(val_mae) and val_mae >= WORKING_THRESHOLD_MAE:
        reasons.append(f"val_mae={val_mae:.4f}")

    return {
        "checkpoint": os.path.abspath(path),
        "architecture": model_config,
        "parameters": n_parameters,
        "seed": checkpoint.get("seed"),
        "git_commit": checkpoint.get("git_commit"),
        "configured_epochs": epochs_configured,
        "max_train_cells_per_epoch": cells_per_epoch,
        "best_epoch": best_epoch,
        "val_mae": val_mae,
        "target_mean": checkpoint.get("target_mean"),
        "target_std": checkpoint.get("target_std"),
        "under_trained": bool(reasons),
        "reasons": reasons,
        "recommendation": (
            "Retrain with `python -m src.train --epochs 30 --max-train-cells-per-epoch 0` "
            "before quoting search numbers, or quote the scikit-learn fallback results."
            if reasons
            else "Checkpoint looks trained enough for a benchmark."
        ),
    }


def _manifest_entries(manifest_path: str, split: str = "test") -> list[dict[str, Any]]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if "splits" in manifest:
        entries = manifest["splits"][split]["entries"]
        return [
            {"file": entry.get("file", entry.get("file_name")), "local_index": entry["local_index"]}
            for entry in entries
        ]
    return [
        {"file": entry.get("file", entry.get("file_name")), "local_index": entry["local_index"]}
        for entry in manifest.get(split, [])
    ]


def load_mazes(
    manifest_path: str,
    cleaned_dir: str,
    count: int = 0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load test mazes from the manifest, optionally subsampling."""
    entries = _manifest_entries(manifest_path, "test")
    if count > 0 and count < len(entries):
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(len(entries), size=count, replace=False))
        entries = [entries[i] for i in selected]

    mazes = []
    for entry in entries:
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


def run_search(
    mazes: Sequence[dict[str, Any]],
    checkpoint_path: str,
    alphas: Sequence[float] = (0.6, 0.7, 0.8, 0.9, 1.0),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the torch heuristic modes against baselines on the same mazes."""
    records: list[dict[str, Any]] = []
    for position, maze in enumerate(mazes):
        grid, start, goal = maze["grid"], maze["start"], maze["goal"]
        baselines = {
            "manhattan": ManhattanHeuristic(goal),
            "oracle": OracleHeuristic(maze["h_star"]),
        }

        variants: dict[str, Any] = {}
        raw = NeuralHeuristic(checkpoint_path, grid, goal, alpha=1.0, mode="raw")
        variants["torch_raw"] = raw
        for alpha in alphas:
            variants[f"torch_alpha_{alpha:g}"] = NeuralHeuristic(
                checkpoint_path, grid, goal, alpha=float(alpha), mode="scaled"
            )
        variants["torch_clipped"] = NeuralHeuristic(
            checkpoint_path, grid, goal, alpha=1.0, mode="clipped_manhattan"
        )

        results = {}
        for name, heuristic in baselines.items():
            results[name] = astar(grid, start, goal, heuristic)
        for name, heuristic in variants.items():
            results[name] = astar(grid, start, goal, heuristic)

        for name, result in results.items():
            records.append(
                {
                    "file_name": maze["file_name"],
                    "local_index": maze["local_index"],
                    "grid_size": int(grid.shape[0]),
                    "opt_cost": maze["opt_cost"],
                    "heuristic": name,
                    "found": bool(result.found),
                    "path_cost": float(result.cost),
                    "nodes_expanded": int(result.nodes_expanded),
                    "runtime_s": float(result.runtime_ms / 1000.0),
                    "optimality_ratio": (
                        float(result.cost / maze["opt_cost"])
                        if result.found and maze["opt_cost"] > 0
                        else float("nan")
                    ),
                }
            )
        if (position + 1) % 10 == 0 or position + 1 == len(mazes):
            print(f"searched {position + 1}/{len(mazes)} mazes")

    table = pd.DataFrame(records)
    manhattan_nodes = table.loc[table["heuristic"] == "manhattan", "nodes_expanded"].mean()
    summary: dict[str, Any] = {"mazes": len(mazes)}
    for name, group in table.groupby("heuristic"):
        nodes = group["nodes_expanded"].astype(float)
        summary[name] = {
            "mean_nodes_expanded": float(nodes.mean()),
            "median_nodes_expanded": float(np.median(nodes)),
            "mean_optimality_ratio": float(group["optimality_ratio"].mean()),
            "success_rate": float(group["found"].mean()),
            "mean_runtime_ms": float(group["runtime_s"].mean() * 1000.0),
            "relative_to_manhattan": float(nodes.mean() / manhattan_nodes),
        }
    return table, summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/neural_heuristic_seed42.pt")
    parser.add_argument("--manifest", default="results/manifests/split_manifest.json")
    parser.add_argument("--cleaned-dir", default="data/cleaned")
    parser.add_argument("--mazes", type=int, default=48, help="0 uses all test mazes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-dir", default="results/reproducibility")
    parser.add_argument("--output-dir", default="results/analysis")
    args = parser.parse_args(argv)

    os.makedirs(args.audit_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    audit = audit_checkpoint(args.checkpoint)
    audit_path = os.path.join(args.audit_dir, "torch_checkpoint_audit.json")
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=float)

    print(f"checkpoint: {audit['checkpoint']}")
    print(
        f"best_epoch={audit['best_epoch']} configured_epochs={audit['configured_epochs']} "
        f"cells/epoch={audit['max_train_cells_per_epoch']} val_mae={audit['val_mae']:.4f}"
    )
    if audit["under_trained"]:
        print("WARNING: checkpoint is a smoke-test artifact:")
        for reason in audit["reasons"]:
            print(f"  - {reason}")
        print(f"  {audit['recommendation']}")
    print(f"wrote {audit_path}")

    if args.audit_only:
        return 0
    if audit["under_trained"] and not args.allow_untrained:
        print("Refusing to run a benchmark from an under-trained checkpoint.")
        print("Pass --allow-untrained to run a smoke benchmark anyway.")
        return 2

    mazes = load_mazes(args.manifest, args.cleaned_dir, args.mazes, args.seed)
    table, summary = run_search(mazes, args.checkpoint)
    summary["checkpoint_audit"] = audit
    summary["allow_untrained"] = bool(args.allow_untrained)
    summary["seed"] = args.seed

    csv_path = os.path.join(args.output_dir, "torch_search_per_maze.csv")
    json_path = os.path.join(args.output_dir, "torch_search.json")
    table.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)

    for name in ("manhattan", "torch_raw", "torch_alpha_0.8", "torch_clipped", "oracle"):
        if name in summary:
            entry = summary[name]
            print(
                f"{name:16s} nodes={entry['mean_nodes_expanded']:7.1f} "
                f"({entry['relative_to_manhattan']:.2f}x) "
                f"opt={entry['mean_optimality_ratio']:.3f}"
            )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
