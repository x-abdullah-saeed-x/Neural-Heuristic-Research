"""Combine the Deliverable 3 ablations into one summary and LaTeX assets.

Inputs:
    results/analysis/feature_ablation.csv/json   (run_feature_ablation.py)
    results/evaluation/test_summary.json         (search-time variants)
    results/evaluation/alpha_sweep.json          (alpha scaling sweep)

Outputs:
    results/analysis/ablation_summary.json
    report/generated/ablation_table.tex
    report/generated/search_mode_table.tex
    report/generated/ablation_macros.tex
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DISPLAY_NAMES = {
    "all": "All 29 features",
    "no_coordinates": "Without coordinates",
    "no_goal_offsets": "Without goal offsets",
    "no_wall_window": "Without $5\\times5$ wall window",
    "goal_offsets_only": "Goal offsets only",
    "coordinates_only": "Coordinates only",
    "wall_window_only": "Wall window only",
}

SEARCH_MODES = [
    ("Raw neural ($\\alpha=1.0$)", "neural_alpha_1.0"),
    ("Scaled neural ($\\alpha=0.8$)", "neural_alpha_0.8"),
    ("Clipped to Manhattan", "neural_clipped_manhattan"),
    ("Manhattan baseline", "manhattan"),
    ("Oracle ($h^\\ast$)", "oracle"),
]


def load_feature_ablation(csv_path: str) -> pd.DataFrame:
    """Load and order the feature-ablation table."""
    table = pd.read_csv(csv_path)
    order = list(DISPLAY_NAMES)
    table["__order"] = table["variant"].map({name: i for i, name in enumerate(order)})
    return table.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def build_summary(
    feature_csv: str,
    feature_json: str,
    test_summary: str,
    alpha_sweep: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble the ablation summary from its component result files."""
    features = load_feature_ablation(feature_csv)
    with open(feature_json, "r", encoding="utf-8") as handle:
        feature_meta = json.load(handle)
    with open(test_summary, "r", encoding="utf-8") as handle:
        search_summary = json.load(handle)

    by_name = features.set_index("variant")
    all_row = by_name.loc["all"]

    contributions = []
    for name, row in by_name.iterrows():
        if name == "all":
            continue
        contributions.append(
            {
                "variant": name,
                "features_removed_or_kept": int(row["n_features"]),
                "r2": float(row["r2"]),
                "r2_change_vs_all": float(row["r2"] - all_row["r2"]),
                "mae": float(row["mae"]),
                "nodes_expanded": float(row["mean_nodes_expanded"]),
                "nodes_ratio_vs_all": float(row["mean_nodes_expanded"] / all_row["mean_nodes_expanded"]),
            }
        )
    contributions.sort(key=lambda item: item["r2_change_vs_all"])

    search_modes = []
    for label, key in SEARCH_MODES:
        if key not in search_summary:
            continue
        entry = search_summary[key]
        search_modes.append(
            {
                "label": label,
                "key": key,
                "mean_nodes_expanded": float(entry["mean_nodes_expanded"]),
                "median_nodes_expanded": float(entry["median_nodes_expanded"]),
                "mean_optimality_ratio": float(entry["mean_optimality_ratio"]),
                "success_rate": float(entry["success_rate"]),
                "mean_runtime_ms": float(entry["mean_runtime_ms"]),
                "relative_to_manhattan": float(
                    entry["mean_nodes_expanded"] / search_summary["manhattan"]["mean_nodes_expanded"]
                ),
            }
        )

    alpha_rows = []
    if alpha_sweep and os.path.exists(alpha_sweep):
        with open(alpha_sweep, "r", encoding="utf-8") as handle:
            alpha_data = json.load(handle)
        for alpha, entry in sorted(alpha_data.items(), key=lambda item: float(item[0])):
            alpha_rows.append(
                {
                    "alpha": float(alpha),
                    "mean_nodes_expanded": float(entry["mean_nodes_expanded"]),
                    "mean_optimality_ratio": float(entry["mean_optimality_ratio"]),
                    "relative_to_manhattan": float(
                        entry["mean_nodes_expanded"]
                        / search_summary["manhattan"]["mean_nodes_expanded"]
                    ),
                }
            )

    return {
        "feature_ablation": features.to_dict(orient="records"),
        "feature_meta": feature_meta,
        "contributions_ranked": contributions,
        "search_modes": search_modes,
        "alpha_sweep": alpha_rows,
        "headline": {
            "all_r2": float(all_row["r2"]),
            "all_nodes": float(all_row["mean_nodes_expanded"]),
            "no_goal_offsets_r2": float(by_name.loc["no_goal_offsets", "r2"]),
            "goal_offsets_only_r2": float(by_name.loc["goal_offsets_only", "r2"]),
            "no_wall_window_r2": float(by_name.loc["no_wall_window", "r2"]),
            "no_coordinates_r2": float(by_name.loc["no_coordinates", "r2"]),
            "goal_offsets_only_nodes": float(by_name.loc["goal_offsets_only", "mean_nodes_expanded"]),
            "manhattan_nodes": float(feature_meta["manhattan_mean_nodes"]),
        },
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_tables(summary: dict[str, Any], out_dir: str) -> list[str]:
    """Write the LaTeX tables and macros used by the final report."""
    os.makedirs(out_dir, exist_ok=True)
    written = []

    feature_rows = []
    for row in summary["feature_ablation"]:
        feature_rows.append(
            " & ".join(
                [
                    DISPLAY_NAMES.get(row["variant"], str(row["variant"])),
                    str(int(row["n_features"])),
                    _fmt(row["r2"]),
                    _fmt(row["mae"], 4),
                    _fmt(row["mae_cells_n60"], 2),
                    f"{row['mean_nodes_expanded']:.1f}",
                    f"{row['nodes_ratio_vs_manhattan']:.2f}",
                    f"{row['mean_optimality_ratio']:.3f}",
                ]
            )
            + r" \\"
        )
    feature_tex = (
        r"""\begin{table}[H]
\centering
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Feature set} & \textbf{Dim.} & \textbf{$R^2$} & \textbf{MAE} & \textbf{MAE (cells)} & \textbf{Nodes} & \textbf{vs Man.} & \textbf{Opt. ratio} \\
\midrule
"""
        + "\n".join(feature_rows)
        + r"""
\bottomrule
\end{tabular}}
\caption{Feature-group ablation. Each row retrains the same MLP on the stated
input subset and evaluates regression fidelity on the test cells and A$^\ast$
expansions on a common 24-maze test subset. ``MAE (cells)'' scales the
normalised MAE by $N=60$ for comparability with earlier tables.}
\label{tab:feature_ablation}
\end{table}
"""
    )
    path = os.path.join(out_dir, "ablation_table.tex")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(feature_tex)
    written.append(path)

    search_rows = []
    for row in summary["search_modes"]:
        search_rows.append(
            " & ".join(
                [
                    row["label"],
                    f"{row['mean_nodes_expanded']:.1f}",
                    f"{row['median_nodes_expanded']:.0f}",
                    f"{row['relative_to_manhattan']:.2f}$\\times$",
                    f"{row['mean_optimality_ratio']:.3f}",
                    f"{row['success_rate'] * 100:.1f}\\%",
                    f"{row['mean_runtime_ms']:.2f}",
                ]
            )
            + r" \\"
        )
    search_tex = (
        r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Search variant} & \textbf{Expanded} & \textbf{Median} & \textbf{vs Man.} & \textbf{Opt. ratio} & \textbf{Success} & \textbf{ms} \\
\midrule
"""
        + "\n".join(search_rows)
        + r"""
\bottomrule
\end{tabular}
\caption{Search-time ablation of the heuristic mode on all 282 test mazes.
Clipping costs optimality but restores admissibility; raw and scaled modes are
more aggressive but expand far more nodes.}
\label{tab:search_mode_ablation}
\end{table}
"""
    )
    path = os.path.join(out_dir, "search_mode_table.tex")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(search_tex)
    written.append(path)

    head = summary["headline"]
    macros = {
        "AblAllR2": _fmt(head["all_r2"]),
        "AblAllNodes": f"{head['all_nodes']:.1f}",
        "AblNoGoalR2": _fmt(head["no_goal_offsets_r2"]),
        "AblGoalOnlyR2": _fmt(head["goal_offsets_only_r2"]),
        "AblNoWallR2": _fmt(head["no_wall_window_r2"]),
        "AblNoCoordR2": _fmt(head["no_coordinates_r2"]),
        "AblGoalOnlyNodes": f"{head['goal_offsets_only_nodes']:.1f}",
        "AblSubsetManhattanNodes": f"{head['manhattan_nodes']:.1f}",
    }
    macro_tex = "% Generated by scripts/make_ablation_summary.py -- do not edit.\n"
    for name, value in macros.items():
        macro_tex += f"\\newcommand{{\\{name}}}{{{value}}}\n"
    path = os.path.join(out_dir, "ablation_macros.tex")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(macro_tex)
    written.append(path)
    return written


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-csv", default="results/analysis/feature_ablation.csv")
    parser.add_argument("--feature-json", default="results/analysis/feature_ablation.json")
    parser.add_argument("--test-summary", default="results/evaluation/test_summary.json")
    parser.add_argument("--alpha-sweep", default="results/evaluation/alpha_sweep.json")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--generated-dir", default="report/generated")
    args = parser.parse_args(argv)

    summary = build_summary(
        args.feature_csv, args.feature_json, args.test_summary, args.alpha_sweep
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "ablation_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)
    written = write_tables(summary, args.generated_dir)

    for item in summary["contributions_ranked"]:
        print(
            f"{item['variant']:20s} R2={item['r2']:.4f} "
            f"(delta {item['r2_change_vs_all']:+.4f}), nodes x{item['nodes_ratio_vs_all']:.2f}"
        )
    print("wrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
