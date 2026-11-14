"""Generate the Phase 5 LaTeX tables and numeric macros from analysis outputs.

Reads:
    results/analysis/failure_summary.json
    results/analysis/failure_by_group.csv
    results/analysis/feature_ablation.json
    results/analysis/torch_search.json
    results/reproducibility/torch_checkpoint_audit.json

Writes:
    report/generated/phase5_comparison_table.tex
    report/generated/phase5_consistency_table.tex
    report/generated/phase5_strata_table.tex
    report/generated/phase5_macros.tex
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Optional

import pandas as pd

GROUP_LABELS = {
    ("generator", "backtracker"): "Generator: backtracker",
    ("generator", "perlin"): "Generator: Perlin",
    ("size", 20): "Size: 20",
    ("size", 40): "Size: 40",
    ("size", 60): "Size: 60",
    ("size", 80): "Size: 80",
    ("size", 100): "Size: 100",
    ("density", 10): "Density: 10\\%",
    ("density", 20): "Density: 20\\%",
    ("density", 30): "Density: 30\\%",
    ("density", 40): "Density: 40\\%",
    ("tortuosity_stratum", "direct (1.00)"): "Tortuosity: direct",
    ("tortuosity_stratum", "mild (<1.25)"): "Tortuosity: mild",
    ("tortuosity_stratum", "moderate (<1.50)"): "Tortuosity: moderate",
    ("tortuosity_stratum", "severe (>=1.50)"): "Tortuosity: severe",
}


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _p(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return f"{mantissa:.2f}\\times10^{{{exponent}}}"


def _comparison_row(label: str, entry: dict[str, Any]) -> str:
    wins = entry["treatment_wins"]
    ties = entry["ties"]
    losses = entry["treatment_losses"]
    return " & ".join(
        [
            label,
            f"{entry['baseline_mean']:.1f}",
            f"{entry['treatment_mean']:.1f}",
            f"{entry['relative_change'] * 100:+.1f}\\%",
            f"[{entry['mean_difference_ci_low']:.1f}, {entry['mean_difference_ci_high']:.1f}]",
            f"${_p(entry['wilcoxon_p_value'])}$",
            f"{wins}/{ties}/{losses}",
        ]
    ) + r" \\"


def write_comparison_table(summary: dict[str, Any], out_dir: str) -> str:
    rows = [
        _comparison_row("Raw neural vs Manhattan", summary["raw_vs_manhattan"]),
        _comparison_row("Clipped neural vs Manhattan", summary["clipped_vs_manhattan"]),
        _comparison_row("Raw neural vs clipped", summary["raw_vs_clipped"]),
    ]
    tex = (
        r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Comparison} & \textbf{Base} & \textbf{Treat.} & \textbf{Change} & \textbf{95\% CI} & \textbf{$p$} & \textbf{W/T/L} \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\caption{Paired node-expansion comparisons over the 282 test mazes. Change is
relative to the baseline; W/T/L counts mazes where the treatment expands fewer,
the same, or more nodes. $p$ is the two-sided Wilcoxon signed-rank test.}
\label{tab:phase5_comparison}
\end{table}
"""
    )
    path = os.path.join(out_dir, "phase5_comparison_table.tex")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(tex)
    return path


def write_consistency_table(summary: dict[str, Any], out_dir: str) -> str:
    data = summary["consistency_and_bias"]
    rates = data["consistency_violation_rate"]
    raw = data["raw_field_summary"][0]
    clipped = data["clipped_field_summary"][0]
    rows = [
        f"Manhattan & {rates['manhattan'] * 100:.1f}\\% & 0.00 & 0.00 & 0.0\\% & 100.0\\% \\\\",
        f"Raw neural & {rates['neural_alpha_1.0'] * 100:.1f}\\% & {raw['mae_cells']:.2f} & {raw['mean_signed_error_cells']:+.2f} & {raw['overestimation_rate'] * 100:.1f}\\% & {raw['within_one_cell_rate'] * 100:.1f}\\% \\\\",
        f"Clipped neural & {rates['neural_clipped_manhattan'] * 100:.1f}\\% & {clipped['mae_cells']:.2f} & {clipped['mean_signed_error_cells']:+.2f} & {clipped['overestimation_rate'] * 100:.1f}\\% & {clipped['within_one_cell_rate'] * 100:.1f}\\% \\\\",
    ]
    tex = (
        r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Field} & \textbf{Incons.} & \textbf{MAE} & \textbf{Bias} & \textbf{Overest.} & \textbf{Within 1} \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\caption{Field-level diagnostics pooled over 48 test mazes and
"""
        + f"{data['raw_field_summary'][0]['n_cells']:,}"
        + r""" sampled free cells. Inconsistency is the fraction of adjacent free
cell pairs with $|h(u)-h(v)|>1$; error columns are in grid cells.}
\label{tab:phase5_consistency}
\end{table}
"""
    )
    path = os.path.join(out_dir, "phase5_consistency_table.tex")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(tex)
    return path


def write_strata_table(groups: pd.DataFrame, out_dir: str) -> str:
    rows = []
    for _, row in groups.iterrows():
        value = row["value"]
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value)
        key = (row["group"], value)
        label = GROUP_LABELS.get(key, f"{row['group']}: {row['value']}")
        rows.append(
            " & ".join(
                [
                    label,
                    str(int(row["n_mazes"])),
                    f"{row['manhattan_nodes']:.1f}",
                    f"{row['neural_nodes']:.1f}",
                    f"{row['neural_ratio']:.2f}$\\times$",
                    f"{row['clipped_ratio']:.2f}$\\times$",
                    f"{row['neural_win_rate'] * 100:.1f}\\%",
                ]
            )
            + r" \\"
        )
    tex = (
        r"""\begin{table}[H]
\centering
\small
\begin{tabular}{lrccccc}
\toprule
\textbf{Stratum} & \textbf{Mazes} & \textbf{Man.} & \textbf{Neural} & \textbf{Ratio} & \textbf{Clipped} & \textbf{Win \%} \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\caption{Mean A$^\ast$ expansions by environment stratum for the raw neural
heuristic. ``Ratio'' is neural/Manhattan; ``Clipped'' is the clipped variant's
ratio; ``Win \%'' is the fraction of mazes where raw neural expands fewer nodes
than Manhattan. The failure is largest on the easiest, most direct mazes.}
\label{tab:phase5_strata}
\end{table}
"""
    )
    path = os.path.join(out_dir, "phase5_strata_table.tex")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(tex)
    return path


def build_macros(
    summary: dict[str, Any],
    groups: pd.DataFrame,
    ablation: dict[str, Any],
    torch_search: Optional[dict[str, Any]],
    torch_audit: Optional[dict[str, Any]],
    regression: Optional[dict[str, Any]] = None,
    classification: Optional[dict[str, Any]] = None,
    test_summary: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    raw = summary["raw_vs_manhattan"]
    clipped = summary["clipped_vs_manhattan"]
    raw_clipped = summary["raw_vs_clipped"]
    field = summary["consistency_and_bias"]
    raw_field = field["raw_field_summary"][0]
    clipped_field = field["clipped_field_summary"][0]
    dead = {row["stratum"]: row for row in field["dead_end_vs_other"]}
    walls = {row["neighbour_walls"]: row for row in field["error_by_local_walls"]}
    distance = field["error_by_true_distance"]
    raw_distance = [row for row in distance if row["variant"] == "raw"]
    clipped_distance = [row for row in distance if row["variant"] == "clipped"]

    def group_ratio(group: str, value: Any) -> float:
        match = groups[
            (groups["group"] == group) & (groups["value"].astype(str) == str(value))
        ]
        if match.empty:
            return float("nan")
        return float(match.iloc[0]["neural_ratio"])

    ab = {row["variant"]: row for row in ablation["variables"]} if "variables" in ablation else {
        row["variant"]: row for row in ablation["variants"]
    }

    macros = {
        "PfiveRawNodes": f"{raw['treatment_mean']:.1f}",
        "PfiveRawRelative": f"{raw['relative_change'] * 100:+.1f}",
        "PfiveRawWins": f"{raw['win_rate'] * 100:.1f}",
        "PfiveRawP": _p(raw["wilcoxon_p_value"]),
        "PfiveRawCiLow": f"{raw['mean_difference_ci_low']:.1f}",
        "PfiveRawCiHigh": f"{raw['mean_difference_ci_high']:.1f}",
        "PfiveClipNodes": f"{clipped['treatment_mean']:.1f}",
        "PfiveClipRelative": f"{clipped['relative_change'] * 100:+.1f}",
        "PfiveClipWins": f"{clipped['win_rate'] * 100:.1f}",
        "PfiveClipP": _p(clipped["wilcoxon_p_value"]),
        "PfiveClipCiLow": f"{clipped['mean_difference_ci_low']:.1f}",
        "PfiveClipCiHigh": f"{clipped['mean_difference_ci_high']:.1f}",
        "PfiveRawVsClipRelative": f"{raw_clipped['relative_change'] * 100:+.1f}",
        "PfiveManhattanNodes": f"{raw['baseline_mean']:.1f}",
        "PfiveConsistencyRaw": f"{field['consistency_violation_rate']['neural_alpha_1.0'] * 100:.1f}",
        "PfiveConsistencyClipped": f"{field['consistency_violation_rate']['neural_clipped_manhattan'] * 100:.1f}",
        "PfiveFieldRawMae": f"{raw_field['mae_cells']:.2f}",
        "PfiveFieldRawBias": f"{raw_field['mean_signed_error_cells']:+.2f}",
        "PfiveFieldRawOver": f"{raw_field['overestimation_rate'] * 100:.1f}",
        "PfiveFieldRawWithin": f"{raw_field['within_one_cell_rate'] * 100:.1f}",
        "PfiveFieldClipMae": f"{clipped_field['mae_cells']:.2f}",
        "PfiveFieldClipBias": f"{clipped_field['mean_signed_error_cells']:+.2f}",
        "PfiveFieldClipUnder": f"{clipped_field['underestimation_rate'] * 100:.1f}",
        "PfiveRawWinsCount": f"{raw['treatment_wins']}",
        "PfiveRawTies": f"{raw['ties']}",
        "PfiveRawLosses": f"{raw['treatment_losses']}",
        "PfiveClipWinsCount": f"{clipped['treatment_wins']}",
        "PfiveClipTies": f"{clipped['ties']}",
        "PfiveClipLosses": f"{clipped['treatment_losses']}",
        "PfivePerMazeReduction": f"{(raw['mean_difference'] / raw['baseline_mean']):.2f}",
        "PfiveDeadMae": f"{dead['dead_end']['mae_cells']:.2f}",
        "PfiveOtherMae": f"{dead['other_free']['mae_cells']:.2f}",
        "PfiveDeadBias": f"{dead['dead_end']['mean_signed_error_cells']:+.2f}",
        "PfiveOtherBias": f"{dead['other_free']['mean_signed_error_cells']:+.2f}",
        "PfiveDeadOver": f"{dead['dead_end']['overestimation_rate'] * 100:.1f}",
        "PfiveOtherOver": f"{dead['other_free']['overestimation_rate'] * 100:.1f}",
        "PfiveWallMaeLow": f"{walls[0]['mae_cells']:.2f}",
        "PfiveWallMaeHigh": f"{walls[max(walls)]['mae_cells']:.2f}",
        "PfiveWallBiasLow": f"{walls[0]['mean_signed_error_cells']:+.2f}",
        "PfiveWallBiasHigh": f"{walls[max(walls)]['mean_signed_error_cells']:+.2f}",
        "PfiveDistanceNearBias": f"{raw_distance[0]['mean_signed_error_cells']:+.2f}",
        "PfiveDistanceFarBias": f"{raw_distance[-1]['mean_signed_error_cells']:+.2f}",
        "PfiveClipDistanceNear": f"{clipped_distance[0]['mean_signed_error_cells']:+.2f}",
        "PfiveClipDistanceFar": f"{clipped_distance[-1]['mean_signed_error_cells']:+.2f}",
        "PfiveDirectFraction": f"{summary['tortuosity_direct_fraction'] * 100:.1f}",
        "PfiveTortSpearman": f"{summary['correlations']['tortuosity']['spearman_vs_log_ratio']:.3f}",
        "PfiveDensityTenRatio": f"{group_ratio('density', 10):.2f}",
        "PfiveDensityFortyRatio": f"{group_ratio('density', 40):.2f}",
        "PfiveSizeTwentyRatio": f"{group_ratio('size', 20):.2f}",
        "PfiveSizeSixtyRatio": f"{group_ratio('size', 60):.2f}",
        "PfiveSizeHundredRatio": f"{group_ratio('size', 100):.2f}",
        "PfiveBacktrackerRatio": f"{group_ratio('generator', 'backtracker'):.2f}",
        "PfivePerlinRatio": f"{group_ratio('generator', 'perlin'):.2f}",
        "PfiveDirectRatio": f"{group_ratio('tortuosity_stratum', 'direct (1.00)'):.2f}",
        "PfiveMildRatio": f"{group_ratio('tortuosity_stratum', 'mild (<1.25)'):.2f}",
        "PfiveModerateRatio": f"{group_ratio('tortuosity_stratum', 'moderate (<1.50)'):.2f}",
        "PfiveSevereRatio": f"{group_ratio('tortuosity_stratum', 'severe (>=1.50)'):.2f}",
        "PfiveAblAllRtwo": f"{ab['all']['r2']:.4f}",
        "PfiveAblAllNodes": f"{ab['all']['mean_nodes_expanded']:.1f}",
        "PfiveAblNoGoalRtwo": f"{ab['no_goal_offsets']['r2']:.4f}",
        "PfiveAblGoalOnlyRtwo": f"{ab['goal_offsets_only']['r2']:.4f}",
        "PfiveAblNoWallRtwo": f"{ab['no_wall_window']['r2']:.4f}",
        "PfiveAblGoalOnlyNodes": f"{ab['goal_offsets_only']['mean_nodes_expanded']:.1f}",
        "PfiveAblManhattanNodes": f"{ablation['manhattan_mean_nodes']:.1f}",
        "PfiveAblMazes": f"{int(ablation['search_mazes'])}",
    }
    if regression:
        macros.update(
            {
                "PfiveRtwo": f"{regression['r2']:.4f}",
                "PfiveMae": f"{regression['mae']:.4f}",
                "PfiveMaeCells": f"{regression['mae_cells']:.2f}",
                "PfiveRmseCells": f"{regression['rmse_cells']:.2f}",
                "PfivePearson": f"{regression['pearson']:.3f}",
                "PfiveSpearman": f"{regression['spearman']:.3f}",
                "PfiveMaxAbsCells": f"{regression['max_absolute_error_cells']:.1f}",
                "PfiveSignedBias": f"{regression['mean_signed_error']:+.4f}",
                "PfiveSignedBiasCells": f"{regression['mean_signed_error_cells']:+.2f}",
            }
        )
    if classification:
        macros.update(
            {
                "PfiveAuc": f"{classification['auc_roc_ovr_macro']:.4f}",
                "PfiveMacroFone": f"{classification['macro_f1']:.4f}",
            }
        )
    if test_summary:
        macros.update(
            {
                "PfiveManhattanMedian": f"{test_summary['manhattan']['median_nodes_expanded']:.0f}",
                "PfiveRawMedian": f"{test_summary['neural_alpha_1.0']['median_nodes_expanded']:.0f}",
                "PfiveClipMedian": f"{test_summary['neural_clipped_manhattan']['median_nodes_expanded']:.0f}",
                "PfiveRawRuntime": f"{test_summary['neural_alpha_1.0']['mean_runtime_ms']:.2f}",
                "PfiveClipRuntime": f"{test_summary['neural_clipped_manhattan']['mean_runtime_ms']:.2f}",
                "PfiveOracleNodes": f"{test_summary['oracle']['mean_nodes_expanded']:.1f}",
                "PfiveAlphaSixNodes": f"{test_summary['neural_alpha_0.6']['mean_nodes_expanded']:.1f}",
                "PfiveAlphaOneOpt": f"{test_summary['neural_alpha_1.0']['mean_optimality_ratio']:.3f}",
            }
        )
    if torch_search:
        torch_raw = torch_search.get("torch_raw", {})
        macros.update(
            {
                "PfiveTorchNodes": f"{torch_raw.get('mean_nodes_expanded', float('nan')):.1f}",
                "PfiveTorchRelative": f"{torch_raw.get('relative_to_manhattan', float('nan')):.2f}",
            }
        )
    if torch_audit:
        macros.update(
            {
                "PfiveTorchValMae": f"{torch_audit.get('val_mae', float('nan')):.4f}",
                "PfiveTorchEpochs": f"{int(torch_audit.get('configured_epochs', 0))}",
                "PfiveTorchCells": f"{int(torch_audit.get('max_train_cells_per_epoch', 0)):,}",
            }
        )
    return macros


def write_macros(macros: dict[str, str], out_dir: str) -> str:
    tex = "% Generated by scripts/make_phase5_report_assets.py -- do not edit.\n"
    for name, value in macros.items():
        tex += f"\\newcommand{{\\{name}}}{{{value}}}\n"
    path = os.path.join(out_dir, "phase5_macros.tex")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(tex)
    return path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default="results/analysis")
    parser.add_argument("--audit-dir", default="results/reproducibility")
    parser.add_argument("--eval-dir", default="results/evaluation")
    parser.add_argument("--output-dir", default="report/generated")
    args = parser.parse_args(argv)

    summary = load_json(os.path.join(args.analysis_dir, "failure_summary.json"))
    groups = pd.read_csv(os.path.join(args.analysis_dir, "failure_by_group.csv"))
    ablation = load_json(os.path.join(args.analysis_dir, "feature_ablation.json"))
    torch_search_path = os.path.join(args.analysis_dir, "torch_search.json")
    torch_search = load_json(torch_search_path) if os.path.exists(torch_search_path) else None
    audit_path = os.path.join(args.audit_dir, "torch_checkpoint_audit.json")
    torch_audit = load_json(audit_path) if os.path.exists(audit_path) else None

    def optional(path: str) -> Optional[dict[str, Any]]:
        return load_json(path) if os.path.exists(path) else None

    regression = optional(os.path.join(args.eval_dir, "regression_metrics.json"))
    classification = optional(os.path.join(args.eval_dir, "classification_metrics.json"))
    test_summary = optional(os.path.join(args.eval_dir, "test_summary.json"))

    os.makedirs(args.output_dir, exist_ok=True)
    paths = [
        write_comparison_table(summary, args.output_dir),
        write_consistency_table(summary, args.output_dir),
        write_strata_table(groups, args.output_dir),
        write_macros(
            build_macros(
                summary,
                groups,
                ablation,
                torch_search,
                torch_audit,
                regression,
                classification,
                test_summary,
            ),
            args.output_dir,
        ),
    ]
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
