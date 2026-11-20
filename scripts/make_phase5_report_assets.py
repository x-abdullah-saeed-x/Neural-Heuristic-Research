"""Generate D3-only LaTeX tables/macros from fresh D3 analysis; never overwrite D2 tables.

Usage: python scripts/make_phase5_report_assets.py
Requires results/analysis/d3/paired_comparisons.csv; ablation input is optional.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def esc(value):
    return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def latex_table(df: pd.DataFrame, columns: list[tuple[str, str]], label: str,
                caption: str) -> str:
    lines = [r'\begin{table}[ht]', r'\centering', r'\small',
             r'\begin{tabular}{' + 'l' + 'r' * (len(columns) - 1) + '}',
             r'\toprule', ' & '.join(header for _, header in columns) + r' \\', r'\midrule']
    for _, row in df.iterrows():
        vals = []
        for key, _ in columns:
            val = row[key]
            vals.append(esc(val) if isinstance(val, str) else
                        (str(int(val)) if key in {'n_matched_mazes','n_features'} else f'{float(val):.3f}'))
        lines.append(' & '.join(vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\caption{' + caption + '}',
              r'\label{' + label + '}', r'\end{table}', '']
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', default='results/analysis/d3')
    p.add_argument('--output-dir', default='report/generated/d3')
    a = p.parse_args(argv)
    src, dest = Path(a.input_dir), Path(a.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    compare = pd.read_csv(src / 'paired_comparisons.csv')
    columns = [('treatment', 'Heuristic'), ('n_matched_mazes', 'Mazes'),
               ('mean_node_difference_treatment_minus_baseline', 'Mean $\\Delta$ nodes'),
               ('mean_expansion_ratio', 'Node ratio')]
    text = latex_table(compare, columns, 'tab:d3_paired',
                       'Paired comparisons against Manhattan A*, calculated on matched test mazes.')
    (dest / 'paired_table.tex').write_text(text, encoding='utf-8')
    ablation = src / 'feature_ablation.csv'
    if ablation.is_file():
        ab = pd.read_csv(ablation)
        cols = [('variant', 'Variant'), ('n_features', 'Features'),
                ('mae_normalized', 'MAE'), ('r2', '$R^2$')]
        text = latex_table(ab, cols, 'tab:d3_features',
                           'Fresh feature-ablation results from the original D2 data split.')
        (dest / 'feature_table.tex').write_text(text, encoding='utf-8')
    print(f'Generated D3-only LaTeX files in {dest}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
