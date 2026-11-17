"""Summarise newly generated D3 feature ablations without modifying D2 assets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='results/analysis/d3/feature_ablation.csv')
    p.add_argument('--output', default='results/analysis/d3/ablation_summary.json')
    a = p.parse_args(argv)
    df = pd.read_csv(a.input)
    cols = ['variant', 'n_features', 'mae_normalized', 'r2']
    if not set(cols).issubset(df):
        raise ValueError(f'Ablation data must include {cols}')
    full = df.loc[df.variant == 'full_29']
    if len(full) != 1:
        raise ValueError('Expected exactly one full_29 baseline')
    ref = full.iloc[0]
    rows = []
    for _, r in df.iterrows():
        entry = {'variant': r['variant'], 'n_features': int(r['n_features']),
                 'mae_normalized': float(r['mae_normalized']),
                 'mae_change_vs_full': float(r['mae_normalized'] - ref['mae_normalized']),
                 'r2': float(r['r2'])}
        if 'neural_mean_nodes' in df and pd.notna(r.get('neural_mean_nodes')):
            entry['neural_mean_nodes'] = float(r['neural_mean_nodes'])
        rows.append(entry)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'feature_variants': rows}, indent=2), encoding='utf-8')
    print(f'Created {output}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
